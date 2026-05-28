"""
GDN-2 block for NaylisGDN.

Token mixer: GatedDeltaNet-2 (arXiv 2605.22791, Hatamizadeh et al. 2026 — NVIDIA).
Kernel priority:
  1. gdn2_ops.chunk_gdn2 / fused_recurrent_gdn2  — kernels Triton officiels NVIDIA (inclus dans le repo)
  2. PyTorch pur GDN-2                            — fallback si Triton indisponible

Outer block structure (identique au reste de NaylisGDN) :
  pre-norm → GDN-2 mixer → résidu → pre-norm → FFN → résidu
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from naylisgdn.norm import RMSNorm
from naylisgdn.feedforward import FeedForward

try:
    import transformer_engine.pytorch as te
    _TE_AVAILABLE = True
except ImportError:
    _TE_AVAILABLE = False

_GDN2_CHUNK     = None
_GDN2_RECURRENT = None

try:
    from naylisgdn.gdn2_ops import chunk_gdn2, fused_recurrent_gdn2
    _GDN2_CHUNK     = chunk_gdn2
    _GDN2_RECURRENT = fused_recurrent_gdn2
    print("  ⚡ GDN-2 : kernels Triton officiels NVIDIA (chunk + fused_recurrent)")
except Exception as _e:
    print(f"  ⚠️  GDN-2 kernels Triton indisponibles ({_e}) — fallback PyTorch")

GDNState = torch.Tensor


def _gdn2_torch(
    q:             torch.Tensor,
    k:             torch.Tensor,
    v:             torch.Tensor,
    g:             torch.Tensor,
    b:             torch.Tensor,
    w:             torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implémentation de référence PyTorch de la récurrence GDN-2.

    Shapes (head_first=False) :
        q, k  : [B, T, H, d_k]
        v     : [B, T, H, d_v]
        g     : [B, T, H, d_k]  — log-decay (négatif), ∈ (-∞, 0)  →  alpha = exp(g) ∈ (0,1)
        b     : [B, T, H, d_k]  — erase gate channel-wise (key-axis),   sigmoid ∈ (0,1)
        w     : [B, T, H, d_v]  — write gate channel-wise (value-axis), sigmoid ∈ (0,1)
        state : [B, H, d_k, d_v]

    Récurrence (arXiv 2605.22791) :
        S_t = diag(exp(g_t)) * S_{t-1}
            - k_t ⊗ ((b_t * k_t)ᵀ S_{t-1_decayed})     # EFFACER
            + k_t ⊗ (w_t * v_t)                          # ÉCRIRE
        y_t = S_t q_t
    """
    B, T, H, d_k = q.shape
    d_v = v.shape[-1]

    S = initial_state if initial_state is not None else \
        torch.zeros(B, H, d_k, d_v, device=q.device, dtype=q.dtype)

    k = F.normalize(k, p=2, dim=-1)

    outputs = []
    for t in range(T):
        q_t = q[:, t]           # [B, H, d_k]
        k_t = k[:, t]           # [B, H, d_k]
        v_t = v[:, t]           # [B, H, d_v]
        g_t = g[:, t]           # [B, H, d_k]
        b_t = b[:, t]           # [B, H, d_k]
        w_t = w[:, t]           # [B, H, d_v]

        alpha_t = g_t.exp()                                           # [B, H, d_k]
        S_dec   = alpha_t.unsqueeze(-1) * S                          # [B, H, d_k, d_v]

        gated_read = torch.einsum("bhd,bhd,bhdv->bhv", b_t, k_t, S_dec)  # [B, H, d_v]

        S = S_dec \
            - torch.einsum("bhd,bhv->bhdv", k_t, gated_read) \
            + torch.einsum("bhd,bhv->bhdv", k_t, w_t * v_t)

        y_t = torch.einsum("bhd,bhdv->bhv", q_t, S)                 # [B, H, d_v]
        outputs.append(y_t.unsqueeze(1))

    return torch.cat(outputs, dim=1), S


class GDNBlock(nn.Module):
    """
    Bloc NaylisGDN utilisant le token-mixer GatedDeltaNet-2 (GDN-2).

    Projections (fidèles au repo officiel NVlabs/GatedDeltaNet-2) :
      q_proj, k_proj, v_proj  — Q / K / V linéaires
      f_proj   [D → head_v_dim → key_dim]  — pre-activation du log-decay
      b_proj   [D → key_dim]               — erase gate (axe key, sigmoid)
      w_proj   [D → value_dim]             — write gate (axe value, sigmoid)
      g_proj   [D → head_v_dim → value_dim] — gate de sortie (SiLU)
      A_log    Paramètre [num_heads]        — taux de décroissance par tête
      dt_bias  Paramètre [key_dim]          — biais de pas de temps par canal
      o_proj   [value_dim → D]             — projection de sortie

    Interface forward → (x, out_state) identique au reste du modèle NaylisGDN.
    """

    def __init__(
        self,
        embed_dim:   int,
        num_heads:   int,
        dropout:     float = 0.0,
        use_swiglu:  bool  = True,
        use_fp8:     bool  = False,
        head_dim:    Optional[int]  = None,
        v_heads:     Optional[int]  = None,
        expand_v:    float = 1.0,
        use_short_conv: bool = False,
        allow_neg_eigval: bool = False,
    ):
        super().__init__()

        self.embed_dim   = embed_dim
        self.num_heads   = num_heads
        self.head_k_dim  = head_dim if head_dim is not None else (embed_dim // num_heads)
        self.num_v_heads = v_heads  if v_heads  is not None else num_heads
        self.expand_v    = expand_v
        self.use_fp8     = use_fp8 and _TE_AVAILABLE
        self.allow_neg_eigval = allow_neg_eigval

        assert self.num_v_heads % self.num_heads == 0, \
            f"v_heads ({self.num_v_heads}) doit être un multiple de num_heads ({self.num_heads})"

        self.head_v_dim = int(self.head_k_dim * self.expand_v)
        self.key_dim    = self.num_heads   * self.head_k_dim
        self.value_dim  = self.num_v_heads * self.head_v_dim

        Linear = te.Linear if self.use_fp8 else nn.Linear

        self.q_proj = Linear(embed_dim, self.key_dim,   bias=False)
        self.k_proj = Linear(embed_dim, self.key_dim,   bias=False)
        self.v_proj = Linear(embed_dim, self.value_dim, bias=False)

        self.f_proj = nn.Sequential(
            nn.Linear(embed_dim,      self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim,   bias=False),
        )
        self.b_proj = nn.Linear(embed_dim, self.key_dim,   bias=False)
        self.w_proj = nn.Linear(embed_dim, self.value_dim, bias=False)

        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16))
        )
        self.A_log._no_weight_decay = True

        dt = torch.exp(
            torch.rand(self.key_dim, dtype=torch.float32)
            * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        self.g_proj = nn.Sequential(
            nn.Linear(embed_dim,       self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim,  bias=True),
        )
        self.o_norm = RMSNorm(self.head_v_dim)
        self.o_proj = Linear(self.value_dim, embed_dim, bias=False)

        self.norm1   = RMSNorm(embed_dim)
        self.norm2   = RMSNorm(embed_dim)
        self.ffn     = FeedForward(embed_dim, dropout, use_swiglu=use_swiglu, use_fp8=use_fp8)
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if getattr(m, "_is_hf_initialized", False):
            return
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=2 ** -2.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        m._is_hf_initialized = True

    def _compute_log_decay(self, h: torch.Tensor) -> torch.Tensor:
        """Log-decay g ∈ (-∞, 0) — calculé en fp32 pour stabilité numérique.

        On exécute f_proj dans le dtype de h (bf16/fp32) puis on caste le
        résultat en fp32 — même ordre que le repo officiel NVlabs/GatedDeltaNet-2.
        Caster l'entrée en float() avant la projection causerait des incohérences
        de dtype sous FSDP MixedPrecision (poids bf16, entrée fp32).
        """
        A_exp = self.A_log.float().exp().repeat_interleave(self.head_k_dim)
        return -A_exp * F.softplus(self.f_proj(h).float() + self.dt_bias)

    def forward(
        self,
        x:               torch.Tensor,
        recurrent_state: Optional[GDNState] = None,
        use_recurrent:   bool = False,
        cu_seqlens:      Optional[torch.Tensor] = None,
        max_seqlen:      Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[GDNState]]:

        B, T, D = x.shape
        H, Hv   = self.num_heads, self.num_v_heads
        d_k, d_v = self.head_k_dim, self.head_v_dim

        residual = x
        h = self.norm1(x)
        if h.dtype != torch.bfloat16:
            h = h.to(torch.bfloat16)

        q = F.silu(self.q_proj(h)).view(B, T, H,  d_k)
        k = F.silu(self.k_proj(h)).view(B, T, H,  d_k)
        v = F.silu(self.v_proj(h)).view(B, T, Hv, d_v)

        g = self._compute_log_decay(h).to(h.dtype).view(B, T, H, d_k)
        b = self.b_proj(h).sigmoid().view(B, T, H, d_k)
        w = self.w_proj(h).sigmoid().view(B, T, Hv, d_v)

        if self.allow_neg_eigval:
            b = b * 2.0

        if Hv > H:
            from einops import repeat
            q = repeat(q, "b t h d -> b t (h g) d", g=Hv // H)
            k = repeat(k, "b t h d -> b t (h g) d", g=Hv // H)
            g = repeat(g, "b t h d -> b t (h g) d", g=Hv // H)
            b = repeat(b, "b t h d -> b t (h g) d", g=Hv // H)

        use_rec_mode = use_recurrent or (T == 1)
        new_state    = None

        if _GDN2_CHUNK is not None and _GDN2_RECURRENT is not None:
            if use_rec_mode:
                o, new_state = _GDN2_RECURRENT(
                    q=q, k=k, v=v, g=g, b=b, w=w,
                    initial_state      = recurrent_state,
                    output_final_state = True,
                    use_qk_l2norm_in_kernel = True,
                    use_gate_in_kernel      = False,
                    cu_seqlens = cu_seqlens,
                )
            else:
                o, new_state = _GDN2_CHUNK(
                    q=q, k=k, v=v, g=g, b=b, w=w,
                    initial_state      = recurrent_state,
                    output_final_state = False,
                    use_qk_l2norm_in_kernel = True,
                    use_gate_in_kernel      = False,
                    cu_seqlens = cu_seqlens,
                )
        else:
            o, new_state = _gdn2_torch(q, k, v, g, b, w, initial_state=recurrent_state)

        g_gate = self.g_proj(h).view(B, T, Hv, d_v)
        # o_norm (LigerRMSNorm ou PyTorch) attend du 2D [M, N] — on reshape
        # avant et on restitue la forme [B, T, Hv, d_v] après.
        o      = self.o_norm(o.reshape(-1, d_v)).view(B, T, Hv, d_v)
        o      = o * F.silu(g_gate)
        o      = o.contiguous().view(B, T, Hv * d_v)
        o      = self.o_proj(o)
        o      = self.dropout(o)

        x = residual + o
        x = x + self.ffn(self.norm2(x))

        out_state = new_state if use_rec_mode else None
        return x, out_state
