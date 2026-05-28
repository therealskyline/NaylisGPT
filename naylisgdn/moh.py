import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from naylisgdn.norm import RMSNorm
from naylisgdn.rope import RotaryPositionalEmbedding

try:
    import transformer_engine.pytorch as te
    _TE_AVAILABLE = True
except ImportError:
    _TE_AVAILABLE = False

try:
    from flash_attn.flash_attn_interface import flash_attn_func as _fa_func
    _FA_OK = True
except ImportError:
    _FA_OK = False

KVCache = Tuple[torch.Tensor, torch.Tensor]


class MixtureOfHeads(nn.Module):
    """
    Mixture-of-Heads attention avec head_dim découplé (style Qwen3.5).

    H têtes totales = shared_heads (toujours actives)
                    + routed_heads (top-k_routed sélectionnées par token).

    head_dim peut être découplé de embed_dim/num_heads.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        shared_heads: int,
        top_k_routed: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        rope_base: int = 10000,
        use_yarn: bool = False,
        yarn_scale: float = 1.0,
        yarn_original_max_len: int = 1024,
        n_kv_heads: Optional[int] = None,
        use_qk_norm: bool = False,
        soft_cap: Optional[float] = None,
        use_fp8: bool = False,
        head_dim: Optional[int] = None,
    ):
        super().__init__()
        routed_heads = num_heads - shared_heads
        assert 0 < top_k_routed <= routed_heads
        assert shared_heads > 0

        self.embed_dim    = embed_dim
        self.num_heads    = num_heads
        self.head_dim     = head_dim if head_dim is not None else (embed_dim // num_heads)
        self.qo_dim       = num_heads * self.head_dim
        self.shared_heads = shared_heads
        self.routed_heads = routed_heads
        self.top_k_routed = top_k_routed
        self.soft_cap     = soft_cap
        self.use_rope     = use_rope
        self.use_qk_norm  = use_qk_norm
        self.dropout_mod  = nn.Dropout(dropout)

        self.n_kv_heads       = n_kv_heads if n_kv_heads is not None else num_heads
        assert num_heads % self.n_kv_heads == 0
        self.n_queries_per_kv = num_heads // self.n_kv_heads
        self.kv_dim           = self.n_kv_heads * self.head_dim

        Linear = te.Linear if (use_fp8 and _TE_AVAILABLE) else nn.Linear

        self.q_proj      = Linear(embed_dim, self.qo_dim,  bias=False)
        self.k_proj      = Linear(embed_dim, self.kv_dim,  bias=False)
        self.v_proj      = Linear(embed_dim, self.kv_dim,  bias=False)
        self.out_proj    = Linear(self.qo_dim, embed_dim,  bias=False)
        self.head_router = nn.Linear(embed_dim, routed_heads, bias=False)

        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None

        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                self.head_dim, max_seq_len,
                rope_base=rope_base,
                use_yarn=use_yarn, yarn_scale=yarn_scale,
                yarn_original_max_len=yarn_original_max_len,
            )
        else:
            self.rope = None

    def _scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[KVCache] = None,
        use_kv_cache: bool = False,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:

        B, T, _ = x.shape
        H, d    = self.num_heads, self.head_dim
        Hs      = self.shared_heads

        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        q = self.q_proj(x).view(B, T, H,              d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, d).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        pos_off = past_kv[0].shape[2] if past_kv is not None else 0
        if self.use_rope and self.rope is not None:
            q, k = self.rope(q, k, position_offset=pos_off)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv: Optional[KVCache] = (k, v) if use_kv_cache else None

        if self.n_kv_heads != H:
            k = k.repeat_interleave(self.n_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.n_queries_per_kv, dim=1)

        scale     = self._scale()
        is_causal = T > 1 and past_kv is None

        if _FA_OK and self.soft_cap is None and mask is None:
            y_all = _fa_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=self.dropout_mod.p if self.training else 0.0,
                softmax_scale=scale, causal=is_causal,
            ).transpose(1, 2)
        elif hasattr(F, "scaled_dot_product_attention") and self.soft_cap is None and mask is None:
            y_all = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=is_causal,
                dropout_p=self.dropout_mod.p if self.training else 0.0, scale=scale,
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if self.soft_cap is not None:
                scores = self.soft_cap * torch.tanh(scores / self.soft_cap)
            if is_causal:
                cm     = torch.triu(torch.ones(T, k.shape[2], device=x.device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(cm.unsqueeze(0).unsqueeze(0), float("-inf"))
            y_all = torch.matmul(F.softmax(scores, dim=-1), v)

        route_logits = self.head_router(x)
        route_probs  = F.softmax(route_logits, dim=-1)
        top_probs, top_idx = torch.topk(route_probs, self.top_k_routed, dim=-1)
        top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-9)

        weights = torch.zeros(B, T, H, device=x.device, dtype=x.dtype)
        weights[:, :, :Hs] = 1.0
        weights[:, :, Hs:].scatter_(2, top_idx, top_probs)

        y_weighted = (y_all.transpose(1, 2) * weights.unsqueeze(-1)).contiguous().view(B, T, self.qo_dim)

        out = self.out_proj(y_weighted)
        out = self.dropout_mod(out)
        return out, new_kv
