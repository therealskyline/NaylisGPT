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

KVCache = Tuple[torch.Tensor, torch.Tensor]

_FA_LEVEL       = 0
_FA_VARLEN_FUNC = None
_FA_FUNC        = None


def _detect_flash_attn():
    global _FA_LEVEL, _FA_VARLEN_FUNC, _FA_FUNC
    try:
        import flash_attn
        version = tuple(int(x) for x in flash_attn.__version__.split(".")[:2])

        if version >= (3, 0) and torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            for min_cap, level, label in [(12, 4, "FA4 Blackwell SM120"), (9, 3, "FA3 Hopper SM90")]:
                if cap[0] >= min_cap:
                    try:
                        from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
                        _FA_FUNC, _FA_VARLEN_FUNC, _FA_LEVEL = flash_attn_func, flash_attn_varlen_func, level
                        print(f"  ⚡ FlashAttention-{level} ({label}) détecté")
                        return
                    except ImportError:
                        pass

        if version >= (2, 0):
            from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
            _FA_FUNC, _FA_VARLEN_FUNC, _FA_LEVEL = flash_attn_func, flash_attn_varlen_func, 2
            print("  ⚡ FlashAttention-2 détecté")
            return

    except ImportError:
        pass

    if hasattr(F, "scaled_dot_product_attention"):
        _FA_LEVEL = 1
        print("  ⚡ Flash Attention : SDPA PyTorch (fallback)")
    else:
        print("  ⚠️  Aucune Flash Attention (PyTorch < 2.0)")


_detect_flash_attn()


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention avec head_dim découplé (style Qwen3.5).

    Si `head_dim` est fourni explicitement, q/o_proj utilisent num_heads × head_dim
    même si ce produit ≠ embed_dim (projections rectangulaires).
    Sinon : head_dim = embed_dim // num_heads (comportement standard).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        rope_base: int = 10000,
        use_yarn: bool = False,
        yarn_scale: float = 1.0,
        yarn_original_max_len: int = 1024,
        n_kv_heads: Optional[int] = None,
        use_qk_norm: bool = False,
        use_flash_attn: bool = True,
        soft_cap: Optional[float] = None,
        use_fp8: bool = False,
        head_dim: Optional[int] = None,
    ):
        super().__init__()
        if head_dim is None:
            assert embed_dim % num_heads == 0
        if soft_cap is not None:
            assert soft_cap > 0

        self.embed_dim      = embed_dim
        self.num_heads      = num_heads
        self.head_dim       = head_dim if head_dim is not None else (embed_dim // num_heads)
        self.qo_dim         = self.num_heads * self.head_dim
        self.use_rope       = use_rope
        self.use_qk_norm    = use_qk_norm
        self.use_flash_attn = use_flash_attn
        self.soft_cap       = soft_cap
        self.use_fp8        = use_fp8 and _TE_AVAILABLE

        self.n_kv_heads         = n_kv_heads if n_kv_heads is not None else num_heads
        assert num_heads % self.n_kv_heads == 0
        self.num_queries_per_kv = num_heads // self.n_kv_heads
        self.kv_dim             = self.n_kv_heads * self.head_dim

        Linear = te.Linear if self.use_fp8 else nn.Linear
        self.q_proj   = Linear(embed_dim, self.qo_dim,  bias=False)
        self.k_proj   = Linear(embed_dim, self.kv_dim,  bias=False)
        self.v_proj   = Linear(embed_dim, self.kv_dim,  bias=False)
        self.out_proj = Linear(self.qo_dim, embed_dim,  bias=False)

        self.dropout = nn.Dropout(dropout)

        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None

        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                self.head_dim, max_seq_len,
                rope_base=rope_base,
                use_yarn=use_yarn,
                yarn_scale=yarn_scale,
                yarn_original_max_len=yarn_original_max_len,
            )
        else:
            self.rope = None

        self._fa_level  = _FA_LEVEL if use_flash_attn else 0
        self._fa_varlen = _FA_VARLEN_FUNC
        self._fa_func   = _FA_FUNC
        self._sdpa_ok   = hasattr(F, "scaled_dot_product_attention")

    def _attn_scale(self) -> float:
        if (self.use_rope and self.rope is not None
                and self.rope.use_yarn and self.rope.yarn_scale > 1.0):
            return math.sqrt(self.rope.yarn_scale) / math.sqrt(self.head_dim)
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

        batch_size, seq_len, _ = x.shape
        scale = self._attn_scale()
        H, d  = self.num_heads, self.head_dim

        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        q = self.q_proj(x).view(batch_size, seq_len, H,              d).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, d).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, d).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        position_offset = past_kv[0].shape[2] if past_kv is not None else 0
        if self.use_rope:
            q, k = self.rope(q, k, position_offset=position_offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv_cache: Optional[KVCache] = (k, v) if use_kv_cache else None

        if self.n_kv_heads != H:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        use_varlen = (
            cu_seqlens_q is not None
            and self._fa_level >= 2
            and self._fa_varlen is not None
            and self.soft_cap is None
            and past_kv is None
        )

        if use_varlen:
            q_var = q.permute(0, 2, 1, 3).reshape(-1, H, d)
            k_var = k.permute(0, 2, 1, 3).reshape(-1, H, d)
            v_var = v.permute(0, 2, 1, 3).reshape(-1, H, d)
            _msl_q = max_seqlen_q if max_seqlen_q is not None else seq_len
            _msl_k = max_seqlen_k if max_seqlen_k is not None else seq_len
            output = self._fa_varlen(
                q_var, k_var, v_var,
                cu_seqlens_q, cu_seqlens_k, _msl_q, _msl_k,
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=scale, causal=True,
            )
            output = output.reshape(batch_size, seq_len, H, d).transpose(1, 2)

        elif (self._fa_level >= 2 and self._fa_func is not None
              and self.soft_cap is None and mask is None):
            if q.dtype == torch.float32:
                q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
            is_causal = seq_len > 1 and past_kv is None
            output = self._fa_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=scale, causal=is_causal,
            ).transpose(1, 2)

        elif self._sdpa_ok and self.soft_cap is None and mask is None:
            is_causal = seq_len > 1 and past_kv is None
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=is_causal,
                dropout_p=self.dropout.p if self.training else 0.0, scale=scale,
            )

        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if self.soft_cap is not None:
                scores = self.soft_cap * torch.tanh(scores / self.soft_cap)
            if seq_len > 1 and past_kv is None:
                if mask is not None:
                    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                else:
                    total_len   = k.shape[2]
                    causal_bool = torch.triu(
                        torch.ones(seq_len, total_len, device=q.device, dtype=torch.bool), diagonal=1
                    )
                    scores = scores.masked_fill(causal_bool.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            if self.training and self.dropout.p > 0:
                attn_weights = self.dropout(attn_weights)
            output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.qo_dim)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output, new_kv_cache
