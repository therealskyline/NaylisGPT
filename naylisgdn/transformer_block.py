import torch
import torch.nn as nn
from typing import Optional, Tuple

from naylisgdn.norm import RMSNorm
from naylisgdn.attention import MultiHeadAttention, KVCache
from naylisgdn.feedforward import FeedForward


class TransformerBlock(nn.Module):

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
        use_swiglu: bool = True,
        n_kv_heads: Optional[int] = None,
        use_qk_norm: bool = False,
        use_flash_attn: bool = True,
        soft_cap: Optional[float] = None,
        use_fp8: bool = False,
        attn_head_dim: Optional[int] = None,
        use_moe: bool = False,
        num_experts: int = 16,
        top_k_experts: int = 2,
        shared_experts: int = 2,
        expert_hidden_dim: Optional[int] = None,
        moe_aux_coeff: float = 0.01,
        use_moh: bool = False,
        moh_shared_heads: Optional[int] = None,
        moh_top_k_routed: Optional[int] = None,
    ):
        super().__init__()

        self.use_moe       = use_moe
        self.moe_aux_coeff = moe_aux_coeff
        self.use_moh       = use_moh

        self.ln1 = RMSNorm(embed_dim)

        if use_moh:
            from naylisgdn.moh import MixtureOfHeads
            sh = moh_shared_heads if moh_shared_heads is not None else (num_heads // 2)
            kr = moh_top_k_routed if moh_top_k_routed is not None else (num_heads - sh) // 2
            self.attention = MixtureOfHeads(
                embed_dim, num_heads, shared_heads=sh, top_k_routed=kr,
                dropout=dropout, use_rope=use_rope, max_seq_len=max_seq_len,
                rope_base=rope_base,
                use_yarn=use_yarn, yarn_scale=yarn_scale,
                yarn_original_max_len=yarn_original_max_len,
                n_kv_heads=n_kv_heads, use_qk_norm=use_qk_norm,
                soft_cap=soft_cap, use_fp8=use_fp8,
                head_dim=attn_head_dim,
            )
        else:
            self.attention = MultiHeadAttention(
                embed_dim, num_heads, dropout,
                use_rope=use_rope, max_seq_len=max_seq_len,
                rope_base=rope_base,
                use_yarn=use_yarn, yarn_scale=yarn_scale,
                yarn_original_max_len=yarn_original_max_len,
                n_kv_heads=n_kv_heads, use_qk_norm=use_qk_norm,
                use_flash_attn=use_flash_attn, soft_cap=soft_cap,
                use_fp8=use_fp8, head_dim=attn_head_dim,
            )

        self.ln2 = RMSNorm(embed_dim)

        if use_moe:
            from naylisgdn.moe import SparseMoE
            self.ffn = SparseMoE(
                embed_dim=embed_dim,
                num_experts=num_experts,
                top_k=top_k_experts,
                shared_experts=shared_experts,
                expert_hidden_dim=expert_hidden_dim,
                dropout=dropout,
                use_fp8=use_fp8,
            )
        else:
            self.ffn = FeedForward(embed_dim, dropout, use_swiglu=use_swiglu, use_fp8=use_fp8)

    def forward(
        self,
        x: torch.Tensor,
        mask=None,
        past_kv=None,
        use_kv_cache: bool = False,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
    ) -> Tuple[torch.Tensor, Optional[KVCache], torch.Tensor]:

        residual = x
        x, new_kv = self.attention(
            self.ln1(x), mask=mask, past_kv=past_kv, use_kv_cache=use_kv_cache,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        )
        x = residual + x

        residual = x
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(self.ln2(x))
            x = residual + ffn_out
        else:
            x        = residual + self.ffn(self.ln2(x))
            aux_loss = torch.zeros(1, device=x.device, dtype=x.dtype)

        return x, new_kv, aux_loss
