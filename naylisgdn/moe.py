import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    import transformer_engine.pytorch as te
    _TE_AVAILABLE = True
except ImportError:
    _TE_AVAILABLE = False


class ExpertFFN(nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, use_fp8: bool = False):
        super().__init__()
        Linear         = te.Linear if (use_fp8 and _TE_AVAILABLE) else nn.Linear
        self.gate_proj = Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj   = Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TopKRouter(nn.Module):

    def __init__(self, embed_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.gate        = nn.Linear(embed_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D   = x.shape
        flat      = x.view(-1, D)
        logits    = self.gate(flat)
        probs     = F.softmax(logits, dim=-1)

        top_vals, top_idx = torch.topk(probs, self.top_k, dim=-1)
        top_vals          = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-9)

        tokens_per_expert = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
        tokens_per_expert.scatter_add_(0, top_idx.view(-1),
                                       torch.ones(flat.size(0) * self.top_k, device=x.device, dtype=x.dtype))
        tokens_per_expert = tokens_per_expert / (flat.size(0) * self.top_k + 1e-9)

        mean_prob = probs.mean(dim=0)
        aux_loss  = self.num_experts * (tokens_per_expert * mean_prob).sum()

        return top_idx, top_vals, aux_loss


class SparseMoE(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_experts: int = 16,
        top_k: int = 2,
        shared_experts: int = 2,
        expert_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_fp8: bool = False,
    ):
        super().__init__()
        self.embed_dim      = embed_dim
        self.num_experts    = num_experts
        self.top_k          = top_k
        self.shared_experts = shared_experts

        if expert_hidden_dim is None:
            expert_hidden_dim = (int(embed_dim * 8 / 3 / top_k) + 63) // 64 * 64

        self.expert_hidden_dim = expert_hidden_dim

        self.router  = TopKRouter(embed_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            ExpertFFN(embed_dim, expert_hidden_dim, use_fp8=use_fp8)
            for _ in range(num_experts)
        ])
        self.shared  = nn.ModuleList([
            ExpertFFN(embed_dim, expert_hidden_dim, use_fp8=use_fp8)
            for _ in range(shared_experts)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D   = x.shape
        flat      = x.view(-1, D)

        top_idx, top_vals, aux_loss = self.router(x)

        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            mask  = (top_idx == i).any(dim=-1)
            if not mask.any():
                continue
            pos   = mask.nonzero(as_tuple=True)[0]
            k_pos = (top_idx[pos] == i).nonzero(as_tuple=True)[1]
            w     = top_vals[pos, k_pos].unsqueeze(-1)
            out[pos] += w * expert(flat[pos])

        for shared_expert in self.shared:
            out += shared_expert(flat)

        out = self.dropout(out.view(B, T, D))
        return out, aux_loss
