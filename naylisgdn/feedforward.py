import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import transformer_engine.pytorch as te
    _TE_AVAILABLE = True
except ImportError:
    _TE_AVAILABLE = False


class FeedForward(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
        use_swiglu: bool = True,
        use_fp8: bool = False,
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.use_swiglu = use_swiglu
        self.use_fp8    = use_fp8 and _TE_AVAILABLE

        Linear = te.Linear if self.use_fp8 else nn.Linear

        if use_swiglu:
            self.hidden_dim = (int(8 * embed_dim / 3) + 63) // 64 * 64
            self.gate_proj  = Linear(embed_dim, self.hidden_dim, bias=False)
            self.up_proj    = Linear(embed_dim, self.hidden_dim, bias=False)
            self.down_proj  = Linear(self.hidden_dim, embed_dim, bias=False)
        else:
            self.hidden_dim = 4 * embed_dim
            self.fc1 = Linear(embed_dim, self.hidden_dim, bias=False)
            self.fc2 = Linear(self.hidden_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fp8 and x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        if self.use_swiglu:
            x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        else:
            x = self.fc2(F.gelu(self.fc1(x)))

        return self.dropout(x)
