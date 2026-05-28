import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

from naylisgdn.norm import RMSNorm

try:
    import transformer_engine.pytorch as te
    _TE_AVAILABLE = True
except ImportError:
    _TE_AVAILABLE = False


class MTPModule(nn.Module):

    def __init__(self, embed_dim: int, vocab_size: int, use_fp8: bool = False):
        super().__init__()
        Linear = te.Linear if (use_fp8 and _TE_AVAILABLE) else nn.Linear

        self.norm_h    = RMSNorm(embed_dim)
        self.norm_e    = RMSNorm(embed_dim)
        self.proj      = Linear(embed_dim * 2, embed_dim, bias=False)
        self.norm_out  = RMSNorm(embed_dim)
        self.head      = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        next_embeds: torch.Tensor,
    ) -> torch.Tensor:
        h  = self.norm_h(hidden)
        e  = self.norm_e(next_embeds)
        x  = self.proj(torch.cat([h, e], dim=-1))
        x  = self.norm_out(x)
        return self.head(x), x


class MultiTokenPrediction(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        num_steps: int = 3,
        use_fp8: bool = False,
        weight: float = 0.3,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.weight    = weight
        self.modules_  = nn.ModuleList([
            MTPModule(embed_dim, vocab_size, use_fp8=use_fp8)
            for _ in range(num_steps)
        ])

    def forward(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        embed_fn: Callable[[torch.Tensor], torch.Tensor],
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        B, T, D     = hidden.shape
        ignore      = pad_token_id if pad_token_id is not None else -100
        total_loss  = torch.zeros(1, device=hidden.device, dtype=hidden.dtype)

        h = hidden.detach()

        for step, module in enumerate(self.modules_):
            k = step + 1
            if T - k <= 0:
                break

            h_in        = h[:, :T - k]
            target_ids  = targets[:, k:]

            with torch.no_grad():
                next_tok    = targets[:, k - 1:T - 1].clamp(min=0)
                next_embeds = embed_fn(next_tok)

            logits, h = module(h_in, next_embeds)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=ignore,
            )
            total_loss = total_loss + loss

        return total_loss * self.weight / max(self.num_steps, 1)
