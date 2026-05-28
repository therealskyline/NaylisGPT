import torch
import torch.nn as nn

_LIGER_RMSNORM = None

try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    _LIGER_RMSNORM = LigerRMSNorm
except ImportError:
    pass


class RMSNorm(nn.Module):
    """RMSNorm with optional Liger-Kernel Triton fusion.

    When liger-kernel is installed, delegates entirely to LigerRMSNorm so the
    weight is registered once and the Triton kernel handles the computation.
    Falls back to a pure-PyTorch implementation otherwise.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        if _LIGER_RMSNORM is not None:
            self._impl = _LIGER_RMSNORM(dim, eps=eps)
        else:
            self._impl = None
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._impl is not None:
            return self._impl(x)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight
