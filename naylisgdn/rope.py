import torch
import torch.nn as nn
from typing import Tuple


class RotaryPositionalEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
        rope_base: int = 0,
        device=None,
        use_yarn: bool = False,
        yarn_scale: float = 1.0,
        yarn_original_max_len: int = 1024,
    ):
        super().__init__()
        self.dim                   = dim
        self.max_seq_len           = max_seq_len
        self.base                  = rope_base if rope_base > 0 else base
        self.use_yarn              = use_yarn
        self.yarn_scale            = yarn_scale
        self.yarn_original_max_len = yarn_original_max_len

        if use_yarn:
            assert 0.1 <= yarn_scale <= 16.0
            inv_freq = self._compute_yarn_frequencies()
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = None
        self._cos_cached     = None
        self._sin_cached     = None

    def _compute_yarn_frequencies(self) -> torch.Tensor:
        freqs         = torch.arange(0, self.dim, 2).float() / self.dim
        inv_freq_base = 1.0 / (self.base ** freqs)
        if self.yarn_scale == 1.0:
            return inv_freq_base
        alpha = self.yarn_scale
        beta  = max(self.dim // 2, int(self.dim * 0.25))
        dims  = torch.arange(0, self.dim, 2).float()
        scale = torch.where(
            dims < beta,
            torch.ones_like(dims),
            1 + (alpha - 1) * (dims - beta) / (self.dim - beta),
        )
        return inv_freq_base / scale

    def _update_cos_sin_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            seq_len != self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            self._seq_len_cached = seq_len
            t     = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(dtype))
            emb   = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
        return self._cos_cached, self._sin_cached

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len   = q.shape[2]
        total_len = seq_len + position_offset
        cos, sin  = self._update_cos_sin_cache(total_len, q.device, q.dtype)
        cos = cos[position_offset : position_offset + seq_len][None, None, :, :]
        sin = sin[position_offset : position_offset + seq_len][None, None, :, :]
        return (
            (q * cos) + (self._rotate_half(q) * sin),
            (k * cos) + (self._rotate_half(k) * sin),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rotary_pos_emb(q, k, position_offset)
