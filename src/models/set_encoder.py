"""Joint T-day Zone/Type Encoder + Shared Day PE (SPEC 4.2, 4.3)."""
from __future__ import annotations

import torch
from torch import nn

from src.models.patch_embedding import DayPatchEmbedding, SinusoidalDayPE


class JointDayEncoder(nn.Module):
    """그룹 건물들의 T일 Day Token을 day-major로 Flatten해 하나의 Self-Attention 시퀀스로 처리한다.

    건물 나열 순서용 PE는 없고, 같은 날짜 Token에는 같은 PE(d)를 더한다.
    날짜 Causal Mask 없이 Window 전체를 양방향으로 본다 (오프라인/사후 탐지).
    """

    def __init__(
        self, patch: DayPatchEmbedding, day_pe: SinusoidalDayPE | None,
        d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float, norm_first: bool,
    ):
        super().__init__()
        self.patch, self.day_pe = patch, day_pe
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=True, norm_first=norm_first
            )
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def embed(self, daily: torch.Tensor) -> torch.Tensor:
        """daily: `[B, T, G, 24]` -> Flatten된 입력 Token `[B, T×G, D]`."""
        b, t, g, _ = daily.shape
        tokens = self.patch(daily)
        if self.day_pe is not None:
            tokens = tokens + self.day_pe(t)[None, :, None, :]
        return tokens.reshape(b, t * g, -1)

    def forward(self, daily: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """mask: `[B, T×G]`, True = 유효 Token(그룹 패딩이 아님) -> 메모리 `[B, T×G, D]`."""
        h = self.embed(daily)
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=~mask)
        return self.norm(h)
