"""Zone/Type Encoder: 순서 없는 건물 집합 Transformer (SPEC 4.2, 4.3)."""
from __future__ import annotations

import torch
from torch import nn


def safe_key_mask(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """유효 토큰이 하나도 없는 행은 Attention이 NaN이 되므로 첫 칸을 임시로 연다.

    mask: `[N, S]`, True = 유효 토큰. 반환: (PyTorch용 key_padding_mask, 빈 행 여부 `[N]`).
    """
    empty = ~mask.any(-1)
    opened = mask.clone()
    opened[empty, 0] = True
    return ~opened, empty


class SetEncoder(nn.Module):
    """Positional Encoding 없이 Self-Attention만 쌓아 순열 등변성을 유지한다."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float, norm_first: bool):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=True, norm_first=norm_first
            )
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """tokens: `[N, S, D]`, mask: `[N, S]` -> `[N, S, D]`."""
        key_padding, _ = safe_key_mask(mask)
        h = tokens
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=key_padding)
        return self.norm(h)
