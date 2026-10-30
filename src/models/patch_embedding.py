"""하루 24시간 -> Day Token 투영 `Linear(24, D)` (SPEC 3.1)."""
from __future__ import annotations

import math

import torch
from torch import nn


class DayPatchEmbedding(nn.Module):
    """ViT Patch Embedding처럼 24개 시간값을 직접 `D`차원으로 투영한다. ID Lookup은 없다."""

    def __init__(self, d_model: int, patch_size_hours: int = 24):
        super().__init__()
        self.proj = nn.Linear(patch_size_hours, d_model)

    def forward(self, daily: torch.Tensor) -> torch.Tensor:
        return self.proj(daily)


class SinusoidalDayPE(nn.Module):
    """Decoder 날짜 토큰 순서용 Positional Encoding. Window 내 위치 1..T를 표현한다."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.pe[: tokens.shape[-2]]
