"""날짜별 `Linear(D, 24)` 시간별 이상 Logit Head (SPEC 5.2)."""
from __future__ import annotations

import torch
from torch import nn


class HourlyAnomalyHead(nn.Module):
    def __init__(self, d_model: int, hours: int = 24):
        super().__init__()
        self.proj = nn.Linear(d_model, hours)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """`[B, T, D]` -> `[B, T, 24]`."""
        return self.proj(h)
