"""Pretraining ID 분류 Head와 Downstream 이상 탐지 Head (SPEC 5.1, 6.2)."""
from __future__ import annotations

import torch
from torch import nn


class BuildingIdHead(nn.Module):
    """`Linear(D, N)`. Softmax는 Cross-Entropy 손실 안에서 적용한다."""

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


class HourlyAnomalyHead(nn.Module):
    """Day Token 표현 하나에서 그 날짜 24시간의 이상 Logit을 출력한다."""

    def __init__(self, d_model: int, hours: int = 24):
        super().__init__()
        self.proj = nn.Linear(d_model, hours)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)
