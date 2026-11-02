"""학습/추론 공용 유틸."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.lead_loader import LeadGrid
from src.data.peer_groups import group_members
from src.models.dual_context import ModelConfig


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def build_groups(grid: LeadGrid, targets: np.ndarray, pool: np.ndarray, cfg: ModelConfig):
    """site_id(Zone proxy)·primary_use(Type)로 그룹을 만든다. 열 0은 대상 건물. ID 자체는 모델 입력이 아니다."""
    zone = group_members(grid.site_id, targets, pool) if "zone" in cfg.contexts else None
    typ = group_members(grid.primary_use, targets, pool) if "type" in cfg.contexts else None
    return zone, typ


def max_group_size(*groups: np.ndarray | None) -> int:
    return max([1] + [int((g >= 0).sum(1).max()) for g in groups if g is not None])


def run_dir(cfg: dict, fold: int) -> Path:
    out = Path(cfg.get("output_dir", "runs")) / ("full" if fold < 0 else f"fold{fold}")
    out.mkdir(parents=True, exist_ok=True)
    return out
