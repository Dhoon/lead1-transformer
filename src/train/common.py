"""Pretraining/Fine-tuning 공용 유틸."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.lead_loader import LeadGrid
from src.data.peer_groups import peer_index
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


def build_peers(grid: LeadGrid, targets: np.ndarray, pool: np.ndarray, cfg: ModelConfig):
    """site_id(Zone proxy)·primary_use(Type)로 Peer 행 인덱스를 만든다. ID 자체는 모델 입력이 아니다."""
    zone = peer_index(grid.site_id, targets, pool, cfg.exclude_target_from_peers) if cfg.use_zone_encoder else None
    typ = peer_index(grid.primary_use, targets, pool, cfg.exclude_target_from_peers) if cfg.use_type_encoder else None
    return zone, typ


def run_dir(cfg: dict, name: str, fold: int) -> Path:
    out = Path(cfg.get("output_dir", "runs")) / name / ("full" if fold < 0 else f"fold{fold}")
    out.mkdir(parents=True, exist_ok=True)
    return out
