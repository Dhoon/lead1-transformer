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
    if cfg.encoder_peers == "target_only":  # E1b: Encoder 경로는 두고 Peer만 뺀다
        zone = None if zone is None else zone[:, :1]
        typ = None if typ is None else typ[:, :1]
    return zone, typ


def max_group_size(*groups: np.ndarray | None) -> int:
    return max([1] + [int((g >= 0).sum(1).max()) for g in groups if g is not None])


def run_dir(cfg: dict, fold: int) -> Path:
    out = Path(cfg.get("output_dir", "runs")) / ("full" if fold < 0 else f"fold{fold}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _hours(section: dict, key: str):
    """`{key}_hours`가 있으면 그대로, 없으면 `{key}_days`×24. 값이 리스트면 원소마다 변환한다. 없으면 None."""
    if section.get(f"{key}_hours") is not None:
        return section[f"{key}_hours"]
    days = section.get(f"{key}_days")
    if days is None:
        return None
    return [d * 24 for d in days] if isinstance(days, list) else days * 24


def to_steps(hours: int, patch_size_hours: int) -> int:
    """시간 길이 -> Token 개수. Day Token(24)이면 일 수, Hour Token(1)이면 시간 수."""
    if hours % patch_size_hours:
        raise ValueError(f"Window {hours}시간이 Token 크기 {patch_size_hours}시간으로 나누어떨어지지 않는다.")
    return hours // patch_size_hours


def window_steps(cfg: dict, patch_size_hours: int) -> dict:
    """train/eval의 Window 설정을 Token 단위로 바꾼다.

    v3 이전 Config는 `*_days`(일), v5는 `*_hours`(시간)로 적는다. 모델은 Token 개수 T만 본다.
    반환: min_t, max_t (학습), eval (평가 길이 리스트, 첫 값이 best.pt·제출 기준), stride (None = Window와 같음).
    """
    tcfg, ev = cfg["train"], cfg["eval"]
    evals = _hours(ev, "window")
    evals = evals if isinstance(evals, list) else [evals]
    stride = _hours(ev, "stride")
    return {
        "min_t": to_steps(_hours(tcfg, "min_window"), patch_size_hours),
        "max_t": to_steps(_hours(tcfg, "max_window"), patch_size_hours),
        "eval": [to_steps(w, patch_size_hours) for w in evals],
        "stride": None if stride is None else to_steps(stride, patch_size_hours),
    }
