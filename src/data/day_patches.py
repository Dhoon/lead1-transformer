"""hourly -> 24-hour Day Vector 변환과 정규화 (SPEC 3.1, 7.1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOURS = 24


def hourly_to_grid(
    building: np.ndarray,
    timestamps: pd.Series,
    columns: dict[str, np.ndarray],
    building_ids: np.ndarray,
    start: pd.Timestamp,
    n_days: int,
) -> dict[str, np.ndarray]:
    """시간별 Row들을 `[nb, n_days, 24]` 격자에 배치한다. 반환값에는 Row 존재 여부 `present`가 포함된다."""
    b_pos = pd.Index(building_ids).get_indexer(building)
    if (b_pos < 0).any():
        raise ValueError("메타데이터에 없는 building_id가 있습니다.")
    delta = (pd.DatetimeIndex(timestamps) - start)
    day = np.asarray(delta.days)
    hour = np.asarray(delta.seconds // 3600)

    shape = (len(building_ids), n_days, HOURS)
    out: dict[str, np.ndarray] = {"present": np.zeros(shape, dtype=bool)}
    out["present"][b_pos, day, hour] = True
    for name, col in columns.items():
        fill = np.nan if np.issubdtype(col.dtype, np.floating) else 0
        grid = np.full(shape, fill, dtype=col.dtype)
        grid[b_pos, day, hour] = col
        out[name] = grid
    return out


@dataclass
class LogStandardizer:
    """`log1p(meter_reading)` 후 학습 구간 통계로 표준화한다. 결측값은 0(=평균)으로 대체한다."""

    mean: float = 0.0
    std: float = 1.0

    def fit(self, values: np.ndarray, valid: np.ndarray) -> "LogStandardizer":
        logv = np.log1p(np.clip(values[valid], 0, None))
        self.mean = float(logv.mean())
        self.std = float(logv.std() + 1e-6)
        return self

    def transform(self, values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        out = (np.log1p(np.clip(np.nan_to_num(values), 0, None)) - self.mean) / self.std
        return np.where(valid, out, 0.0).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}
