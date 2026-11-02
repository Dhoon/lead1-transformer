"""hourly -> 24-hour Day Vector 변환, 결측 대체, 정규화 (SPEC 3.1, 7.1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOURS = 24
MISSING_STRATEGIES = ("zero_fill", "building_median")


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


def impute_missing(values: np.ndarray, strategy: str) -> np.ndarray:
    """결측 `meter_reading`을 원래 소비량 단위에서 값으로 대체한다. Attention Mask가 아니다.

    zero_fill (V1): NaN -> 0. 실제 0 소비량은 그대로 둔다.
    building_median (V2): NaN -> 해당 건물의 관측값 중앙값. 라벨은 쓰지 않는다.
      관측이 하나도 없는 건물은 SPEC 7.1 [미정]이므로 0으로 둔다.
    """
    if strategy not in MISSING_STRATEGIES:
        raise ValueError(f"missing_value_strategy must be one of {MISSING_STRATEGIES}")
    observed = ~np.isnan(values)
    if strategy == "zero_fill":
        fill = np.zeros(values.shape[0], dtype=values.dtype)
    else:
        flat = values.reshape(values.shape[0], -1)
        has_obs = observed.reshape(values.shape[0], -1).any(1)
        fill = np.zeros(values.shape[0], dtype=values.dtype)
        fill[has_obs] = np.nanmedian(flat[has_obs], axis=1)
    return np.where(observed, values, fill[:, None, None]).astype(values.dtype)


@dataclass
class LogStandardizer:
    """`log1p(meter_reading)` 후 Training Fold 관측값 통계로 표준화한다."""

    mean: float = 0.0
    std: float = 1.0

    def fit(self, values: np.ndarray, observed: np.ndarray) -> "LogStandardizer":
        logv = np.log1p(np.clip(values[observed], 0, None))
        self.mean = float(logv.mean())
        self.std = float(logv.std() + 1e-6)
        return self

    def transform(self, filled: np.ndarray) -> np.ndarray:
        """filled: 결측 대체가 끝난 값 (NaN 없음)."""
        return ((np.log1p(np.clip(filled, 0, None)) - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}
