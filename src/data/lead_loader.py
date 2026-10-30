"""LEAD 대회 CSV 로딩 및 건물/시간 정렬 (SPEC 3, 7.1)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.day_patches import HOURS, hourly_to_grid

META_COLUMNS = ["building_id", "site_id", "primary_use"]


@dataclass
class LeadGrid:
    """건물 x 날짜 x 시간 격자. 모든 배열의 첫 축은 `building_ids` 순서를 따른다."""

    building_ids: np.ndarray  # [nb]
    site_id: np.ndarray  # [nb] Zone proxy (SPEC 3.2)
    primary_use: np.ndarray  # [nb] Building Type
    values: np.ndarray  # [nb, n_days, 24] raw meter_reading, 결측은 nan
    present: np.ndarray  # [nb, n_days, 24] 원본 CSV에 Row가 존재하는지
    anomaly: np.ndarray | None  # [nb, n_days, 24] int8, test는 None
    row_id: np.ndarray  # [nb, n_days, 24] 원본 Row 식별자, 없으면 -1
    start: pd.Timestamp

    @property
    def valid(self) -> np.ndarray:
        """원래 관측된 시간값 여부 (`day_valid_mask`)."""
        return ~np.isnan(self.values)

    @property
    def n_days(self) -> int:
        return self.values.shape[1]

    def subset(self, idx: np.ndarray) -> "LeadGrid":
        return LeadGrid(
            building_ids=self.building_ids[idx],
            site_id=self.site_id[idx],
            primary_use=self.primary_use[idx],
            values=self.values[idx],
            present=self.present[idx],
            anomaly=None if self.anomaly is None else self.anomaly[idx],
            row_id=self.row_id[idx],
            start=self.start,
        )


def concat_grids(a: LeadGrid, b: LeadGrid) -> LeadGrid:
    """두 격자를 건물 축으로 이어 붙인다. 레이블이 없는 쪽은 anomaly=0으로 채우지 않고 None을 유지한다."""
    assert a.start == b.start and a.n_days == b.n_days
    anomaly = None
    if a.anomaly is not None and b.anomaly is not None:
        anomaly = np.concatenate([a.anomaly, b.anomaly])
    return LeadGrid(
        building_ids=np.concatenate([a.building_ids, b.building_ids]),
        site_id=np.concatenate([a.site_id, b.site_id]),
        primary_use=np.concatenate([a.primary_use, b.primary_use]),
        values=np.concatenate([a.values, b.values]),
        present=np.concatenate([a.present, b.present]),
        anomaly=anomaly,
        row_id=np.concatenate([a.row_id, b.row_id]),
        start=a.start,
    )


def load_meta(raw_dir: Path, split: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """`{split}_features.csv`에서 건물 메타데이터만 읽는다. 파일이 커서 결과를 캐시한다."""
    cache = None if cache_dir is None else Path(cache_dir) / f"meta_{split}.csv"
    if cache is not None and cache.exists():
        return pd.read_csv(cache)
    meta = (
        pd.read_csv(Path(raw_dir) / f"{split}_features.csv", usecols=META_COLUMNS)
        .drop_duplicates("building_id")
        .sort_values("building_id")
        .reset_index(drop=True)
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(cache, index=False)
    return meta


def grid_from_frame(df: pd.DataFrame, meta: pd.DataFrame) -> LeadGrid:
    """시간별 Long-format Frame을 건물 x 날짜 x 24 격자로 정렬한다."""
    meta = meta.sort_values("building_id").reset_index(drop=True)
    building_ids = meta["building_id"].to_numpy()
    timestamps = pd.to_datetime(df["timestamp"])
    start = timestamps.min().normalize()
    end = timestamps.max().normalize()
    n_days = (end - start).days + 1

    row_id = df["row_id"].to_numpy() if "row_id" in df else df.index.to_numpy()
    columns = {"values": df["meter_reading"].to_numpy(dtype=np.float64), "row_id": row_id}
    if "anomaly" in df:
        columns["anomaly"] = df["anomaly"].to_numpy()
    grids = hourly_to_grid(df["building_id"].to_numpy(), timestamps, columns, building_ids, start, n_days)

    return LeadGrid(
        building_ids=building_ids,
        site_id=meta["site_id"].to_numpy(),
        primary_use=meta["primary_use"].astype(str).to_numpy(),
        values=grids["values"].astype(np.float32),
        present=grids["present"],
        anomaly=grids["anomaly"].astype(np.int8) if "anomaly" in grids else None,
        row_id=np.where(grids["present"], grids["row_id"], -1).astype(np.int64),
        start=start,
    )


def load_split(raw_dir: str | Path, split: str, cache_dir: str | Path | None = None) -> LeadGrid:
    """`train` 또는 `test` 분할을 격자로 읽는다."""
    raw_dir = Path(raw_dir)
    df = pd.read_csv(raw_dir / f"{split}.csv")
    meta = load_meta(raw_dir, split, None if cache_dir is None else Path(cache_dir))
    grid = grid_from_frame(df, meta)
    assert grid.values.shape[2] == HOURS
    return grid
