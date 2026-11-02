"""Building-aware CV (SPEC 7.2)."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


def building_folds(building_ids: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Building ID 기준 K-Fold. 반환되는 인덱스는 `building_ids`의 위치다."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(building_ids))
    folds = []
    for tr, va in GroupKFold(n_splits=n_splits).split(order, groups=building_ids[order]):
        folds.append((np.sort(order[tr]), np.sort(order[va])))
    return folds


def select_fold(building_ids: np.ndarray, fold: int, n_splits: int, seed: int):
    """fold < 0이면 전체 건물을 학습에 쓰고 검증 건물은 비운다 (최종 제출 학습)."""
    if fold < 0:
        return np.arange(len(building_ids)), np.array([], dtype=np.int64)
    return building_folds(building_ids, n_splits, seed)[fold]
