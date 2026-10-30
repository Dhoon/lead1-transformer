"""Zone/Type 동기식 Peer 그룹 구성 (SPEC 3.2, 4.2, 4.3)."""
from __future__ import annotations

import numpy as np


def peer_index(
    group_key: np.ndarray,
    targets: np.ndarray,
    pool: np.ndarray,
    exclude_self: bool = True,
) -> np.ndarray:
    """각 대상 건물과 같은 그룹에 속한 Peer 건물의 행 인덱스를 반환한다.

    group_key: 전체 격자 건물별 그룹 값 (`site_id` 또는 `primary_use`).
    targets, pool: 전체 격자의 행 인덱스. Peer는 `pool` 안에서만 고른다.
    반환: `[len(targets), G]`, 빈 자리는 -1. G는 최소 1이다.
    """
    groups = [
        [p for p in pool if group_key[p] == group_key[t] and not (exclude_self and p == t)]
        for t in targets
    ]
    width = max(1, max((len(g) for g in groups), default=0))
    out = np.full((len(targets), width), -1, dtype=np.int64)
    for i, g in enumerate(groups):
        out[i, : len(g)] = g
    return out
