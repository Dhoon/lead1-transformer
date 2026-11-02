"""Zone/Type 날짜 동기화 그룹 구성 (SPEC 3.2, 4.2, 4.3)."""
from __future__ import annotations

import numpy as np


def group_members(group_key: np.ndarray, targets: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """대상 건물을 첫 번째에 두고, `pool`에서 같은 그룹 값을 가진 나머지 건물을 잇는다.

    group_key: 전체 격자 건물별 그룹 값 (`site_id` 또는 `primary_use`).
    targets, pool: 전체 격자의 행 인덱스.
    반환: `[len(targets), G]`, 빈 자리는 -1. 열 0은 항상 대상 건물이다.
    Peer가 없으면 대상 건물 단독 그룹이 된다 (SPEC 4.5).
    """
    groups = [[t] + [p for p in pool if p != t and group_key[p] == group_key[t]] for t in targets]
    width = max(len(g) for g in groups) if groups else 1
    out = np.full((len(targets), width), -1, dtype=np.int64)
    for i, g in enumerate(groups):
        out[i, : len(g)] = g
    return out
