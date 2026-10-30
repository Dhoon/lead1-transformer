import numpy as np

from src.data.day_patches import LogStandardizer
from src.data.peer_groups import peer_index
from src.data.splits import building_folds


def test_building_folds_are_disjoint_and_cover_all():
    ids = np.arange(200) + 1000
    seen = []
    for tr, va in building_folds(ids, 5, seed=42):
        assert len(tr) == 160 and len(va) == 40
        assert not set(ids[tr]) & set(ids[va])
        seen.extend(ids[va])
    assert sorted(seen) == sorted(ids)


def test_peers_drawn_only_from_pool_and_exclude_self():
    key = np.array([0, 0, 0, 1, 1, 0])
    pool = np.array([0, 1, 3, 4])
    out = peer_index(key, np.array([0, 5, 3]), pool, exclude_self=True)
    assert set(out[0][out[0] >= 0]) == {1}
    assert set(out[1][out[1] >= 0]) == {0, 1}  # 검증 건물 5는 학습 Pool에서만 Peer를 얻는다
    assert set(out[2][out[2] >= 0]) == {4}


def test_normalizer_fit_uses_only_given_rows():
    values = np.array([[1.0, 2.0], [1000.0, 2000.0]])
    valid = np.ones_like(values, dtype=bool)
    assert LogStandardizer().fit(values[:1], valid[:1]).mean != LogStandardizer().fit(values, valid).mean
