import numpy as np

from src.data.day_patches import LogStandardizer
from src.data.peer_groups import group_members
from src.data.splits import building_folds


def test_building_folds_are_disjoint_and_cover_all():
    ids = np.arange(200) + 1000
    seen = []
    for tr, va in building_folds(ids, 5, seed=42):
        assert len(tr) == 160 and len(va) == 40
        assert not set(ids[tr]) & set(ids[va])
        seen.extend(ids[va])
    assert sorted(seen) == sorted(ids)


def test_train_groups_use_only_training_buildings():
    key = np.zeros(6, dtype=int)
    train, val = np.array([0, 1, 2]), np.array([3, 4, 5])
    tr = group_members(key, train, train)
    va = group_members(key, val, val)  # within_split
    assert set(tr[tr >= 0]) <= set(train)
    assert set(va[va >= 0]) <= set(val)


def test_eval_groups_match_training_buildings():
    """peer_pool=train_only: 평가 건물이 첫 Token, 나머지 Peer는 학습 건물에서만 온다."""
    key = np.array([0, 0, 1, 0, 1, 2])
    train, val = np.array([0, 1, 2]), np.array([3, 4, 5])
    out = group_members(key, val, train)
    assert out[:, 0].tolist() == [3, 4, 5]
    assert set(out[0][out[0] >= 0]) == {3, 0, 1}
    assert set(out[1][out[1] >= 0]) == {4, 2}
    assert set(out[2][out[2] >= 0]) == {5}  # 학습 건물에 없는 그룹 -> 대상 단독
    assert all(set(row[1:][row[1:] >= 0]) <= set(train) for row in out)


def test_normalizer_fit_uses_only_given_rows():
    values = np.array([[1.0, 2.0], [1000.0, 2000.0]])
    observed = np.ones_like(values, dtype=bool)
    assert LogStandardizer().fit(values[:1], observed[:1]).mean != LogStandardizer().fit(values, observed).mean
