"""data.peer_missing_mask (Day Token): 하루 24시간이 모두 결측인 Peer를 그 날짜의 Attention에서 가린다."""
import numpy as np
import torch

from src.data.dataset import WindowDataset, collate_windows
from tests.test_per_day_encoder import per_day_model

N, DAYS = 5, 12
KEYS = ("target_daily", "zone_daily", "zone_mask", "type_daily", "type_mask")


def _dataset(mask_missing_peers):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N, DAYS, 24)).astype(np.float32)
    observed = np.ones((N, DAYS, 24), dtype=bool)
    observed[2, 3:6] = False  # Peer 2: 3~5일차 전체 결측
    observed[2, 7, :12] = False  # 7일차는 절반만 결측 -> 가리지 않는다
    observed[0, 1] = False  # 대상 건물 0: 1일차 전체 결측
    zone = np.array([[0, 1, 2, -1], [1, 0, 2, 3]])  # 열 0 = 대상, -1 = Padding
    typ = np.array([[0, 2, 3, 4], [1, 2, -1, -1]])
    return WindowDataset(x, np.array([0, 1]), zone, typ, observed, mask_missing_peers=mask_missing_peers)


def _batch(ds, t=10):
    return collate_windows([ds[(0, 0, t)], ds[(1, 0, t)]])


def test_mask_marks_fully_missing_peer_days_only():
    zm = _batch(_dataset(True))["zone_mask"].reshape(2, 10, 4)
    assert zm[0, :, 2].tolist() == [not 3 <= d < 6 for d in range(10)]  # 7일차(절반 결측)는 유효
    assert zm[:, :, 0].all()  # 대상 건물은 결측(1일차)이어도 유효
    assert not zm[0, :, 3].any()  # Padding은 그대로 False
    assert zm[1, :, 1].tolist() == [d != 1 for d in range(10)]  # 건물 0은 대상 1의 그룹에선 Peer
    tm = _batch(_dataset(True))["type_mask"].reshape(2, 10, 4)
    assert not tm[0, 3:6, 1].any() and not tm[1, 3:6, 1].any() and not tm[1, :, 2:].any()


def test_off_keeps_previous_mask():
    zm = _batch(_dataset(False))["zone_mask"].reshape(2, 10, 4)
    assert zm[0, :, :3].all() and not zm[0, :, 3].any()


def test_masked_peer_values_are_ignored_by_model():
    model = per_day_model()
    ds = _dataset(True)
    base = model(**{k: _batch(ds)[k] for k in KEYS})
    ds.x[2, 3:6] = 99.0  # 가려진 날짜의 Peer 값은 결과에 영향이 없다
    assert torch.allclose(model(**{k: _batch(ds)[k] for k in KEYS}), base, atol=1e-5)
    ds.x[2, 8] = 99.0  # 관측 날짜의 Peer 값은 영향이 있다
    assert not torch.allclose(model(**{k: _batch(ds)[k] for k in KEYS}), base, atol=1e-5)
