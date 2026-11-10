"""data.peer_missing_mask: 결측 Peer Token을 Attention에서 가린다. 대상 건물은 가리지 않는다."""
import numpy as np
import torch

from src.data.dataset import WindowDataset, collate_windows
from src.models.dual_context import DualContextTransformer
from tests.test_hour_level import hour_cfg

N, STEPS = 5, 48
KEYS = ("target_daily", "zone_daily", "zone_mask", "type_daily", "type_mask")


def _dataset(mask_missing_peers):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N, STEPS, 1)).astype(np.float32)
    observed = np.ones((N, STEPS, 1), dtype=bool)
    observed[2, 10:30] = False  # Peer 2: 10~29시간 결측
    observed[0, 5:8] = False  # 대상 건물 0: 5~7시간 결측
    zone = np.array([[0, 1, 2, -1], [1, 0, 2, 3]])  # 열 0 = 대상, -1 = Padding
    typ = np.array([[0, 2, 3, 4], [1, 2, -1, -1]])
    return WindowDataset(x, np.array([0, 1]), zone, typ, observed, mask_missing_peers=mask_missing_peers)


def _batch(ds, start=0, t=40):
    return collate_windows([ds[(0, start, t)], ds[(1, start, t)]])


def test_mask_marks_missing_peer_steps_only():
    b = _batch(_dataset(True))
    zm = b["zone_mask"].reshape(2, 40, 4)
    assert not zm[0, 10:30, 2].any() and zm[0, :10, 2].all() and zm[0, 30:, 2].all()  # Peer 2 결측 구간만
    assert zm[:, :, 0].all()  # 대상 건물은 결측(5~7시간)이어도 유효
    assert not zm[0, :, 3].any()  # Padding은 그대로 False
    # 건물 0은 대상 1의 그룹에서는 Peer이므로 결측 시점(5~7시간)이 가려진다
    assert zm[1, :, 1].tolist() == [not 5 <= h < 8 for h in range(40)] and not zm[1, 10:30, 2].any()
    tm = b["type_mask"].reshape(2, 40, 4)
    assert not tm[0, 10:30, 1].any() and not tm[1, 10:30, 1].any() and not tm[1, :, 2:].any()


def test_off_keeps_previous_mask():
    b = _batch(_dataset(False))
    zm = b["zone_mask"].reshape(2, 40, 4)
    assert zm[0, :, :3].all() and not zm[0, :, 3].any()  # 건물 단위 Mask: 시간에 따라 바뀌지 않는다


def test_masked_peer_values_are_ignored_by_model():
    torch.manual_seed(0)
    model = DualContextTransformer(hour_cfg()).eval()
    ds = _dataset(True)
    base = model(**{k: _batch(ds)[k] for k in KEYS})
    ds.x[2, 10:30] = 99.0  # 가려진 시점의 Peer 값은 결과에 영향이 없다
    assert torch.allclose(model(**{k: _batch(ds)[k] for k in KEYS}), base, atol=1e-5)
    ds.x[2, 35] = 99.0  # 관측 시점의 Peer 값은 영향이 있다
    assert not torch.allclose(model(**{k: _batch(ds)[k] for k in KEYS}), base, atol=1e-5)


def test_day_token_masks_only_fully_missing_days():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 4, 24)).astype(np.float32)
    observed = np.ones((3, 4, 24), dtype=bool)
    observed[1, 1] = False  # Peer 1의 1일차 전체 결측
    observed[1, 2, :12] = False  # 2일차는 절반만 결측 -> 가리지 않는다
    ds = WindowDataset(x, np.array([0]), np.array([[0, 1, 2]]), None, observed, mask_missing_peers=True)
    zm = collate_windows([ds[(0, 0, 4)]])["zone_mask"].reshape(4, 3)
    assert zm[:, 0].all() and zm[:, 2].all()
    assert zm[:, 1].tolist() == [True, False, True, True]
