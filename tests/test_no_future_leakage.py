import numpy as np
import torch

from src.data.dataset import WindowDataset
from tests.conftest import random_batch


def test_future_peer_context_does_not_leak(model):
    batch = random_batch()
    d = 2
    base = model(**batch)[:, : d + 1]
    changed = dict(batch)
    for k in ("zone_daily", "type_daily"):
        changed[k] = batch[k].clone()
        changed[k][:, d + 1 :] = torch.randn_like(changed[k][:, d + 1 :]) * 5
    for k in ("zone_mask", "type_mask"):
        changed[k] = batch[k].clone()
        changed[k][:, d + 1 :] = False
    assert torch.allclose(model(**changed)[:, : d + 1], base, atol=1e-5)


def test_window_contains_only_days_up_to_end_day():
    n, days, t = 3, 8, 4
    x = np.arange(n * days * 24, dtype=np.float32).reshape(n, days, 24)
    valid = np.ones_like(x, dtype=bool)
    ds = WindowDataset(x, valid, np.array([0]), t, [1], zone_idx=np.array([[1, 2]]))
    item = ds[0]
    # 끝 날짜 1 -> 날짜 -2, -1(패딩), 0, 1
    assert torch.equal(item["target_daily"][-1], torch.from_numpy(x[0, 1]))
    assert torch.equal(item["target_daily"][-2], torch.from_numpy(x[0, 0]))
    assert not item["day_valid_mask"][:2].any()
    assert torch.equal(item["zone_daily"][-1, 0], torch.from_numpy(x[1, 1]))
    assert not item["zone_mask"][:2].any()
