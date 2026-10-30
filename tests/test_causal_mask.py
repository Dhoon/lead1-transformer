import torch

from src.models.temporal_decoder import causal_mask
from tests.conftest import random_batch


def test_mask_pattern():
    m = causal_mask(4)
    assert not m[2, :3].any() and m[2, 3:].all()


def test_day_state_independent_of_later_target_days(model):
    batch = random_batch()
    d = 2
    base = model(**batch)[:, : d + 1]
    changed = dict(batch)
    changed["target_daily"] = batch["target_daily"].clone()
    changed["target_daily"][:, d + 1 :] += 10.0
    assert torch.allclose(model(**changed)[:, : d + 1], base, atol=1e-5)
