import torch

from tests.conftest import random_batch


def test_peer_order_does_not_change_prediction(model):
    batch = random_batch()
    base = model(**batch)
    perm = dict(batch)
    for name, size in (("zone", 4), ("type", 6)):
        p = torch.randperm(size, generator=torch.Generator().manual_seed(1))
        perm[f"{name}_daily"] = batch[f"{name}_daily"][:, :, p]
        perm[f"{name}_mask"] = batch[f"{name}_mask"][:, :, p]
    assert torch.allclose(model(**perm), base, atol=1e-5)


def test_padded_peers_are_ignored(model):
    batch = random_batch()
    base = model(**batch)
    noisy = dict(batch)
    noisy["zone_daily"] = batch["zone_daily"].clone()
    noisy["zone_daily"][~batch["zone_mask"]] = 99.0
    assert torch.allclose(model(**noisy), base, atol=1e-5)
