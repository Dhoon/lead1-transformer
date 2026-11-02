import torch

from tests.conftest import random_batch


def _permute_peers(batch, name, size, t, seed):
    """대상 건물(열 0)은 그대로 두고 Peer 열 순서를 모든 날짜에서 같이 섞는다."""
    perm = torch.cat([torch.zeros(1, dtype=torch.long), 1 + torch.randperm(size - 1, generator=torch.Generator().manual_seed(seed))])
    b = batch[f"{name}_daily"].shape[0]
    daily = batch[f"{name}_daily"][:, :, perm]
    mask = batch[f"{name}_mask"].reshape(b, t, size)[:, :, perm].reshape(b, t * size)
    return daily, mask


def test_peer_order_does_not_change_prediction(model):
    batch = random_batch(t=5, z=4, u=6)
    base = model(**batch)
    perm = dict(batch)
    perm["zone_daily"], perm["zone_mask"] = _permute_peers(batch, "zone", 4, 5, seed=1)
    perm["type_daily"], perm["type_mask"] = _permute_peers(batch, "type", 6, 5, seed=2)
    assert torch.allclose(model(**perm), base, atol=1e-5)


def test_padded_group_tokens_are_ignored(model):
    batch = random_batch()
    base = model(**batch)
    noisy = dict(batch)
    b, t, z, _ = batch["zone_daily"].shape
    pad = ~batch["zone_mask"].reshape(b, t, z)
    noisy["zone_daily"] = batch["zone_daily"].clone()
    noisy["zone_daily"][pad] = 99.0
    assert torch.allclose(model(**noisy), base, atol=1e-5)
