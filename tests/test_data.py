import numpy as np
import torch

from src.data.dataset import RandomWindowBatchSampler, WindowDataset, collate_windows
from src.data.day_patches import LogStandardizer, impute_missing
from src.data.peer_groups import group_members


def test_zero_fill_v1_only_replaces_nan():
    values = np.array([[[np.nan, 0.0, 5.0], [3.0, np.nan, 0.0]]], dtype=np.float32)
    filled = impute_missing(values, "zero_fill")
    assert not np.isnan(filled).any()
    np.testing.assert_array_equal(filled, [[[0.0, 0.0, 5.0], [3.0, 0.0, 0.0]]])


def test_building_median_v2():
    values = np.array([
        [[1.0, np.nan], [3.0, 5.0]],
        [[np.nan, np.nan], [np.nan, np.nan]],  # 관측 없음 -> 0
    ], dtype=np.float32)
    filled = impute_missing(values, "building_median")
    assert not np.isnan(filled).any()
    assert filled[0, 0, 1] == 3.0 and filled[0, 1, 1] == 5.0
    assert (filled[1] == 0).all()


def test_normalized_input_has_no_nan():
    values = np.array([[[np.nan, 10.0, 100.0]]], dtype=np.float32)
    norm = LogStandardizer().fit(values, ~np.isnan(values))
    assert np.isfinite(norm.transform(impute_missing(values, "zero_fill"))).all()


def test_group_target_first():
    key = np.array([0, 0, 1, 0, 1])
    out = group_members(key, np.array([3, 2]), np.array([0, 1, 2, 3, 4]))
    assert out[0, 0] == 3 and set(out[0][out[0] >= 0]) == {0, 1, 3}
    assert out[1, 0] == 2 and set(out[1][out[1] >= 0]) == {2, 4}
    alone = group_members(key, np.array([2]), np.array([0, 1]))
    assert alone.tolist() == [[2]]  # Peer가 없으면 대상 단독


def test_encoder_first_token_equals_target_same_dates():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 10, 24)).astype(np.float32)
    ds = WindowDataset(x, np.array([2, 0]), np.array([[2, 0, 1], [0, 3, -1]]), np.array([[2, 3, -1], [0, 1, 2]]), np.ones_like(x, dtype=bool))
    batch = collate_windows([ds[(0, 3, 5)], ds[(1, 3, 5)]])
    assert torch.equal(batch["zone_daily"][:, :, 0], batch["target_daily"])
    assert torch.equal(batch["type_daily"][:, :, 0], batch["target_daily"])
    assert torch.equal(batch["zone_daily"][0, :, 1], torch.from_numpy(x[0, 3:8]))  # 같은 날짜 구간
    assert batch["zone_mask"].shape == (2, 5 * 3)
    assert batch["zone_mask"].reshape(2, 5, 3)[1, :, 2].logical_not().all()  # 그룹 Padding만 Mask


def test_sampler_uniform_t_contiguous_windows():
    sampler = RandomWindowBatchSampler(
        n_targets=5, n_steps=366, min_t=1, max_t=64, samples_per_epoch=20000,
        batch_size=8, group_size=10, max_tokens=None, seed=0,
    )
    ts = []
    for batch in sampler:
        t = batch[0][2]
        assert all(k[2] == t for k in batch)  # 배치 내 같은 T
        assert all(0 <= s and s + t <= 366 for _, s, _ in batch)
        ts.append(t)
    assert min(ts) == 1 and max(ts) == 64
    assert abs(np.mean(ts) - 32.5) < 2.0


def test_token_budget_shrinks_batch_for_long_t():
    sampler = RandomWindowBatchSampler(5, 366, 64, 64, 10, 32, group_size=80, max_tokens=16384, seed=0)
    assert len(next(iter(sampler))) == 16384 // (64 * 80)
