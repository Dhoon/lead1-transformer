import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import WindowDataset, inference_batches, inference_windows
from src.data.day_patches import LogStandardizer, impute_missing
from src.data.lead_loader import grid_from_frame, load_grid, save_grid
from src.data.peer_groups import group_members
from src.eval.kaggle_submission import make_submission
from src.models.dual_context import DualContextTransformer
from src.train.train_supervised import predict
from tests.conftest import make_frame, small_cfg


def test_grid_row_mapping_round_trip():
    df, meta = make_frame()
    grid = grid_from_frame(df, meta)
    assert grid.values.shape == (6, 10, 24)
    assert grid.present.sum() == len(df)
    b = pd.Index(grid.building_ids).get_indexer(df["building_id"])
    delta = pd.to_datetime(df["timestamp"]) - grid.start
    d, h = delta.dt.days.to_numpy(), (delta.dt.seconds // 3600).to_numpy()
    assert (grid.row_id[b, d, h] == df.index.to_numpy()).all()
    np.testing.assert_array_equal(grid.anomaly[b, d, h], df["anomaly"].to_numpy())
    np.testing.assert_allclose(grid.values[b, d, h], df["meter_reading"].to_numpy(), equal_nan=True)


@pytest.mark.parametrize("window,stride", [(4, 4), (4, 1), (64, 64), (3, 2)])
def test_inference_windows_cover_every_day(window, stride):
    n_days = 10
    cover = np.zeros(n_days, dtype=int)
    for s, t in inference_windows(n_days, window, stride):
        assert 1 <= t <= window and s + t <= n_days
        cover[s : s + t] += 1
    assert (cover >= 1).all()
    if window == stride:
        assert (cover == 1).all()  # 겹침 없음 -> 한 행당 Window 1개


def test_end_to_end_test_inference_to_submission():
    """레이블 없는 Test 건물로 전체 모델(Encoder 포함)을 돌려 row_id마다 점수 1개를 만든다."""
    df, meta = make_frame(labels=False)
    df.insert(0, "row_id", np.arange(len(df)) + 1000)
    grid = grid_from_frame(df, meta)
    x = LogStandardizer().fit(grid.values, grid.observed).transform(impute_missing(grid.values, "zero_fill"))
    targets = np.arange(len(grid.building_ids))
    ds = WindowDataset(
        x, targets, group_members(grid.site_id, targets, targets),
        group_members(grid.primary_use, targets, targets), grid.observed,
    )
    model = DualContextTransformer(small_cfg())
    batches = inference_batches(len(targets), grid.n_days, 4, 2, 8, 6, None)
    prob = predict(model, ds, batches, torch.device("cpu"))
    assert prob.shape == grid.values.shape and np.isfinite(prob).all()

    sample = pd.DataFrame({"row_id": df["row_id"].sample(frac=1, random_state=0).to_numpy(), "anomaly": 0})
    sub = make_submission(prob, grid.row_id, grid.present, sample)
    assert list(sub.columns) == ["row_id", "anomaly"]
    assert (sub["row_id"].to_numpy() == sample["row_id"].to_numpy()).all()
    assert sub["anomaly"].between(0, 1).all()
    # 결측 meter_reading Row도 누락 없이 점수를 받는다.
    missing_rows = df.loc[df["meter_reading"].isna(), "row_id"]
    assert sub.set_index("row_id").loc[missing_rows, "anomaly"].notna().all()
    with pytest.raises(ValueError):
        make_submission(prob, grid.row_id, grid.present, pd.concat([sample, pd.DataFrame({"row_id": [-5]})]))


def test_grid_cache_round_trip(tmp_path):
    df, meta = make_frame()
    grid = grid_from_frame(df, meta)
    save_grid(grid, tmp_path / "grid.npz")
    loaded, _ = load_grid(tmp_path / "grid.npz")
    for name in ("building_ids", "site_id", "primary_use", "values", "present", "anomaly", "row_id"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(grid, name))
    assert loaded.start == grid.start
