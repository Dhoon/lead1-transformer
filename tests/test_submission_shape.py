import numpy as np
import pandas as pd
import pytest

from src.data.lead_loader import grid_from_frame
from src.eval.kaggle_submission import make_submission
from tests.conftest import make_frame


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


def test_submission_follows_sample_order():
    df, meta = make_frame(labels=False)
    df.insert(0, "row_id", np.arange(len(df)) + 1000)
    grid = grid_from_frame(df, meta)
    prob = np.random.default_rng(0).random(grid.values.shape)
    sample = pd.DataFrame({"row_id": df["row_id"].sample(frac=1, random_state=0).to_numpy(), "anomaly": 0})
    sub = make_submission(prob, grid.row_id, grid.present, sample)
    assert list(sub.columns) == ["row_id", "anomaly"]
    assert (sub["row_id"].to_numpy() == sample["row_id"].to_numpy()).all()
    assert sub["anomaly"].between(0, 1).all()
    with pytest.raises(ValueError):
        make_submission(prob, grid.row_id, grid.present, pd.concat([sample, pd.DataFrame({"row_id": [-5]})]))
