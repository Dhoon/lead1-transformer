import numpy as np
import pandas as pd
import pytest
import torch

from src.models.dual_context import DualContextTransformer, ModelConfig


def make_frame(n_buildings=6, n_days=10, seed=0, labels=True, drop=5):
    """LEAD train.csv 형식의 합성 데이터. 일부 Row를 빼서 결측 시각을 만든다."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2016-01-01", periods=n_days * 24, freq="h")
    rows = []
    for b in range(n_buildings):
        reading = rng.gamma(2.0, 50.0, size=len(ts))
        reading[rng.choice(len(ts), size=drop, replace=False)] = np.nan
        rows.append(pd.DataFrame({"building_id": 100 + b, "timestamp": ts, "meter_reading": reading}))
    df = pd.concat(rows, ignore_index=True)
    if labels:
        df["anomaly"] = (rng.random(len(df)) < 0.05).astype(int)
    df = df.drop(index=rng.choice(len(df), size=drop, replace=False)).reset_index(drop=True)
    meta = pd.DataFrame({
        "building_id": 100 + np.arange(n_buildings),
        "site_id": np.arange(n_buildings) % 2,
        "primary_use": np.where(np.arange(n_buildings) % 3 == 0, "Office", "Education"),
    })
    return df, meta


def small_cfg(**kw):
    base = dict(d_model=32, n_heads=4, head_dim=None, d_ff=64, dropout=0.0, decoder_window_days=5)
    base.update(kw)
    return ModelConfig(**base)


def random_batch(b=3, t=5, z=4, u=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    zone_mask = torch.ones(b, t, z, dtype=torch.bool)
    zone_mask[0, :, -1] = False
    zone_mask[-1, 2] = False  # 빈 Zone 그룹 -> NULL_CONTEXT
    type_mask = torch.ones(b, t, u, dtype=torch.bool)
    type_mask[-1, :, -2:] = False
    return {
        "target_daily": torch.randn(b, t, 24, generator=g),
        "zone_daily": torch.randn(b, t, z, 24, generator=g),
        "zone_mask": zone_mask,
        "type_daily": torch.randn(b, t, u, 24, generator=g),
        "type_mask": type_mask,
    }


@pytest.fixture(params=["parallel", "sequential", "gate"])
def fusion(request):
    return request.param


@pytest.fixture
def model(fusion):
    torch.manual_seed(0)
    return DualContextTransformer(small_cfg(cross_attention_fusion=fusion)).eval()
