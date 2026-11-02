import inspect

import numpy as np
import pytest
import torch
from torch import nn

from src.data.dataset import WindowDataset, collate_windows
from src.models.dual_context import DualContextTransformer, ModelConfig, model_inputs


def test_model_has_no_id_embedding_or_id_argument():
    model = DualContextTransformer(ModelConfig())
    assert not any(isinstance(m, nn.Embedding) for m in model.modules())
    params = inspect.signature(model.forward).parameters
    assert not any("building" in p or p.endswith("_id") or "anomaly" in p or "observation" in p for p in params)
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"id_lookup_embedding": True})


def test_labels_and_masks_not_passed_to_model():
    x = np.random.default_rng(0).normal(size=(3, 6, 24)).astype(np.float32)
    labels = np.ones((3, 6, 24), dtype=np.int8)
    ds = WindowDataset(
        x, np.array([0]), np.array([[0, 1]]), None, np.ones_like(x, dtype=bool),
        labels, np.ones_like(x, dtype=bool),
    )
    batch = collate_windows([ds[(0, 1, 3)]])
    assert set(model_inputs(batch)) == {"target_daily", "zone_daily", "zone_mask"}


def test_anomaly_hours_kept_in_inputs_and_targets():
    """라벨에 근거해 입력이나 날짜를 제외·마스킹하지 않는다 (SPEC 5.1)."""
    x = np.random.default_rng(1).normal(size=(2, 4, 24)).astype(np.float32)
    labels = np.zeros((2, 4, 24), dtype=np.int8)
    labels[0, 2, 5:9] = 1
    ds = WindowDataset(
        x, np.array([0]), np.array([[0, 1]]), None, np.ones_like(x, dtype=bool),
        labels, np.ones_like(x, dtype=bool),
    )
    item = ds[(0, 0, 4)]
    assert torch.equal(item["target_daily"], torch.from_numpy(x[0]))
    assert item["anomaly_target"][2, 5:9].eq(1).all()
    assert item["label_mask"].all()
