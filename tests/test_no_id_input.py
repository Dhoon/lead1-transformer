import inspect

import numpy as np
import pytest
import torch
from torch import nn

from src.data.dataset import IGNORE_INDEX, WindowDataset
from src.models.dual_context import DualContextTransformer, ModelConfig, backbone_inputs
from src.models.heads import BuildingIdHead
from src.train.pretrain import id_loss
from tests.conftest import random_batch, small_cfg


def test_model_has_no_id_embedding_or_id_argument():
    model = DualContextTransformer(ModelConfig())
    assert not any(isinstance(m, nn.Embedding) for m in model.modules())
    params = inspect.signature(model.forward).parameters
    assert not any("building" in p or p.endswith("_id") for p in params)
    with pytest.raises(ValueError):
        ModelConfig(id_lookup_embedding=True)


def test_building_target_not_passed_to_model():
    batch = random_batch()
    batch["building_target"] = torch.zeros(3, 5, dtype=torch.long)
    assert "building_target" not in backbone_inputs(batch)


def test_dataset_input_is_readings_only():
    x = np.random.default_rng(0).normal(size=(3, 6, 24)).astype(np.float32)
    ds = WindowDataset(
        x, np.ones_like(x, dtype=bool), np.array([2]), 3, [5],
        class_index=np.array([7]), day_ok=np.ones((3, 6), dtype=bool),
    )
    item = ds[0]
    assert torch.equal(item["target_daily"], torch.from_numpy(x[2, 3:6]))
    assert item["building_target"].tolist() == [7, 7, 7]


def test_cross_entropy_sends_gradient_to_all_classes():
    model = DualContextTransformer(small_cfg()).eval()
    head = BuildingIdHead(32, 200)
    logits = head(model(**random_batch(b=1)))
    logits.retain_grad()
    target = torch.full((1, 5), IGNORE_INDEX)
    target[0, -1] = 1
    id_loss(logits, target, "all_valid_days").backward()
    grad = logits.grad[0, -1]
    assert (grad.abs() > 0).all()
    expected = torch.softmax(logits[0, -1], -1).detach() - torch.eye(200)[1]
    assert torch.allclose(grad, expected, atol=1e-6)
    assert (head.proj.weight.grad.abs().sum(1) > 0).all()
