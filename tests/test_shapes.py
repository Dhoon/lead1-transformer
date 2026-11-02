import pytest
import torch
from torch import nn

from src.models.dual_context import DualContextTransformer, ModelConfig
from src.train.common import load_config
from tests.conftest import random_batch, small_cfg


def test_shapes_through_model(model):
    batch = random_batch(b=3, t=5, z=4, u=6)
    assert batch["target_daily"].shape == (3, 5, 24)
    assert model.patch(batch["target_daily"]).shape == (3, 5, 32)
    assert model.zone_encoder.embed(batch["zone_daily"]).shape == (3, 5 * 4, 32)
    assert model.type_encoder.embed(batch["type_daily"]).shape == (3, 5 * 6, 32)
    logits = model(**batch)
    assert logits.shape == (3, 5, 24)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("t", [1, 7, 64])
def test_output_follows_variable_t(t):
    model = DualContextTransformer(small_cfg()).eval()
    assert model(**random_batch(t=t)).shape == (3, t, 24)


def test_initial_config_matches_spec():
    cfg = ModelConfig.from_dict(load_config("configs/train.yaml")["model"])
    assert (cfg.zone_encoder_layers, cfg.type_encoder_layers, cfg.decoder_layers) == (2, 2, 2)
    assert (cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.d_ff, cfg.dropout) == (128, 4, 32, 512, 0.1)
    model = DualContextTransformer(cfg)
    assert len(model.zone_encoder.layers) == 2
    assert len(model.type_encoder.layers) == 2
    assert len(model.decoder.layers) == 2
    attn = model.decoder.layers[0].blocks[0]
    assert attn.num_heads == 4 and attn.head_dim == 32
    assert model.zone_encoder.layers[0].linear1.out_features == 512
    assert isinstance(model.head.proj, nn.Linear) and model.head.proj.out_features == 24


def test_decoder_block_order():
    layer = DualContextTransformer(ModelConfig()).decoder.layers[0]
    assert layer.block_names == ["self", "ffn", "zone", "ffn", "type", "ffn"]
    reverse = DualContextTransformer(ModelConfig(cross_attention_order=["type", "zone"])).decoder.layers[0]
    assert reverse.block_names == ["self", "ffn", "type", "ffn", "zone", "ffn"]


def test_spec_fixed_keys_are_enforced():
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"cross_attention_fusion": "parallel"})
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"exclude_target_from_peers": True})


def test_shared_day_pe_in_encoder():
    model = DualContextTransformer(small_cfg()).eval()
    daily = torch.ones(1, 3, 4, 24)  # 모든 건물·날짜가 같은 소비량
    tokens = model.zone_encoder.embed(daily).reshape(1, 3, 4, -1)
    for d in range(3):
        assert torch.allclose(tokens[0, d], tokens[0, d, :1].expand(4, -1))  # 같은 날짜 -> 같은 PE
    assert not torch.allclose(tokens[0, 0, 0], tokens[0, 1, 0])  # 다른 날짜 -> 다른 PE
    no_pe = DualContextTransformer(small_cfg(encoder_positional_encoding="none"))
    flat = no_pe.zone_encoder.embed(daily)
    assert torch.allclose(flat, flat[:, :1].expand_as(flat))


def test_context_ablation_runs_without_encoders():
    model = DualContextTransformer(small_cfg(cross_attention_order=[])).eval()
    assert not hasattr(model, "zone_encoder") and not hasattr(model, "type_encoder")
    assert model(random_batch()["target_daily"]).shape == (3, 5, 24)
