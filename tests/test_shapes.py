import torch
from torch import nn

from src.models.dual_context import DualContextTransformer, ModelConfig
from src.models.heads import BuildingIdHead, HourlyAnomalyHead
from tests.conftest import random_batch, small_cfg


def test_patch_and_decoder_shapes(model):
    batch = random_batch(b=3, t=5)
    assert batch["target_daily"].shape == (3, 5, 24)
    assert model.patch(batch["target_daily"]).shape == (3, 5, 32)
    h = model(**batch)
    assert h.shape == (3, 5, 32)
    assert torch.isfinite(h).all()


def test_pretrain_logits_shape_and_d_n_independent():
    model = DualContextTransformer(small_cfg(d_model=128, n_heads=4, d_ff=512)).eval()
    head = BuildingIdHead(128, 200)
    h = model(**random_batch())
    assert head(h).shape == (3, 5, 200)
    assert head(h[:, -1]).shape == (3, 200)


def test_hourly_head_shape(model):
    assert HourlyAnomalyHead(32)(model(**random_batch())).shape == (3, 5, 24)


def test_initial_config_matches_spec():
    cfg = ModelConfig()
    assert (cfg.zone_encoder_layers, cfg.type_encoder_layers, cfg.decoder_layers) == (2, 2, 2)
    assert (cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.d_ff, cfg.dropout) == (128, 4, 32, 512, 0.1)
    model = DualContextTransformer(cfg)
    assert len(model.zone_encoder.layers) == 2
    assert len(model.type_encoder.layers) == 2
    assert len(model.decoder.layers) == 2
    attn = model.decoder.layers[0].self_attn
    assert attn.num_heads == 4 and attn.head_dim == 32
    ffn = model.zone_encoder.layers[0].linear1
    assert isinstance(ffn, nn.Linear) and ffn.out_features == 512


def test_ablation_without_encoders():
    model = DualContextTransformer(small_cfg(use_zone_encoder=False, use_type_encoder=False)).eval()
    assert model(random_batch()["target_daily"]).shape == (3, 5, 32)
