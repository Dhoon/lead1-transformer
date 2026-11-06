"""날짜별 건물 Encoder (docs/LEAD_Encoder_CrossAttention_Change.md 구현 검증 기준)."""
import pytest
import torch

from src.models.dual_context import DualContextTransformer, ModelConfig
from src.models.set_encoder import JointDayEncoder, PerDayEncoder
from src.train.common import load_config
from tests.conftest import random_batch, small_cfg


def per_day_model(**kw):
    torch.manual_seed(0)
    model = DualContextTransformer(small_cfg(encoder_attention_scope="per_day_buildings", target_role_embedding=True, **kw)).eval()
    with torch.no_grad():  # Role은 0으로 초기화되므로 학습된 모델처럼 값을 넣는다
        for name, p in model.named_parameters():
            if ".role." in name:
                p.normal_()
    return model


def _permute_each_day(batch, name, size, t, seed):
    """날짜마다 다른 순서로 Peer 열(1..G-1)을 섞는다. 대상 건물(열 0)과 Padding Mask도 함께 옮긴다."""
    g = torch.Generator().manual_seed(seed)
    b = batch[f"{name}_daily"].shape[0]
    daily, mask = batch[f"{name}_daily"].clone(), batch[f"{name}_mask"].reshape(b, t, size).clone()
    for d in range(t):
        perm = torch.cat([torch.zeros(1, dtype=torch.long), 1 + torch.randperm(size - 1, generator=g)])
        daily[:, d], mask[:, d] = daily[:, d, perm], mask[:, d, perm]
    return daily, mask.reshape(b, t * size)


def test_config_builds_per_day_encoder():
    cfg = ModelConfig.from_dict(load_config("configs/v3_perday.yaml")["model"])
    assert cfg.per_day and not cfg.same_building_attention_bias and cfg.target_role_embedding
    model = DualContextTransformer(cfg)
    assert isinstance(model.zone_encoder, PerDayEncoder) and isinstance(model.type_encoder, PerDayEncoder)
    assert model.zone_encoder is not model.type_encoder
    assert not any(n.endswith(".beta") for n, _ in model.named_parameters())  # 동일 건물 Bias 경로 없음
    # 이전 Config / Checkpoint(키 없음)는 그대로 통합 Encoder
    assert isinstance(DualContextTransformer(ModelConfig()).zone_encoder, JointDayEncoder)


def test_inconsistent_scope_keys_are_rejected():
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"encoder_attention_scope": "per_day_buildings", "encoder_joint_days": True})
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"encoder_attention_scope": "per_day_buildings", "same_building_attention_bias": True})
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"encoder_attention_scope": "per_day_buildings", "encoder_internal_day_pe": True})


def test_shapes_and_memory():
    model = per_day_model()
    batch = random_batch(b=3, t=5, z=4, u=6)
    enc = model.zone_encoder
    assert enc.embed(batch["zone_daily"]).shape == (3 * 5, 4, 32)  # [B×T, G, D]
    assert enc(batch["zone_daily"], batch["zone_mask"]).shape == (3, 5, 32)  # 날짜별 대상 Context
    assert model.type_encoder(batch["type_daily"], batch["type_mask"]).shape == (3, 5, 32)
    out = model(**batch)
    assert out.shape == (3, 5, 24) and torch.isfinite(out).all()


def test_cross_attention_scores_are_t_by_t():
    model = per_day_model()
    batch = random_batch(b=2, t=6, z=4, u=6)
    shapes = []
    for layer in model.decoder.layers:
        for name, block in zip(layer.block_names, layer.blocks):
            if name in ("zone", "type"):
                block.register_forward_hook(lambda m, inp, out: shapes.append(inp[1].shape))
    model(**batch)
    assert shapes and all(s == (2, 6, 32) for s in shapes)  # Key 길이 T -> Score [B, H, T, T]


@pytest.mark.parametrize("t", [1, 7, 64])
def test_variable_t(t):
    assert per_day_model()(**random_batch(t=t)).shape == (3, t, 24)


def test_peer_order_per_day_does_not_change_prediction():
    model = per_day_model()
    batch = random_batch(t=5, z=4, u=6)
    perm = dict(batch)
    perm["zone_daily"], perm["zone_mask"] = _permute_each_day(batch, "zone", 4, 5, seed=1)
    perm["type_daily"], perm["type_mask"] = _permute_each_day(batch, "type", 6, 5, seed=2)
    assert torch.allclose(model(**perm), model(**batch), atol=1e-5)


def test_day_d_context_depends_only_on_day_d():
    enc = per_day_model().zone_encoder
    batch = random_batch(b=2, t=5, z=4, u=6)
    base = enc(batch["zone_daily"], batch["zone_mask"])
    changed = batch["zone_daily"].clone()
    changed[:, 2, 1:] += 3.0  # 2일차 Peer만 바꾼다
    out = enc(changed, batch["zone_mask"])
    others = [0, 1, 3, 4]
    assert torch.allclose(out[:, others], base[:, others], atol=1e-5)
    assert not torch.allclose(out[:, 2], base[:, 2], atol=1e-4)


def test_context_day_pe_added_after_encoder():
    model = per_day_model()
    enc = model.zone_encoder
    daily = torch.ones(1, 3, 4, 24)  # 모든 날짜가 같은 값
    mask = torch.ones(1, 3 * 4, dtype=torch.bool)
    tokens = enc.embed(daily).reshape(1, 3, 4, -1)
    assert torch.allclose(tokens[0, 0], tokens[0, 1])  # Encoder 입력에는 날짜 PE가 없다
    c = enc(daily, mask)
    assert torch.allclose(c[0, 1] - c[0, 0], model.day_pe(3)[1] - model.day_pe(3)[0], atol=1e-5)
    no_pe = per_day_model(encoder_context_day_pe="none").zone_encoder(daily, mask)
    assert torch.allclose(no_pe[0, 0], no_pe[0, 1], atol=1e-6)


def test_padded_peers_are_ignored():
    model = per_day_model()
    batch = random_batch()
    base = model(**batch)
    noisy = dict(batch)
    b, t, z, _ = batch["zone_daily"].shape
    pad = ~batch["zone_mask"].reshape(b, t, z)
    noisy["zone_daily"] = batch["zone_daily"].clone()
    noisy["zone_daily"][pad] = 99.0
    assert torch.allclose(model(**noisy), base, atol=1e-5)
