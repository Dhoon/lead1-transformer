"""Target + Peer Summary 2T Cross-Attention (docs/LEAD_PeerSummary_2T_CrossAttention_Change.md 6장 검증 항목)."""
import pytest
import torch

from src.models.dual_context import DualContextTransformer, ModelConfig
from src.train.common import load_config
from tests.conftest import random_batch, small_cfg
from tests.test_per_day_encoder import _permute_each_day


def peer_mean_model(**kw):
    torch.manual_seed(0)
    kw = {"encoder_context_pooling": "target_plus_peer_mean", **kw}
    model = DualContextTransformer(small_cfg(
        encoder_attention_scope="per_day_buildings", target_role_embedding=True, **kw,
    )).eval()
    with torch.no_grad():  # Role은 0으로 초기화되므로 학습된 모델처럼 값을 넣는다
        for name, p in model.named_parameters():
            if ".role." in name:
                p.normal_()
    return model


def _encoder_output(enc, daily, mask):
    """Encoder 최종 LayerNorm까지 거친 날짜별 전체 건물 출력 `[B, T, G, D]` (Pooling 전)."""
    b, t, g, _ = daily.shape
    h = enc.embed(daily)
    for layer in enc.layers:
        h = layer(h, src_key_padding_mask=~mask.reshape(b * t, g))
    return enc.norm(h).reshape(b, t, g, -1)


def test_config_and_param_count_unchanged():
    cfg = ModelConfig.from_dict(load_config("configs/v3_peermean.yaml")["model"])
    assert cfg.per_day and cfg.encoder_context_pooling == "target_plus_peer_mean"
    base = ModelConfig.from_dict(load_config("configs/v3_perday.yaml")["model"])
    count = lambda c: sum(p.numel() for p in DualContextTransformer(c).parameters())
    assert count(cfg) == count(base)  # Pooling만 바뀌고 새 파라미터는 없다


def test_invalid_pooling_is_rejected():
    with pytest.raises(ValueError):
        ModelConfig(encoder_context_pooling="target_plus_peer_mean")  # joint_days에서는 불가
    with pytest.raises(ValueError):
        small_cfg(encoder_attention_scope="per_day_buildings", encoder_context_pooling="all_peers")


def test_memory_shapes_and_cross_attention_scores_are_t_by_2t():
    model = peer_mean_model()
    batch = random_batch(b=2, t=6, z=4, u=6)
    mem, mask = model.zone_encoder(batch["zone_daily"], batch["zone_mask"])
    assert mem.shape == (2, 12, 32) and mask.shape == (2, 12)
    assert model.type_encoder(batch["type_daily"], batch["type_mask"])[0].shape == (2, 12, 32)
    keys = []
    for layer in model.decoder.layers:
        for name, block in zip(layer.block_names, layer.blocks):
            if name in ("zone", "type"):
                block.register_forward_hook(lambda m, inp, out: keys.append(inp[1].shape))
    out = model(**batch)
    assert out.shape == (2, 6, 24) and torch.isfinite(out).all()
    assert keys and all(s == (2, 12, 32) for s in keys)  # Key 길이 2T -> Score [B, H, T, 2T]


def test_target_slot_is_slot0_and_peer_slot_is_valid_peer_mean():
    enc = peer_mean_model(encoder_context_day_pe="none").zone_encoder
    batch = random_batch(b=3, t=4, z=4, u=6)  # 샘플 0: Peer 2개(1개 Padding), 샘플 2: Peer 0개
    daily, mask = batch["zone_daily"], batch["zone_mask"]
    mem, _ = enc(daily, mask)
    mem = mem.reshape(3, 4, 2, -1)
    h = _encoder_output(enc, daily, mask)
    assert torch.allclose(mem[:, :, 0], h[:, :, 0], atol=1e-5)
    assert torch.allclose(mem[0, :, 1], h[0, :, 1:3].mean(dim=1), atol=1e-5)  # Padding 슬롯 3 제외
    assert torch.allclose(mem[1, :, 1], h[1, :, 1:].mean(dim=1), atol=1e-5)


def test_target_slot_matches_v3_target_context():
    v3 = peer_mean_model(encoder_context_pooling="target_token")
    new = peer_mean_model()
    batch = random_batch(b=2, t=5)
    c3, _ = v3.zone_encoder(batch["zone_daily"], batch["zone_mask"])
    c2, _ = new.zone_encoder(batch["zone_daily"], batch["zone_mask"])
    assert torch.allclose(c2.reshape(2, 5, 2, -1)[:, :, 0], c3, atol=1e-5)


def test_padded_peers_are_ignored():
    model = peer_mean_model()
    batch = random_batch()
    base = model(**batch)
    noisy = dict(batch)
    b, t, z, _ = batch["zone_daily"].shape
    pad = ~batch["zone_mask"].reshape(b, t, z)
    noisy["zone_daily"] = batch["zone_daily"].clone()
    noisy["zone_daily"][pad] = 99.0
    assert torch.allclose(model(**noisy), base, atol=1e-5)


def test_peer_order_per_day_does_not_change_summary_or_prediction():
    model = peer_mean_model()
    batch = random_batch(t=5, z=4, u=6)
    perm = dict(batch)
    perm["zone_daily"], perm["zone_mask"] = _permute_each_day(batch, "zone", 4, 5, seed=1)
    perm["type_daily"], perm["type_mask"] = _permute_each_day(batch, "type", 6, 5, seed=2)
    a, _ = model.zone_encoder(batch["zone_daily"], batch["zone_mask"])
    b, _ = model.zone_encoder(perm["zone_daily"], perm["zone_mask"])
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(model(**perm), model(**batch), atol=1e-5)


def test_no_peer_day_slot_is_masked_and_ignored():
    model = peer_mean_model()
    batch = random_batch(b=3, t=4, z=4, u=6)  # 샘플 2의 Zone 그룹은 대상 건물 단독
    _, mask = model.zone_encoder(batch["zone_daily"], batch["zone_mask"])
    mask = mask.reshape(3, 4, 2)
    assert mask[:, :, 0].all()                                   # 대상 슬롯은 항상 유효
    assert mask[:2, :, 1].all() and not mask[2, :, 1].any()     # Peer 0개 -> Peer 슬롯 마스킹
    # 단독 그룹만 있는 배치(G=1)도 동작하고, 대상 단독 결과는 v3 Target-only와 같다
    solo = {k: v for k, v in batch.items()}
    solo["zone_daily"], solo["zone_mask"] = batch["zone_daily"][:, :, :1], torch.ones(3, 4, dtype=torch.bool)
    solo["type_daily"], solo["type_mask"] = batch["type_daily"][:, :, :1], torch.ones(3, 4, dtype=torch.bool)
    out = model(**solo)
    assert torch.isfinite(out).all()
    v3 = peer_mean_model(encoder_context_pooling="target_token")
    assert torch.allclose(out, v3(**solo), atol=1e-5)


def test_same_day_pe_on_both_slots():
    model = peer_mean_model()
    enc = model.zone_encoder
    daily = torch.randn(1, 3, 4, 24)
    mask = torch.ones(1, 12, dtype=torch.bool)
    with_pe, _ = enc(daily, mask)
    no_pe, _ = peer_mean_model(encoder_context_day_pe="none").zone_encoder(daily, mask)
    diff = (with_pe - no_pe).reshape(3, 2, -1)
    pe = model.day_pe(3)
    assert torch.allclose(diff[:, 0], pe, atol=1e-5) and torch.allclose(diff[:, 1], pe, atol=1e-5)


@pytest.mark.parametrize("t", [1, 7, 64])
def test_variable_t(t):
    assert peer_mean_model()(**random_batch(t=t)).shape == (3, t, 24)


# ---- C안: 날짜별 Encoder 출력 전체(T×G)를 K/V로 (encoder_context_pooling: all_buildings) ----

def all_model(**kw):
    return peer_mean_model(encoder_context_pooling="all_buildings", **kw)


def test_all_buildings_config_and_shapes():
    cfg = ModelConfig.from_dict(load_config("configs/v3_allpeer.yaml")["model"])
    assert cfg.per_day and cfg.encoder_context_pooling == "all_buildings"
    base = ModelConfig.from_dict(load_config("configs/v3_perday.yaml")["model"])
    count = lambda c: sum(p.numel() for p in DualContextTransformer(c).parameters())
    assert count(cfg) == count(base)
    model = all_model()
    batch = random_batch(b=2, t=6, z=4, u=6)
    mem, mask = model.zone_encoder(batch["zone_daily"], batch["zone_mask"])
    assert mem.shape == (2, 24, 32) and torch.equal(mask, batch["zone_mask"])
    keys = []
    for layer in model.decoder.layers:
        for name, block in zip(layer.block_names, layer.blocks):
            if name in ("zone", "type"):
                block.register_forward_hook(lambda m, inp, out: keys.append(inp[1].shape[1]))
    assert model(**batch).shape == (2, 6, 24)
    assert sorted(set(keys)) == [24, 36]  # Zone T×4, Type T×6 -> Score [B, H, T, T×G]


def test_all_buildings_tokens_equal_encoder_output_plus_day_pe():
    model = all_model()
    enc = model.zone_encoder
    batch = random_batch(b=2, t=5, z=4)
    mem, _ = enc(batch["zone_daily"], batch["zone_mask"])
    h = _encoder_output(enc, batch["zone_daily"], batch["zone_mask"])
    expect = h + model.day_pe(5)[None, :, None, :]  # 같은 날짜의 모든 건물에 같은 PE
    assert torch.allclose(mem, expect.reshape(2, 20, -1), atol=1e-5)
    v3, _ = peer_mean_model(encoder_context_pooling="target_token").zone_encoder(batch["zone_daily"], batch["zone_mask"])
    assert torch.allclose(mem.reshape(2, 5, 4, -1)[:, :, 0], v3, atol=1e-5)  # 대상 Token은 v3와 같다


def test_all_buildings_padding_and_peer_order():
    model = all_model()
    batch = random_batch(t=5, z=4, u=6)
    base = model(**batch)
    noisy = dict(batch)
    b, t, z, _ = batch["zone_daily"].shape
    noisy["zone_daily"] = batch["zone_daily"].clone()
    noisy["zone_daily"][~batch["zone_mask"].reshape(b, t, z)] = 99.0
    assert torch.allclose(model(**noisy), base, atol=1e-5)
    perm = dict(batch)
    perm["zone_daily"], perm["zone_mask"] = _permute_each_day(batch, "zone", 4, 5, seed=1)
    perm["type_daily"], perm["type_mask"] = _permute_each_day(batch, "type", 6, 5, seed=2)
    assert torch.allclose(model(**perm), base, atol=1e-5)


@pytest.mark.parametrize("t", [1, 7, 64])
def test_all_buildings_variable_t(t):
    assert all_model()(**random_batch(t=t)).shape == (3, t, 24)
