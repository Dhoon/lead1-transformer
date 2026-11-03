"""v1.0 Encoder: 동일 건물 Attention Bias, Target Role Embedding, encoder_peers (SPEC 4.2.1, 10)."""
import numpy as np
import pytest
import torch

from src.models.dual_context import DualContextTransformer
from src.models.set_encoder import SameBuildingSelfAttention, slot_index
from src.train.common import build_groups
from tests.conftest import random_batch, small_cfg
from tests.test_encoder_permutation import _permute_peers


def v1_model(**kw):
    torch.manual_seed(0)
    model = DualContextTransformer(small_cfg(target_role_embedding=True, same_building_attention_bias=True, **kw)).eval()
    # β, Role은 0으로 초기화되므로 학습된 모델처럼 0이 아닌 값을 넣어 효과를 드러낸다.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".beta"):
                p.fill_(2.0)
            if ".role." in name:
                p.normal_()
    return model


def _swap_day(batch, name, size, t, day, a, b):
    """상황 B: `day`일에만 Peer a와 b의 값을 맞바꾼다. 날짜별 값 집합은 그대로다."""
    daily = batch[f"{name}_daily"].clone()
    daily[:, day, [a, b]] = daily[:, day, [b, a]]
    return daily


def test_bias_matches_explicit_logit_bias():
    """one-hot 이어 붙이기 구현이 `QK/√d + β·1[같은 slot]`과 같다."""
    torch.manual_seed(0)
    attn = SameBuildingSelfAttention(16, 2, 0.0, bias_init=1.5).eval()
    t, g = 3, 4
    x = torch.randn(2, t * g, 16)
    valid = torch.ones(2, t * g, dtype=torch.bool)
    valid[1, 3::g] = False
    slots = slot_index(t, g)
    out = attn(x, valid, slots, g)

    b, l, _ = x.shape
    q, k, v = attn.qkv(x).view(b, l, 3, 2, 8).permute(2, 0, 3, 1, 4)
    logits = q @ k.transpose(-1, -2) / 8**0.5 + attn.beta[None, :, None, None] * (slots[:, None] == slots[None, :])
    logits = logits.masked_fill(~valid[:, None, None, :], float("-inf"))
    ref = (logits.softmax(-1) @ v).transpose(1, 2).reshape(b, l, 16)
    assert torch.allclose(out, attn.out(ref), atol=1e-5)


@pytest.mark.parametrize("order", [["zone", "type"], ["type", "zone"]])
def test_peer_order_invariant_with_bias(order):
    model = v1_model(cross_attention_order=order)
    batch = random_batch(t=5, z=4, u=6)
    perm = dict(batch)
    perm["zone_daily"], perm["zone_mask"] = _permute_peers(batch, "zone", 4, 5, seed=1)
    perm["type_daily"], perm["type_mask"] = _permute_peers(batch, "type", 6, 5, seed=2)
    assert torch.allclose(model(**perm), model(**batch), atol=1e-5)


def test_cross_day_swap_is_distinguished_only_with_bias():
    """같은 날짜의 값 집합이 같아도, 날짜 간 동일 건물 연결이 바뀌면 Bias가 있을 때만 출력이 달라진다."""
    batch = random_batch(b=2, t=4, z=4, u=6)
    swapped = dict(batch)
    swapped["zone_daily"] = _swap_day(batch, "zone", 4, 4, day=1, a=1, b=2)

    torch.manual_seed(0)
    v09 = DualContextTransformer(small_cfg(cross_attention_order=["zone"])).eval()
    assert torch.allclose(v09(**swapped), v09(**batch), atol=1e-5)

    v1 = v1_model(cross_attention_order=["zone"])
    assert not torch.allclose(v1(**swapped), v1(**batch), atol=1e-4)


def test_role_embedding_only_on_target_slot():
    model = v1_model(cross_attention_order=["zone"])
    enc = model.zone_encoder
    daily = torch.randn(2, 3, 4, 24)
    with_role = enc.embed(daily).reshape(2, 3, 4, -1)
    enc_role, enc.role = enc.role, None
    without = enc.embed(daily).reshape(2, 3, 4, -1)
    enc.role = enc_role
    diff = with_role - without
    assert torch.allclose(diff[:, :, 0], enc.role.weight[0].expand_as(diff[:, :, 0]))
    assert torch.allclose(diff[:, :, 1:], enc.role.weight[1].expand_as(diff[:, :, 1:]))
    assert enc.role.weight.shape[0] == 2  # 건물 수와 무관


def test_encoder_peers_target_only():
    class G:
        site_id = np.array([0, 0, 0, 1])
        primary_use = np.array(["a", "a", "b", "b"])

    pool = np.arange(4)
    zone, typ = build_groups(G, pool, pool, small_cfg(encoder_peers="target_only"))
    assert zone.shape == (4, 1) and typ.shape == (4, 1)
    assert (zone[:, 0] == pool).all() and (typ[:, 0] == pool).all()
    full, _ = build_groups(G, pool, pool, small_cfg())
    assert full.shape[1] == 3


def test_v1_forward_shapes_and_padding():
    model = v1_model()
    batch = random_batch()
    out = model(**batch)
    assert out.shape == (3, 5, 24) and torch.isfinite(out).all()
    noisy = dict(batch)
    b, t, z, _ = batch["zone_daily"].shape
    pad = ~batch["zone_mask"].reshape(b, t, z)
    noisy["zone_daily"] = batch["zone_daily"].clone()
    noisy["zone_daily"][pad] = 99.0
    assert torch.allclose(model(**noisy), out, atol=1e-5)
