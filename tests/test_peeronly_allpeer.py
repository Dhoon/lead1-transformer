"""Peer-only Encoder + All-Peer K/V + Availability-aware Parallel Fusion
(docs/LEAD_v3pm_PeerOnly_AllPeer_Parallel_Change.md 8장 필수 테스트)."""
import pytest
import torch

from src.models.dual_context import DualContextTransformer, ModelConfig
from src.models.temporal_decoder import ParallelFusionDecoderLayer
from src.train.common import load_config
from tests.conftest import random_batch, small_cfg

PEER_ONLY = dict(
    encoder_attention_scope="per_day_buildings", encoder_input_scope="peers_only",
    encoder_context_pooling="all_peers", cross_attention_memory_layout="day_major_peer",
    cross_attention_fusion="parallel_concat_mlp", fusion_include_availability=True,
)


def po_model(**kw):
    torch.manual_seed(0)
    return DualContextTransformer(small_cfg(**{**PEER_ONLY, **kw})).eval()


def _memories(model, batch):
    return {c: getattr(model, f"{c}_encoder")(batch[f"{c}_daily"], batch[f"{c}_mask"]) for c in ("zone", "type")}


def _fusion_inputs(model, batch):
    rec = []
    for layer in model.decoder.layers:
        layer.context_fusion.register_forward_hook(lambda m, inp, out: rec.append((inp[0], out)))
    model(**batch)
    return rec


def test_config_file():
    m = ModelConfig.from_dict(load_config("configs/v3_pm_peeronly_allpeer_parallel.yaml")["model"])
    assert (m.encoder_input_scope, m.encoder_context_pooling, m.decoder_layers) == ("peers_only", "all_peers", 2)
    model = DualContextTransformer(m)
    assert all(isinstance(layer, ParallelFusionDecoderLayer) and layer.availability for layer in model.decoder.layers)
    fusion = model.decoder.layers[0].context_fusion
    assert (fusion[0].in_features, fusion[0].out_features, fusion[2].out_features) == (386, 128, 128)  # 3D+2 -> D -> D
    assert model.zone_encoder.role is None and model.type_encoder.role is None


def test_invalid_combinations_rejected():
    bad = [
        {"encoder_context_pooling": "all_buildings"},             # peers_only인데 all_peers가 아님
        {"encoder_input_scope": "target_and_peers"},              # all_peers인데 대상 포함
        {"target_role_embedding": True},                          # Peer-only에 Role Embedding
        {"cross_attention_memory_layout": "day_major_target_peer"},
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            small_cfg(**{**PEER_ONLY, **kw})
    with pytest.raises(ValueError):  # 순차 구조에 Availability 플래그
        small_cfg(encoder_attention_scope="per_day_buildings", fusion_include_availability=True)


def test_target_excluded_and_memory_shape():
    model = po_model()
    b, t, z, u = 3, 5, 4, 6
    batch = random_batch(b=b, t=t, z=z, u=u)
    mem = _memories(model, batch)
    assert mem["zone"][0].shape == (b, t * (z - 1), 32) and mem["zone"][1].shape == (b, t * (z - 1))
    assert mem["type"][0].shape == (b, t * (u - 1), 32)
    # 대상 건물(열 0) 값을 바꿔도 Encoder Memory는 그대로다
    changed = {k: v.clone() for k, v in batch.items()}
    changed["zone_daily"][:, :, 0] += 5.0
    changed["type_daily"][:, :, 0] += 5.0
    for c in ("zone", "type"):
        assert torch.equal(_memories(model, changed)[c][0], mem[c][0])
    # Memory Mask = 입력 Mask의 Peer 부분
    assert torch.equal(mem["zone"][1], batch["zone_mask"].reshape(b, t, z)[:, :, 1:].reshape(b, -1))


def test_decoder_still_uses_target_and_is_causal():
    model = po_model()
    batch = random_batch(t=6)
    base = model(**batch)
    assert base.shape == (3, 6, 24)
    changed = dict(batch)
    changed["target_daily"] = batch["target_daily"].clone()
    changed["target_daily"][:, 4:] += 3.0
    out = model(**changed)
    assert not torch.allclose(out[:, 4:], base[:, 4:], atol=1e-5)
    # Encoder에는 대상이 없으므로 미래 날짜 대상 변화는 과거 날짜 출력에 영향이 없다 (Decoder Causal)
    assert torch.allclose(out[:, :4], base[:, :4], atol=1e-5)


def test_final_ln_and_same_day_pe_no_pooling():
    model = po_model(encoder_context_day_pe="none")
    enc = model.zone_encoder
    batch = random_batch(t=4, z=4)
    mem, _ = enc(batch["zone_daily"], batch["zone_mask"])
    # PE 없이는 Final LN 출력: Token마다 평균 0, 분산 1 근처(가중치 초기값 1/0)
    assert torch.allclose(mem.mean(-1), torch.zeros(mem.shape[:2]), atol=1e-5)
    with_pe = po_model()
    m2, _ = with_pe.zone_encoder(batch["zone_daily"], batch["zone_mask"])
    pe = with_pe.day_pe(4)
    diff = (m2 - mem).reshape(3, 4, 3, -1)  # 같은 가중치(seed 0) -> 차이 = 날짜 PE
    assert torch.allclose(diff, pe[None, :, None, :].expand_as(diff), atol=1e-5)
    # Pooling 없음: 같은 날짜의 Peer Token이 서로 다르다
    assert not torch.allclose(m2.reshape(3, 4, 3, -1)[0, :, 0], m2.reshape(3, 4, 3, -1)[0, :, 1])


def test_missing_and_padding_peers_excluded():
    model = po_model()
    b, t, z = 3, 5, 4
    batch = random_batch(b=b, t=t, z=z)
    zm = batch["zone_mask"].reshape(b, t, z).clone()
    zm[0, 2, 1] = False  # 샘플 0, 날짜 2, Peer 1 결측
    batch["zone_mask"] = zm.reshape(b, -1)
    base = model(**batch)
    changed = dict(batch)
    changed["zone_daily"] = batch["zone_daily"].clone()
    changed["zone_daily"][0, 2, 1] = 99.0   # 가려진 Peer 날짜
    changed["zone_daily"][0, :, 3] = 77.0   # 샘플 0의 Padding 슬롯
    assert torch.allclose(model(**changed), base, atol=1e-5)
    changed["zone_daily"][0, 3, 1] = 99.0   # 관측된 Peer 날짜는 영향이 있다
    assert not torch.allclose(model(**changed)[0], base[0], atol=1e-5)


@pytest.mark.parametrize("case", ["zone0", "type0", "both0", "one_peer", "some_days0"])
def test_edge_cases_finite_forward_backward(case):
    torch.manual_seed(0)
    model = DualContextTransformer(small_cfg(**PEER_ONLY)).train()
    b, t, z, u = 3, 5, 4, 6
    batch = random_batch(b=b, t=t, z=z, u=u)
    zm, tm = batch["zone_mask"].reshape(b, t, z).clone(), batch["type_mask"].reshape(b, t, u).clone()
    if case in ("zone0", "both0"):
        zm[:, :, 1:] = False
    if case in ("type0", "both0"):
        tm[:, :, 1:] = False
    if case == "one_peer":
        zm[:, :, 2:] = False
        tm[:, :, 2:] = False
    if case == "some_days0":
        zm[0, 1:3, 1:] = False
        tm[1, 0, 1:] = False
    batch["zone_mask"], batch["type_mask"] = zm.reshape(b, -1), tm.reshape(b, -1)
    out = model(**batch)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_group_with_no_peer_slots():
    model = po_model()
    batch = random_batch(z=1, u=1)  # 배치 전체가 대상 단독 그룹
    out = model(**batch)
    assert torch.isfinite(out).all()
    rec = _fusion_inputs(model, batch)
    f_in = rec[0][0]
    assert torch.equal(f_in[..., 32:], torch.zeros_like(f_in[..., 32:]))  # [Z; 0; 0; 0; 0]


def test_availability_flags_and_zeroed_branch():
    model = po_model()
    batch = random_batch()  # 샘플 -1은 Zone Peer가 없다
    rec = _fusion_inputs(model, batch)
    d = 32
    for f_in, f_out in rec:
        assert f_in.shape == (3, 5, 3 * d + 2) and f_out.shape == (3, 5, d)
        m_s, m_u = f_in[..., 3 * d], f_in[..., 3 * d + 1]
        assert m_s[:2].eq(1).all() and m_s[-1].eq(0).all() and m_u.eq(1).all()
        assert torch.equal(f_in[-1, :, d:2 * d], torch.zeros(5, d))      # O_S = 0
        assert not torch.equal(f_in[0, :, d:2 * d], torch.zeros(5, d))


def test_branches_share_query_independent_weights():
    model = po_model()
    rec = []
    layer = model.decoder.layers[0]
    layer.cross_norm.register_forward_hook(lambda m, i, o: rec.append(("z", o)))
    for name, attn in zip(layer.cross_order, layer.cross):
        attn.register_forward_hook(lambda m, i, o, n=name: rec.append((n, i[0])))
    model(**random_batch())
    got = dict(rec)
    assert torch.equal(got["zone"], got["z"]) and torch.equal(got["type"], got["z"])
    zone, typ = layer.cross
    assert zone.in_proj_weight is not typ.in_proj_weight


def test_peer_order_permutation_invariant():
    model = po_model()
    b, t, u = 3, 5, 6
    batch = random_batch(b=b, t=t, u=u)
    base = model(**batch)
    perm = torch.tensor([0, 3, 5, 1, 4, 2])  # 대상(열 0)은 두고 Peer 순서만 바꾼다
    changed = dict(batch)
    changed["type_daily"] = batch["type_daily"][:, :, perm]
    changed["type_mask"] = batch["type_mask"].reshape(b, t, u)[:, :, perm].reshape(b, -1)
    assert torch.allclose(model(**changed), base, atol=1e-5)


def test_existing_parallel_unchanged():
    """기존 v3_pm_parallel 설정은 Fusion 입력 3D 그대로 (Availability 없음)."""
    m = ModelConfig.from_dict(load_config("configs/v3_pm_parallel.yaml")["model"])
    model = DualContextTransformer(m)
    assert not model.decoder.layers[0].availability
    assert model.decoder.layers[0].context_fusion[0].in_features == 384
