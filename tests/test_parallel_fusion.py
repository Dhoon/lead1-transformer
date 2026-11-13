"""cross_attention_fusion: parallel_concat_mlp (docs/LEAD_v3pm_Parallel_Site_Type_CrossAttention_Change.md 6.1)."""
import pytest
import torch

from src.models.dual_context import DualContextTransformer, ModelConfig
from src.models.temporal_decoder import ParallelFusionDecoderLayer, TemporalDecoderLayer
from src.train.common import load_config
from tests.conftest import random_batch
from tests.test_peer_missing_mask import KEYS, _batch, _dataset
from tests.test_per_day_encoder import per_day_model


def par_model(**kw):
    return per_day_model(cross_attention_fusion="parallel_concat_mlp", **kw)


def _capture(model):
    """레이어별 Branch 입력/출력과 Fusion 입력/출력을 모은다."""
    rec = []
    for i, layer in enumerate(model.decoder.layers):
        layer.cross_norm.register_forward_hook(lambda m, inp, out, i=i: rec.append((i, "z", out)))
        for name, attn in zip(layer.cross_order, layer.cross):
            attn.register_forward_hook(
                lambda m, inp, out, i=i, n=name: rec.append((i, n, {"q": inp[0], "kv": inp[1], "out": out[0]}))
            )
        layer.context_fusion.register_forward_hook(lambda m, inp, out, i=i: rec.append((i, "fusion", (inp[0], out))))
    return rec


def _get(rec, i, name):
    return [r[2] for r in rec if r[0] == i and r[1] == name][-1]


def test_v3_pm_parallel_config():
    m = ModelConfig.from_dict(load_config("configs/v3_pm_parallel.yaml")["model"])
    assert m.cross_attention_fusion == "parallel_concat_mlp" and m.encoder_context_pooling == "target_token" and m.per_day
    model = DualContextTransformer(m)
    assert all(isinstance(layer, ParallelFusionDecoderLayer) for layer in model.decoder.layers)
    fusion = model.decoder.layers[0].context_fusion
    assert (fusion[0].in_features, fusion[0].out_features, fusion[2].out_features) == (384, 128, 128)
    # 기존 v3_pm은 그대로 순차 구조
    seq = DualContextTransformer(ModelConfig.from_dict(load_config("configs/v3_pm.yaml")["model"]))
    assert all(isinstance(layer, TemporalDecoderLayer) for layer in seq.decoder.layers)
    assert seq.decoder.layers[0].block_names == ["self", "ffn", "zone", "ffn", "type", "ffn"]


def test_invalid_fusion_settings_rejected():
    for kw in ({"cross_attention_fusion": "parallel"}, {"cross_attention_fusion": "parallel_concat_mlp", "layer_norm": "post"},
               {"cross_attention_fusion": "parallel_concat_mlp", "ffn_after_each_attention": False}):
        with pytest.raises(ValueError):
            ModelConfig.from_dict({"encoder_attention_scope": "per_day_buildings", **kw})


def test_branches_share_query_and_shapes():
    model = par_model()
    rec = _capture(model)
    b, t = 3, 7
    out = model(**random_batch(b=b, t=t))
    assert out.shape == (b, t, 24)
    d = model.cfg.d_model
    for i in range(len(model.decoder.layers)):
        z, zone, typ = _get(rec, i, "z"), _get(rec, i, "zone"), _get(rec, i, "type")
        assert torch.equal(zone["q"], z) and torch.equal(typ["q"], z)  # 두 Branch Query = 같은 LN(H_self)
        assert zone["kv"].shape == typ["kv"].shape == (b, t, d)  # Memory는 날짜별 대상 Context T개 (T×G 아님)
        f_in, f_out = _get(rec, i, "fusion")
        assert f_in.shape == (b, t, 3 * d) and f_out.shape == (b, t, d)
        assert torch.equal(f_in, torch.cat([z, zone["out"], typ["out"]], -1))  # [Z; O_S; O_U]


def test_branch_parameters_are_independent():
    layer = par_model().decoder.layers[0]
    zone, typ = layer.cross
    assert zone.in_proj_weight is not typ.in_proj_weight and zone.out_proj.weight is not typ.out_proj.weight
    assert not torch.equal(zone.in_proj_weight, typ.in_proj_weight)


def test_each_memory_changes_only_its_branch():
    model = par_model()
    batch = random_batch(b=2, t=5)
    rec = _capture(model)
    model(**batch)
    base_zone, base_type = _get(rec, 0, "zone")["out"], _get(rec, 0, "type")["out"]
    changed = dict(batch)
    changed["zone_daily"] = batch["zone_daily"].clone()
    changed["zone_daily"][:, :, 1:] += 2.0  # Zone Peer만 바꾼다
    rec.clear()
    model(**changed)
    assert not torch.allclose(_get(rec, 0, "zone")["out"], base_zone, atol=1e-5)
    assert torch.allclose(_get(rec, 0, "type")["out"], base_type, atol=1e-6)  # 1층 Type Branch는 그대로


def test_causal_self_attention_only():
    torch.manual_seed(0)
    layer = ParallelFusionDecoderLayer(16, 4, 32, 0.0, ("zone", "type")).eval()
    t = 6
    h = torch.randn(2, t, 16)
    mem = {c: (torch.randn(2, t, 16), torch.ones(2, t, dtype=torch.bool)) for c in ("zone", "type")}
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool), 1)
    base = layer(h, mem, mask)
    h2 = h.clone()
    h2[:, 4:] += 3.0  # 미래 날짜 Decoder 입력 변경 -> 과거 날짜 출력 불변 (Self-Attn Causal)
    assert torch.allclose(layer(h2, mem, mask)[:, :4], base[:, :4], atol=1e-5)
    mem2 = dict(mem)
    mem2["zone"] = (mem["zone"][0].clone(), mem["zone"][1])
    mem2["zone"][0][:, 5] += 3.0  # 미래 날짜 Memory 변경 -> 과거 날짜 출력 변화 (Cross-Attn에는 Causal 없음)
    assert not torch.allclose(layer(h, mem2, mask)[:, 0], base[:, 0], atol=1e-5)


def test_missing_peer_mask_still_applies():
    model = par_model()
    ds = _dataset(True)
    base = model(**{k: _batch(ds)[k] for k in KEYS})
    ds.x[2, 3:6] = 99.0  # 가려진 날짜의 Peer 값은 결과에 영향이 없다
    assert torch.allclose(model(**{k: _batch(ds)[k] for k in KEYS}), base, atol=1e-5)
