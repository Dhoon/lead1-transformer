"""Hour Token v5 (docs/LEAD_HourLevel_Transformer_v5_Change.md 8장 Acceptance Criteria)."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import WindowDataset, collate_windows, inference_batches, inference_windows, to_token_grid
from src.data.day_patches import LogStandardizer, impute_missing
from src.data.lead_loader import grid_from_frame
from src.data.peer_groups import group_members
from src.eval.kaggle_submission import make_submission
from src.models.dual_context import DualContextTransformer, ModelConfig
from src.train.common import load_config, window_steps
from src.train.train_supervised import hourly_loss, predict
from tests.conftest import make_frame
from tests.test_per_day_encoder import _permute_each_day


def hour_cfg(**kw):
    base = dict(
        token_unit="hour", d_model=64, n_heads=4, head_dim=16, d_ff=256, dropout=0.0,
        encoder_attention_scope="per_hour_buildings", decoder_positional_encoding="sinusoidal_hour",
        target_role_embedding=True,
    )
    base.update(kw)
    return ModelConfig(**base)


def hour_model(**kw):
    torch.manual_seed(0)
    model = DualContextTransformer(hour_cfg(**kw)).eval()
    with torch.no_grad():  # Role은 0으로 초기화되므로 학습된 모델처럼 값을 넣는다
        for name, p in model.named_parameters():
            if ".role." in name:
                p.normal_()
    return model


def hour_batch(b=3, t=12, z=4, u=6, seed=0):
    """시간마다 그룹의 첫 건물 = 대상 건물. 일부 샘플은 그룹이 작아 Padding되고, 마지막 샘플은 Zone Peer가 없다."""
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(b, t, 1, generator=g)
    zone, typ = torch.randn(b, t, z, 1, generator=g), torch.randn(b, t, u, 1, generator=g)
    zone[:, :, 0], typ[:, :, 0] = target, target
    zm, tm = torch.ones(b, z, dtype=torch.bool), torch.ones(b, u, dtype=torch.bool)
    zm[0, -1], zm[-1, 1:], tm[-1, -2:] = False, False, False
    return {
        "target_daily": target,
        "zone_daily": zone, "zone_mask": zm[:, None].expand(-1, t, -1).reshape(b, t * z),
        "type_daily": typ, "type_mask": tm[:, None].expand(-1, t, -1).reshape(b, t * u),
    }


def test_v5_config():
    cfg = load_config("configs/v5_hour.yaml")
    m = ModelConfig.from_dict(cfg["model"])
    assert m.token_unit == "hour" and m.patch_size_hours == 1 and m.per_day
    assert (m.d_model, m.n_heads, m.d_model // m.n_heads, m.d_ff) == (64, 4, 16, 256)
    assert m.encoder_context_day_pe == "sinusoidal"  # encoder_context_hour_pe
    assert window_steps(cfg, 1) == {"min_t": 1, "max_t": 168, "eval": [168], "stride": None}
    model = DualContextTransformer(m)
    assert (model.patch.proj.in_features, model.patch.proj.out_features) == (1, 64)
    assert (model.head.proj.in_features, model.head.proj.out_features) == (64, 1)  # Linear(64,24) 없음
    assert not any(isinstance(x, torch.nn.Linear) and 24 in (x.in_features, x.out_features) for x in model.modules())
    # 이전 Config는 그대로 Day Token
    assert ModelConfig.from_dict(load_config("configs/v3_perday.yaml")["model"]).patch_size_hours == 24


def test_inconsistent_hour_keys_are_rejected():
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"token_unit": "hour", "patch_size_hours": 24, "encoder_attention_scope": "per_hour_buildings",
                               "decoder_positional_encoding": "sinusoidal_hour"})
    with pytest.raises(ValueError):  # hour Token에 날짜 PE 이름
        hour_cfg(decoder_positional_encoding="sinusoidal_day")
    with pytest.raises(ValueError):  # hour Token인데 날짜별 Scope
        hour_cfg(encoder_attention_scope="per_day_buildings")
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"encoder_attention_scope": "per_hour_buildings", "token_unit": "hour",
                               "decoder_positional_encoding": "sinusoidal_hour", "encoder_internal_hour_pe": True})
    with pytest.raises(ValueError):  # day Token에 hour 키
        ModelConfig.from_dict({"encoder_context_hour_pe": "sinusoidal"})


def test_shapes_encoder_per_hour_and_cross_scores_t_by_t():
    model = hour_model()
    batch = hour_batch(b=2, t=168, z=4, u=6)
    enc = model.zone_encoder
    assert enc.embed(batch["zone_daily"]).shape == (2 * 168, 4, 64)  # [B×T, Z, 64]
    ctx, mask = enc(batch["zone_daily"], batch["zone_mask"])
    assert ctx.shape == (2, 168, 64) and mask.all()  # 시간별 대상 Context
    keys = []
    for layer in model.decoder.layers:
        for name, block in zip(layer.block_names, layer.blocks):
            if name in ("zone", "type"):
                block.register_forward_hook(lambda m, inp, out: keys.append((inp[0].shape, inp[1].shape)))
    out = model(**batch)
    assert out.shape == (2, 168, 1) and torch.isfinite(out).all()  # 시간마다 Logit 1개
    assert keys and all(q == (2, 168, 64) and k == (2, 168, 64) for q, k in keys)  # Score [B, 4, T, T]


def test_hour_h_context_depends_only_on_hour_h():
    enc = hour_model().zone_encoder
    batch = hour_batch(b=2, t=10)
    base, _ = enc(batch["zone_daily"], batch["zone_mask"])
    changed = batch["zone_daily"].clone()
    changed[:, 4, 1:] += 3.0  # 4번째 시간의 Peer만 바꾼다
    out, _ = enc(changed, batch["zone_mask"])
    others = [i for i in range(10) if i != 4]
    assert torch.allclose(out[:, others], base[:, others], atol=1e-5)  # 시간축 전체 Flatten Attention 없음
    assert not torch.allclose(out[:, 4], base[:, 4], atol=1e-4)


def test_hour_pe_added_after_encoder():
    model = hour_model()
    enc = model.zone_encoder
    daily, mask = torch.ones(1, 3, 4, 1), torch.ones(1, 12, dtype=torch.bool)
    c, _ = enc(daily, mask)
    assert torch.allclose(c[0, 1] - c[0, 0], model.day_pe(3)[1] - model.day_pe(3)[0], atol=1e-5)


def test_peer_order_padding_and_no_peer():
    model = hour_model()
    batch = hour_batch(t=8)
    base = model(**batch)
    perm = dict(batch)
    perm["zone_daily"], perm["zone_mask"] = _permute_each_day(batch, "zone", 4, 8, seed=1)
    perm["type_daily"], perm["type_mask"] = _permute_each_day(batch, "type", 6, 8, seed=2)
    assert torch.allclose(model(**perm), base, atol=1e-5)
    noisy = dict(batch)
    noisy["zone_daily"] = batch["zone_daily"].clone()
    noisy["zone_daily"][~batch["zone_mask"].reshape(3, 8, 4)] = 99.0  # Padding 자리 값은 무시
    assert torch.allclose(model(**noisy), base, atol=1e-5)
    zero = dict(batch)
    zero["zone_daily"] = batch["zone_daily"].clone()
    zero["zone_daily"][0, :, 1] = 0.0  # Zero-fill 값은 유효 Token으로 남아 결과에 영향을 준다
    assert not torch.allclose(model(**zero)[0], base[0], atol=1e-5)


@pytest.mark.parametrize("t", [1, 24, 168])
def test_variable_t(t):
    assert hour_model()(**hour_batch(t=t)).shape == (3, t, 1)


def _hour_dataset(labels=True):
    df, meta = make_frame(n_buildings=6, n_days=10, labels=labels)
    if not labels:
        df.insert(0, "row_id", np.arange(len(df)) + 1000)
    grid = grid_from_frame(df, meta)
    x = LogStandardizer().fit(grid.values, grid.observed).transform(impute_missing(grid.values, "zero_fill"))
    targets = np.arange(len(grid.building_ids))
    zone = group_members(grid.site_id, targets, targets)
    typ = group_members(grid.primary_use, targets, targets)
    lab = None if grid.anomaly is None else to_token_grid(grid.anomaly, 1)
    mask = None if grid.anomaly is None else to_token_grid(grid.present, 1)
    ds = WindowDataset(to_token_grid(x, 1), targets, zone, typ, to_token_grid(grid.observed, 1), lab, mask)
    return df, grid, x, ds


def test_token_grid_keeps_real_timestamps():
    df, grid, x, ds = _hour_dataset()
    assert ds.n_steps == 10 * 24 and ds.patch == 1
    batch = collate_windows([ds[(0, 30, 50)], ds[(3, 30, 50)]])  # 30~79시간: 날짜 경계를 넘는 연속 구간
    assert batch["target_daily"].shape == (2, 50, 1)
    assert torch.equal(batch["zone_daily"][:, :, 0], batch["target_daily"])
    assert torch.equal(batch["type_daily"][:, :, 0], batch["target_daily"])
    hours = np.arange(30, 80)
    np.testing.assert_array_equal(batch["target_daily"][0, :, 0].numpy(), x[0, hours // 24, hours % 24])
    peer = int(ds.zone_idx[0][1])
    np.testing.assert_array_equal(batch["zone_daily"][0, :, 1, 0].numpy(), x[peer, hours // 24, hours % 24])
    np.testing.assert_array_equal(batch["anomaly_target"][0, :, 0].numpy(), grid.anomaly[0, hours // 24, hours % 24])


def test_training_step_on_hour_tokens():
    _, _, _, ds = _hour_dataset()
    model = DualContextTransformer(hour_cfg())
    batch = collate_windows([ds[(i, 5, 30)] for i in range(4)])
    logits = model(**{k: batch[k] for k in ("target_daily", "zone_daily", "zone_mask", "type_daily", "type_mask")})
    assert logits.shape == batch["anomaly_target"].shape == (4, 30, 1)
    loss = hourly_loss(logits, batch["anomaly_target"], batch["label_mask"])
    loss.backward()
    assert torch.isfinite(loss) and model.head.proj.weight.grad is not None


def test_inference_168h_windows_to_submission():
    """240시간을 168시간 Window로 덮으면 마지막은 72시간. 모든 row_id가 점수 1개를 받는다."""
    assert inference_windows(240, 168, 168) == [(0, 168), (168, 72)]
    df, grid, _, ds = _hour_dataset(labels=False)
    model = DualContextTransformer(hour_cfg())
    batches = inference_batches(len(ds), ds.n_steps, 168, 168, 8, 6, 65536)
    prob = predict(model, ds, batches, torch.device("cpu"))
    assert prob.shape == grid.values.shape == (6, 10, 24) and np.isfinite(prob).all()
    sample = pd.DataFrame({"row_id": df["row_id"].to_numpy()[::-1], "anomaly": 0})
    sub = make_submission(prob, grid.row_id, grid.present, sample)
    assert len(sub) == len(df) and not sub["row_id"].duplicated().any() and sub["anomaly"].notna().all()
