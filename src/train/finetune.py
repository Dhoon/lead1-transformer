"""사전학습 표현 -> 시간별 이상 탐지 전이 (SPEC 6, 7.2).

python -m src.train.finetune --config configs/downstream.yaml --fold 0     # 미학습 건물 검증
python -m src.train.finetune --config configs/downstream.yaml --fold -1    # 200개 학습 -> test 206개 제출
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from src.data.dataset import WindowDataset
from src.data.day_patches import LogStandardizer
from src.data.lead_loader import concat_grids, load_split
from src.data.splits import select_fold
from src.eval.kaggle_submission import make_submission
from src.eval.metrics import hourly_metrics
from src.models.dual_context import DualContextTransformer, ModelConfig, backbone_inputs
from src.models.heads import HourlyAnomalyHead
from src.train.common import build_peers, get_device, load_config, run_dir, set_seed, to_device

PEER_POOLS = ("inductive", "transductive")


def hourly_loss(logits, labels, mask, loss_on: str, pos_weight):
    """logits/labels/mask: `[B, T, 24]`. loss_on: all_days | last_day."""
    if loss_on == "last_day":
        logits, labels, mask = logits[:, -1], labels[:, -1], mask[:, -1]
    elif loss_on != "all_days":
        raise ValueError(loss_on)
    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def predict_grid(model, head, dataset: WindowDataset, n_days: int, batch_size: int, device) -> np.ndarray:
    """각 날짜를 그 날짜로 끝나는 Window의 마지막 토큰으로 판정한다 (그날까지 관측된 데이터만 사용)."""
    model.eval(), head.eval()
    rows = {int(t): i for i, t in enumerate(dataset.targets)}
    prob = np.zeros((len(dataset.targets), n_days, 24), dtype=np.float32)
    for batch in DataLoader(dataset, batch_size=batch_size):
        logits = head(model(**backbone_inputs(to_device(batch, device))))[:, -1]
        pos = [rows[int(t)] for t in batch["target_row"]]
        prob[pos, batch["end_day"].numpy()] = torch.sigmoid(logits).cpu().numpy()
    return prob


def load_backbone(dcfg: dict, cfg: dict, fold: int, device):
    """init_from이 있으면 사전학습 가중치·정규화 통계를 가져온다. 다른 Fold의 체크포인트는 누수이므로 거부한다."""
    path = dcfg.get("init_from")
    if not path:
        return DualContextTransformer(ModelConfig.from_dict(cfg["model"])).to(device), None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt["fold"] != fold or ckpt["cv"]["n_splits"] != cfg["cv"]["n_splits"] or ckpt["seed"] != cfg["seed"]:
        raise ValueError(f"체크포인트 fold/cv/seed가 다르다: {path} (fold {ckpt['fold']}, 요청 {fold})")
    model = DualContextTransformer(ModelConfig.from_dict(ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["backbone"])
    return model, LogStandardizer(**ckpt["normalizer"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/downstream.yaml")
    ap.add_argument("--fold", type=int, default=None, help="-1 = 200개 전체 학습 후 test 예측")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    fold = cfg["cv"]["fold"] if args.fold is None else args.fold
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    dcfg = cfg["downstream"]
    if dcfg["mode"] != "supervised_hourly_head":
        raise NotImplementedError("SPEC 6.3 정상 전용 Objective는 아직 [미정]이다.")
    if dcfg["peer_pool"] not in PEER_POOLS:
        raise ValueError(f"peer_pool must be one of {PEER_POOLS}")

    data = cfg["data"]
    train_grid = load_split(data["raw_dir"], "train", data.get("cache_dir"))
    if fold < 0:
        test_grid = load_split(data["raw_dir"], "test", data.get("cache_dir"))
        grid = concat_grids(train_grid, test_grid)
        train_idx = np.arange(len(train_grid.building_ids))
        eval_idx = len(train_grid.building_ids) + np.arange(len(test_grid.building_ids))
        labels = np.concatenate([train_grid.anomaly, np.zeros_like(test_grid.present, dtype=np.int8)])
    else:
        grid = train_grid
        train_idx, eval_idx = select_fold(len(grid.building_ids), grid.building_ids, fold, cfg["cv"]["n_splits"], cfg["seed"])
        labels = grid.anomaly
    label_mask = grid.present.copy()
    label_mask[eval_idx] = False  # 검증/테스트 정답은 학습에 쓰지 않는다.

    model, norm = load_backbone(dcfg, cfg, fold, device)
    model_cfg = model.cfg
    if norm is None:
        norm = LogStandardizer().fit(grid.values[train_idx], grid.valid[train_idx])
    x = norm.transform(grid.values, grid.valid)
    window = model_cfg.decoder_window_days
    all_days = range(grid.n_days)

    zone, typ = build_peers(grid, train_idx, train_idx, model_cfg)
    ds_train = WindowDataset(
        x, grid.valid, train_idx, window, all_days, zone, typ, labels=labels, label_mask=label_mask
    )
    pool = train_idx if dcfg["peer_pool"] == "inductive" else np.concatenate([train_idx, eval_idx])
    ezone, etype = build_peers(grid, eval_idx, pool, model_cfg)
    ds_eval = WindowDataset(x, grid.valid, eval_idx, window, all_days, ezone, etype)

    head = HourlyAnomalyHead(model_cfg.d_model).to(device)
    if dcfg.get("freeze_backbone", False):
        model.requires_grad_(False)
    params = [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=dcfg["lr"], weight_decay=dcfg["weight_decay"])

    pw = dcfg.get("pos_weight", "auto")
    if pw == "auto":
        y = labels[train_idx][label_mask[train_idx]]
        pw = float((y == 0).sum() / max((y == 1).sum(), 1))
    pos_weight = None if pw is None else torch.tensor(float(pw), device=device)
    print(f"fold={fold} train={len(ds_train)} eval={len(ds_eval)} pos_weight={pw} peer_pool={dcfg['peer_pool']}")

    n = dcfg.get("samples_per_epoch") or len(ds_train)
    loader = DataLoader(
        ds_train, batch_size=dcfg["batch_size"], sampler=RandomSampler(ds_train, num_samples=n),
        num_workers=cfg.get("num_workers", 0),
    )
    for epoch in range(dcfg["epochs"]):
        model.train(), head.train()
        for step, batch in enumerate(loader):
            batch = to_device(batch, device)
            logits = head(model(**backbone_inputs(batch)))
            loss = hourly_loss(logits, batch["hour_labels"], batch["label_mask"], dcfg["loss_on"], pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, dcfg.get("grad_clip", 1.0))
            opt.step()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    out = run_dir(cfg, "downstream", fold)
    prob = predict_grid(model, head, ds_eval, grid.n_days, dcfg["batch_size"], device)
    eval_grid = grid.subset(eval_idx)
    torch.save({"backbone": model.state_dict(), "head": head.state_dict(), "model_cfg": model_cfg.to_dict()}, out / "model.pt")
    if fold < 0:
        sample = pd.read_csv(f"{data['raw_dir']}/sample_submission.csv")
        sub = make_submission(prob, eval_grid.row_id, eval_grid.present, sample)
        sub.to_csv(out / "submission.csv", index=False)
        print(f"saved {out / 'submission.csv'} ({len(sub)} rows)")
    else:
        keep = eval_grid.present
        metrics = hourly_metrics(eval_grid.anomaly[keep], prob[keep], dcfg.get("threshold", 0.5))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"eval {json.dumps(metrics)}")


if __name__ == "__main__":
    main()
