"""전체 모델 End-to-End 시간별 이상 지도학습 + 추론 (SPEC 5, 6, 7).

python -m src.train.train_supervised --config configs/train.yaml --fold 0    # 미학습 건물 검증
python -m src.train.train_supervised --config configs/train.yaml --fold -1   # 200개 학습 -> test 206개 제출
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import RandomWindowBatchSampler, WindowDataset, collate_windows, inference_batches, to_token_grid
from src.data.day_patches import LogStandardizer, impute_missing
from src.data.lead_loader import concat_grids, load_split
from src.data.splits import select_fold
from src.eval.kaggle_submission import make_submission
from src.eval.metrics import hourly_metrics
from src.models.dual_context import DualContextTransformer, ModelConfig, model_inputs
from src.train.common import (
    build_groups, get_device, load_config, max_group_size, run_dir, set_seed, to_device, window_steps,
)

LOSS_POLICIES = ("include", "exclude_missing")
LR_SCHEDULES = ("constant", "cosine")
# 평가(검증/테스트) 대상 건물의 Peer를 어디서 가져오는가. 레이블은 어느 경우에도 쓰지 않는다.
PEER_POOLS = {
    "train_only": "학습 건물에서만 매칭 (학습 때와 같은 Peer 분포)",
    "within_split": "같은 분할의 평가 건물끼리",
    "with_train": "학습 건물 + 평가 건물",
}


def hourly_loss(logits, target, mask, pos_weight=None):
    """`[B, T, P]` 시간별 BCE (P = Token당 시간 수). mask가 False인 시점(정답 Row 없음 등)은 제외한다."""
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def lr_factor(schedule: str, progress: float, warmup: float) -> float:
    """학습 진행률(0~1)에서 기본 LR에 곱할 배율. cosine = 선형 Warmup 후 0까지 코사인 감소."""
    if schedule == "constant":
        return 1.0
    if progress < warmup:
        return progress / warmup
    return 0.5 * (1.0 + np.cos(np.pi * min((progress - warmup) / (1.0 - warmup), 1.0)))


@torch.no_grad()
def predict(model, dataset: WindowDataset, batches, device) -> np.ndarray:
    """모든 Window의 시간별 확률을 행마다 평균해 한 행당 점수 1개를 만든다.

    Token 단위 `[n_targets, n_steps, P]`로 모은 뒤 원래 격자 `[n_targets, n_days, 24]`로 되돌린다.
    """
    model.eval()
    pos = {int(t): i for i, t in enumerate(dataset.targets)}
    total = np.zeros((len(dataset.targets), dataset.n_steps, dataset.patch))
    count = np.zeros((len(dataset.targets), dataset.n_steps, 1))
    loader = DataLoader(dataset, batch_sampler=batches, collate_fn=collate_windows)
    for batch in loader:
        prob = torch.sigmoid(model(**model_inputs(to_device(batch, device)))).cpu().numpy()
        t = prob.shape[1]
        for k, (row, start) in enumerate(zip(batch["target_row"].tolist(), batch["start"].tolist())):
            total[pos[row], start : start + t] += prob[k]
            count[pos[row], start : start + t] += 1
    assert (count > 0).all(), "추론 Window가 덮지 못한 시점이 있다."
    return (total / count).astype(np.float32).reshape(len(dataset.targets), -1, 24)


def prepare(cfg: dict, fold: int, model_cfg: ModelConfig, norm: LogStandardizer | None = None):
    """격자 로딩 -> 결측 대체 -> 정규화 -> Token 격자 -> 그룹 구성. 통계는 Training Fold에서만 추정한다.

    반환하는 grid는 원래 `[n, n_days, 24]` 격자다. Dataset 안의 입력·레이블만 `[n, n_steps, P]` Token 격자로 바꾼다.

    norm을 주면 새로 추정하지 않고 그대로 쓴다 (Checkpoint 학습 때의 통계로 추론).
    """
    data, ev = cfg["data"], cfg["eval"]
    train_grid = load_split(data["raw_dir"], "train", data.get("cache_dir"))
    if fold < 0:
        test_grid = load_split(data["raw_dir"], "test", data.get("cache_dir"))
        grid = concat_grids(train_grid, test_grid)
        n_train = len(train_grid.building_ids)
        train_idx, eval_idx = np.arange(n_train), n_train + np.arange(len(test_grid.building_ids))
        labels = np.concatenate([train_grid.anomaly, np.zeros_like(test_grid.present, dtype=np.int8)])
    else:
        grid = train_grid
        train_idx, eval_idx = select_fold(grid.building_ids, fold, cfg["cv"]["n_splits"], cfg["seed"])
        labels = grid.anomaly

    observed = grid.observed
    filled = impute_missing(grid.values, data["missing_value_strategy"])  # NaN 없음
    if norm is None:
        norm = LogStandardizer().fit(grid.values[train_idx], observed[train_idx])
    x = norm.transform(filled)

    policy = data["missing_reading_loss_policy"]
    if policy not in LOSS_POLICIES:
        raise ValueError(f"missing_reading_loss_policy must be one of {LOSS_POLICIES}")
    label_mask = grid.present.copy()
    if policy == "exclude_missing":
        label_mask &= observed
    label_mask[eval_idx] = False  # 검증/테스트 정답은 학습에 쓰지 않는다.

    if ev["peer_pool"] not in PEER_POOLS:
        raise ValueError(f"peer_pool must be one of {list(PEER_POOLS)}")
    eval_pool = {
        "train_only": train_idx,
        "within_split": eval_idx,
        "with_train": np.concatenate([train_idx, eval_idx]),
    }[ev["peer_pool"]]
    tz, tt = build_groups(grid, train_idx, train_idx, model_cfg)
    ez, et = build_groups(grid, eval_idx, eval_pool, model_cfg)
    p = model_cfg.patch_size_hours
    x, observed, labels, label_mask = (to_token_grid(a, p) for a in (x, observed, labels, label_mask))
    ds_train = WindowDataset(x, train_idx, tz, tt, observed, labels, label_mask)
    ds_eval = WindowDataset(x, eval_idx, ez, et, observed)
    return grid, ds_train, ds_eval, norm, eval_idx


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--fold", type=int, default=None, help="-1 = 200개 전체 학습 후 test 예측")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    fold = cfg["cv"]["fold"] if args.fold is None else args.fold
    cfg["cv"]["fold"] = fold  # Checkpoint에 실제 학습 Fold를 남긴다 (threshold_curve가 이 값으로 검증 건물을 다시 만든다)
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    tcfg, ev = cfg["train"], cfg["eval"]
    if tcfg.get("normal_only", False):
        raise ValueError("SPEC 5.1: 이상 시간을 포함한 날짜도 학습에 쓴다 (normal_only=false).")
    model_cfg = ModelConfig.from_dict(cfg["model"])
    win = window_steps(cfg, model_cfg.patch_size_hours)  # Window 길이는 모두 Token 개수 (Day: 일, Hour: 시간)
    model_cfg = ModelConfig.from_dict({**cfg["model"], "max_window_days": max([win["max_t"], *win["eval"]])})

    grid, ds_train, ds_eval, norm, eval_idx = prepare(cfg, fold, model_cfg)
    train_group = max_group_size(ds_train.zone_idx, ds_train.type_idx)
    eval_group = max_group_size(ds_eval.zone_idx, ds_eval.type_idx)
    print(
        f"fold={fold} train_buildings={len(ds_train)} eval_buildings={len(ds_eval)} "
        f"max_group train={train_group} eval={eval_group} missing={cfg['data']['missing_value_strategy']}"
    )

    model = DualContextTransformer(model_cfg).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    pos_weight = tcfg.get("pos_weight")
    pos_weight = None if pos_weight is None else torch.tensor(float(pos_weight), device=device)
    schedule, warmup = tcfg.get("lr_schedule", "constant"), tcfg.get("warmup_ratio", 0.0)
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}")
    # Batch 크기가 T마다 달라 Step 수가 일정하지 않으므로 본 샘플 수로 진행률을 잰다.
    total_samples, seen = tcfg["epochs"] * tcfg["samples_per_epoch"], 0

    windows = win["eval"]
    out = run_dir(cfg, fold)
    primary = ev.get("metric_primary", "roc_auc")
    history, best = [], None

    def save(name: str, epoch: int, metrics: dict | None):
        torch.save(
            {
                "model_cfg": model_cfg.to_dict(), "model": model.state_dict(), "normalizer": norm.state_dict(),
                "config": cfg, "epoch": epoch, "metrics": metrics,
            },
            out / name,
        )

    for epoch in range(tcfg["epochs"]):
        model.train()
        sampler = RandomWindowBatchSampler(
            len(ds_train), ds_train.n_steps, win["min_t"], win["max_t"],
            tcfg["samples_per_epoch"], tcfg["batch_size"], train_group, tcfg.get("max_tokens_per_batch"),
            seed=cfg["seed"] + epoch,
        )
        loader = DataLoader(ds_train, batch_sampler=sampler, collate_fn=collate_windows, num_workers=cfg.get("num_workers", 0))
        for step, batch in enumerate(loader):
            batch = to_device(batch, device)
            lr = tcfg["lr"] * lr_factor(schedule, seen / total_samples, warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            logits = model(**model_inputs(batch))
            seen += logits.shape[0]
            loss = hourly_loss(logits, batch["anomaly_target"], batch["label_mask"], pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 1.0))
            opt.step()
            if step % tcfg.get("log_every", 50) == 0:
                print(f"epoch {epoch} step {step} T={logits.shape[1]} B={logits.shape[0]} lr {lr:.2e} loss {loss.item():.4f}")

        if fold >= 0:
            eval_grid = grid.subset(eval_idx)
            keep = eval_grid.present
            row = {"epoch": epoch}
            for w in windows:
                batches = inference_batches(
                    len(ds_eval), ds_eval.n_steps, w, win["stride"] or w,
                    tcfg["batch_size"], eval_group, tcfg.get("max_tokens_per_batch"),
                )
                prob = predict(model, ds_eval, batches, device)
                row[f"T={w}"] = hourly_metrics(eval_grid.anomaly[keep], prob[keep], ev.get("threshold", 0.5))
            history.append(row)
            print(f"epoch {epoch} eval {json.dumps(row)}")
            (out / "metrics.json").write_text(json.dumps(history, indent=2))
            # best: 첫 번째 평가 길이(eval window[0])의 metric_primary 기준
            score = row[f"T={windows[0]}"].get(primary)
            if score is not None and (best is None or score > best):
                best = score
                save("best.pt", epoch, row)
                print(f"epoch {epoch} new best {primary}={score:.4f} (T={windows[0]}) -> best.pt")
        save("last.pt", epoch, history[-1] if history else None)
    if fold < 0:
        w = windows[0]
        batches = inference_batches(
            len(ds_eval), ds_eval.n_steps, w, win["stride"] or w,
            tcfg["batch_size"], eval_group, tcfg.get("max_tokens_per_batch"),
        )
        prob = predict(model, ds_eval, batches, device)
        eval_grid = grid.subset(eval_idx)
        sample = pd.read_csv(f"{cfg['data']['raw_dir']}/sample_submission.csv")
        sub = make_submission(prob, eval_grid.row_id, eval_grid.present, sample)
        sub.to_csv(out / "submission.csv", index=False)
        print(f"saved {out / 'submission.csv'} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
