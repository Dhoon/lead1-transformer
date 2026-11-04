"""저장된 Checkpoint로 검증 Fold를 다시 추론해 Threshold-F1 곡선을 만든다.

python -m src.eval.threshold_curve runs/fold0/last.pt runs/posw/fold0/last.pt --window 64

검증 정답으로 고른 최적 Threshold는 낙관적인 값이다 (같은 건물로 고르고 평가). 모델 간 비교용으로만 쓴다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.data.dataset import inference_batches
from src.models.dual_context import DualContextTransformer, ModelConfig
from src.train.common import get_device, max_group_size
from src.train.train_supervised import predict, prepare

THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def f1_at(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    prec = tp / max(int(pred.sum()), 1)
    rec = tp / max(int((y == 1).sum()), 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": t, "f1": f1, "precision": prec, "recall": rec}


def best_f1(y: np.ndarray, p: np.ndarray) -> dict:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    return {"threshold": float(thr[i]), "f1": float(f1[i]), "precision": float(prec[i]), "recall": float(rec[i])}


def evaluate(ckpt_path: Path, window: int, device, fold: int | None = None) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    # a92e117 이전 Checkpoint는 --fold로 바꾼 Fold가 config에 남지 않았으므로 --fold로 지정한다.
    fold = cfg["cv"]["fold"] if fold is None else fold
    model_cfg = ModelConfig.from_dict(ckpt["model_cfg"])
    grid, ds_train, ds_eval, _, eval_idx = prepare(cfg, fold, model_cfg)
    model = DualContextTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])

    tcfg = cfg["train"]
    batches = inference_batches(
        len(ds_eval), grid.n_days, window, cfg["eval"].get("stride_days") or window,
        tcfg["batch_size"], max_group_size(ds_eval.zone_idx, ds_eval.type_idx), tcfg.get("max_tokens_per_batch"),
    )
    prob = predict(model, ds_eval, batches, device)
    eval_grid = grid.subset(eval_idx)
    keep = eval_grid.present
    y, p = eval_grid.anomaly[keep].astype(int), prob[keep]
    missing = ~eval_grid.observed[keep]

    out = {
        "checkpoint": str(ckpt_path), "epoch": ckpt["epoch"], "window": window,
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "best_f1": best_f1(y, p),
        "curve": [f1_at(y, p, t) for t in THRESHOLDS],
        # 결측 meter_reading 시간(정답은 모두 0)에 준 확률: 오탐이 여기에 몰리는지 확인
        "missing_hours": {
            "n": int(missing.sum()), "pos_rate": float(y[missing].mean()) if missing.any() else 0.0,
            "mean_prob": float(p[missing].mean()) if missing.any() else 0.0,
            "observed_mean_prob": float(p[~missing].mean()),
        },
    }
    np.savez_compressed(ckpt_path.with_suffix(f".val_T{window}.npz"), prob=prob, eval_idx=eval_idx)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fold", type=int, default=None, help="기본 = Checkpoint config의 cv.fold")
    args = ap.parse_args(argv)
    device = get_device(args.device)
    for path in args.checkpoints:
        res = evaluate(path, args.window, device, args.fold)
        path.with_suffix(f".threshold_T{args.window}.json").write_text(json.dumps(res, indent=2))
        b, m = res["best_f1"], res["missing_hours"]
        print(f"\n{path} (epoch {res['epoch']}, T={args.window}) roc {res['roc_auc']:.4f} pr {res['pr_auc']:.4f}")
        print(f"  best F1 {b['f1']:.4f} @ threshold {b['threshold']:.3f} (P {b['precision']:.3f} R {b['recall']:.3f})")
        for c in res["curve"]:
            print(f"  t={c['threshold']:.2f}  F1 {c['f1']:.4f}  P {c['precision']:.3f}  R {c['recall']:.3f}")
        print(f"  missing hours n={m['n']} pos_rate={m['pos_rate']:.4f} mean_prob={m['mean_prob']:.4f} "
              f"(observed mean_prob={m['observed_mean_prob']:.4f})")


if __name__ == "__main__":
    main()
