"""학습된 Checkpoint 여러 개로 test 206개 건물을 예측해 확률을 평균한 제출 파일을 만든다 (Fold 앙상블).

python -m src.eval.predict_test runs/v2_222/fold{0,1,2,3,4}/best.pt --out runs/v2_222/ensemble5/submission.csv

각 Checkpoint는 자기 학습 때의 정규화 통계로 입력을 만든다. test 건물의 Peer는 200개 학습 건물에서 매칭한다
(peer_pool: train_only, 레이블은 쓰지 않는다).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import inference_batches
from src.data.day_patches import LogStandardizer
from src.eval.kaggle_submission import make_submission
from src.models.dual_context import DualContextTransformer, ModelConfig
from src.train.common import get_device, max_group_size, window_steps
from src.train.train_supervised import predict, prepare


def load_normalizer(state: dict) -> LogStandardizer:
    norm = LogStandardizer()
    norm.mean, norm.std = state["mean"], state["std"]
    return norm


def predict_checkpoint(ckpt_path: Path, device, window: int | None = None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_cfg = ModelConfig.from_dict(ckpt["model_cfg"])
    grid, _, ds_eval, _, eval_idx = prepare(cfg, -1, model_cfg, load_normalizer(ckpt["normalizer"]))
    model = DualContextTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])

    tcfg, win = cfg["train"], window_steps(cfg, model_cfg.patch_size_hours)
    window = win["eval"][0] if window is None else window
    batches = inference_batches(
        len(ds_eval), ds_eval.n_steps, window, win["stride"] or window,
        tcfg["batch_size"], max_group_size(ds_eval.zone_idx, ds_eval.type_idx), tcfg.get("max_tokens_per_batch"),
    )
    return predict(model, ds_eval, batches, device), grid.subset(eval_idx), cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=None, help="Token 개수. 기본 = Checkpoint config의 첫 평가 Window")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    device = get_device(args.device)

    probs = []
    for path in args.checkpoints:
        prob, test_grid, cfg = predict_checkpoint(path, device, args.window)
        keep = test_grid.present
        print(f"{path}: mean_prob {prob[keep].mean():.4f}  >=0.5 {(prob[keep] >= 0.5).mean():.4f}")
        probs.append(prob)
    prob = np.mean(probs, axis=0)
    if len(probs) > 1:
        flat = np.stack([p[keep] for p in probs])
        print(f"ensemble: mean_prob {prob[keep].mean():.4f}  >=0.5 {(prob[keep] >= 0.5).mean():.4f}  "
              f"model 간 상관 min {np.corrcoef(flat).min():.3f}")

    sample = pd.read_csv(f"{cfg['data']['raw_dir']}/sample_submission.csv")
    sub = make_submission(prob, test_grid.row_id, test_grid.present, sample)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"saved {args.out} ({len(sub)} rows, {len(probs)} checkpoints)")


if __name__ == "__main__":
    main()
