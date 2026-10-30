"""정상 일별 소비 패턴 -> Building ID 분류 사전학습 (SPEC 5).

python -m src.train.pretrain --config configs/pretrain.yaml --fold 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from src.data.dataset import IGNORE_INDEX, WindowDataset
from src.data.day_patches import LogStandardizer
from src.data.lead_loader import load_split
from src.data.splits import day_ranges, select_fold
from src.models.dual_context import DualContextTransformer, ModelConfig, backbone_inputs
from src.models.heads import BuildingIdHead
from src.train.common import build_peers, get_device, load_config, run_dir, set_seed, to_device


def id_loss(logits: torch.Tensor, target: torch.Tensor, loss_on: str) -> torch.Tensor:
    """logits: `[B, T, N]`, target: `[B, T]` (-100 = 무시). loss_on: all_valid_days | last_day."""
    if loss_on == "last_day":
        target = target.clone()
        target[:, :-1] = IGNORE_INDEX
    elif loss_on != "all_valid_days":
        raise ValueError(loss_on)
    return F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=IGNORE_INDEX)


def build_pretrain_data(grid, train_idx, model_cfg: ModelConfig, pcfg: dict):
    """정상 시간만 입력/손실에 사용하고, 정규화 통계는 학습 건물에서만 계산한다."""
    valid = grid.valid
    if pcfg.get("normal_only", True):
        valid = valid & (grid.anomaly == 0)
    norm = LogStandardizer().fit(grid.values[train_idx], valid[train_idx])
    x = norm.transform(grid.values, valid)

    day_ok = valid.sum(-1) >= pcfg.get("min_valid_hours", 20)
    if pcfg.get("normal_only", True):
        day_ok &= ~(grid.anomaly == 1).any(-1)
    train_days, val_days = day_ranges(grid.n_days, pcfg.get("val_days", 61))
    in_val = np.zeros(grid.n_days, dtype=bool)
    in_val[list(val_days)] = True

    zone, typ = build_peers(grid, train_idx, train_idx, model_cfg)
    common = dict(
        x=x, valid=valid, targets=train_idx, window=model_cfg.decoder_window_days,
        zone_idx=zone, type_idx=typ, class_index=np.arange(len(train_idx)),
    )
    shuffle_pool = train_idx if pcfg.get("context_shuffle", False) else None
    ds_train = WindowDataset(**common, end_days=train_days, day_ok=day_ok & ~in_val, shuffle_pool=shuffle_pool)
    stride = model_cfg.decoder_window_days if pcfg.get("loss_on") == "all_valid_days" else 1
    ds_val = WindowDataset(**common, end_days=list(val_days)[::stride], day_ok=day_ok & in_val, shuffle_pool=shuffle_pool)
    return ds_train, ds_val, norm


@torch.no_grad()
def evaluate(model, head, loader, device, loss_on):
    model.eval(), head.eval()
    total, correct, loss_sum, n_batches = 0, 0, 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        logits = head(model(**backbone_inputs(batch)))
        target = batch["building_target"]
        loss_sum += id_loss(logits, target, loss_on).item()
        n_batches += 1
        if loss_on == "last_day":
            logits, target = logits[:, -1:], target[:, -1:]
        keep = target != IGNORE_INDEX
        correct += (logits.argmax(-1)[keep] == target[keep]).sum().item()
        total += keep.sum().item()
    return {"loss": loss_sum / max(n_batches, 1), "acc": correct / max(total, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("--fold", type=int, default=None, help="-1 = 전체 200개 건물")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    fold = cfg["cv"]["fold"] if args.fold is None else args.fold
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    model_cfg = ModelConfig.from_dict(cfg["model"])
    pcfg = cfg["pretrain"]

    grid = load_split(cfg["data"]["raw_dir"], "train", cfg["data"].get("cache_dir"))
    train_idx, _ = select_fold(len(grid.building_ids), grid.building_ids, fold, cfg["cv"]["n_splits"], cfg["seed"])
    ds_train, ds_val, norm = build_pretrain_data(grid, train_idx, model_cfg, pcfg)
    num_classes = len(train_idx)  # num_classes: auto_from_training_buildings
    print(f"fold={fold} N={num_classes} D={model_cfg.d_model} train={len(ds_train)} val={len(ds_val)}")

    model = DualContextTransformer(model_cfg).to(device)
    head = BuildingIdHead(model_cfg.d_model, num_classes).to(device)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=pcfg["lr"], weight_decay=pcfg["weight_decay"])
    n = pcfg.get("samples_per_epoch") or len(ds_train)
    loader = DataLoader(
        ds_train, batch_size=pcfg["batch_size"], sampler=RandomSampler(ds_train, num_samples=n),
        num_workers=cfg.get("num_workers", 0),
    )
    val_loader = DataLoader(ds_val, batch_size=pcfg["batch_size"], num_workers=cfg.get("num_workers", 0))

    out = run_dir(cfg, "pretrain", fold)
    best = float("inf")
    for epoch in range(pcfg["epochs"]):
        model.train(), head.train()
        for step, batch in enumerate(loader):
            batch = to_device(batch, device)
            loss = id_loss(head(model(**backbone_inputs(batch))), batch["building_target"], pcfg["loss_on"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, pcfg.get("grad_clip", 1.0))
            opt.step()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        metrics = evaluate(model, head, val_loader, device, pcfg["loss_on"])
        print(f"epoch {epoch} val {json.dumps(metrics)}")
        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save(
                {
                    "model_cfg": model_cfg.to_dict(),
                    "backbone": model.state_dict(),
                    "id_head": head.state_dict(),
                    "normalizer": norm.state_dict(),
                    "train_building_ids": grid.building_ids[train_idx].tolist(),
                    "fold": fold,
                    "cv": cfg["cv"],
                    "seed": cfg["seed"],
                    "val": metrics,
                },
                out / "best.pt",
            )
    print(f"saved {out / 'best.pt'} (best val loss {best:.4f})")


if __name__ == "__main__":
    main()
