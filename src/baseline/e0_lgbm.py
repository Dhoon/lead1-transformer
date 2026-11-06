"""E0: LEAD 우승 솔루션(Fu et al., BuildSys '22) 재현 Baseline. Transformer와 같은 Building 5-Fold로 비교한다.

python -m src.baseline.e0_lgbm                      # 5 Fold 검증 + Fold 모델 5개 평균으로 test 제출 파일
python -m src.baseline.e0_lgbm --folds 0 --no-test  # Fold 0만

논문과 같은 점: 건물별 평균으로 결측 대체, Value-change Feature(차이·비율, Shift ±1,2,3,23,24,48,72,168),
정상 Downsampling(정상:이상 = 1:1), 후처리(meter_reading == 1 -> 1, 건물별 첫·마지막 시각 -> 0).
다른 점: LightGBM 하나만 쓴다 (논문은 4개 트리 모델 평균, 단일 대비 +0.002). 건물 조합 문자열 Feature는 뺀다.
Shift는 Row 순서가 아니라 시간 격자 기준이다 (CSV에 없는 시각이 있어도 정확히 s시간 전후와 비교).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.lead_loader import load_split
from src.data.splits import select_fold
from src.eval.metrics import hourly_metrics

SHIFTS = [1, 2, 3, 23, 24, 48, 72, 168]
# 건물 ID가 들어간 조합 문자열은 미학습 건물에서 의미가 없고, 시간 조합은 hour/weekday와 중복이다.
DROP = ["timestamp", "anomaly", "row_id", "year", "weekday_hour", "building_weekday_hour",
        "building_weekday", "building_month", "building_hour", "building_meter"]
LGB_PARAMS = dict(
    objective="binary", learning_rate=0.05, n_estimators=1000, num_leaves=127, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
)


def value_change(values: np.ndarray, shifts=SHIFTS) -> dict[str, np.ndarray]:
    """`[nb, n_days, 24]` 격자(결측 대체 후)에서 시각별 차이 X(t)-X(t-s)와 비율 (X(t)+1)/(X(t-s)+1).

    s > 0은 과거, s < 0은 미래 시점과 비교한다. 시계열 밖은 nan.
    """
    x = values.reshape(len(values), -1)
    out = {}
    for s in [k for k in shifts] + [-k for k in shifts]:
        prev = np.full_like(x, np.nan)
        if s > 0:
            prev[:, s:] = x[:, :-s]
        else:
            prev[:, :s] = x[:, -s:]
        out[f"value_chg_diff_{s}"] = (x - prev).reshape(values.shape)
        out[f"value_chg_ratio_{s}"] = ((x + 1) / (prev + 1)).reshape(values.shape)
    return out


def build_frame(raw_dir: str, cache_dir: str, split: str) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """`{split}_features.csv` + Value-change Feature. 반환: (Feature, raw meter_reading, 격자상 건물 위치)."""
    grid = load_split(raw_dir, split, cache_dir)
    df = pd.read_csv(Path(raw_dir) / f"{split}_features.csv")
    ts = pd.to_datetime(df["timestamp"])
    b = np.searchsorted(grid.building_ids, df["building_id"].to_numpy())
    d = (ts.dt.normalize() - grid.start).dt.days.to_numpy()
    h = ts.dt.hour.to_numpy()

    # 결측 대체: 건물별 관측값 평균 (논문 2.1)
    mean = np.nanmean(grid.values.reshape(len(grid.values), -1), axis=1)
    filled = np.where(np.isnan(grid.values), mean[:, None, None], grid.values)
    raw = df["meter_reading"].copy()
    df["meter_reading"] = filled[b, d, h]
    for name, v in value_change(filled).items():
        df[name] = v[b, d, h].astype(np.float32)
    df["primary_use"] = df["primary_use"].astype("category")
    return df, raw, b


def postprocess(prob: np.ndarray, raw: pd.Series, building: np.ndarray) -> np.ndarray:
    """논문 2.4: meter_reading == 1은 이상, 건물별 첫·마지막 시각은 정상 (Row가 시간순이라고 가정)."""
    prob = prob.copy()
    prob[(raw == 1).to_numpy()] = 1.0
    order = np.lexsort((np.arange(len(building)), building))
    sb = building[order]
    edge = np.r_[True, sb[1:] != sb[:-1]] | np.r_[sb[1:] != sb[:-1], True]
    prob[order[edge]] = 0.0
    return prob


def fit(X: pd.DataFrame, y: np.ndarray, seed: int) -> lgb.LGBMClassifier:
    """정상 Row를 이상 Row 수만큼 무작위로 남겨 1:1로 맞춘 뒤 학습한다 (논문 2.3.2)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    keep = np.sort(np.concatenate([pos, rng.choice(neg, size=len(pos), replace=False)]))
    model = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed, n_jobs=-1)
    model.fit(X.iloc[keep], y[keep])
    return model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--cache-dir", default="data/processed")
    ap.add_argument("--out", type=Path, default=Path("runs/e0_lgbm"))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)  # configs/*.yaml의 seed와 같아야 같은 Fold가 된다
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--no-test", action="store_true")
    args = ap.parse_args(argv)
    folds = list(range(args.n_splits)) if args.folds is None else args.folds

    df, raw, b = build_frame(args.raw_dir, args.cache_dir, "train")
    feats = [c for c in df.columns if c not in DROP]
    y = df["anomaly"].to_numpy()
    building_ids = load_split(args.raw_dir, "train", args.cache_dir).building_ids
    print(f"train rows={len(df)} features={len(feats)} pos_rate={y.mean():.4f}")

    models, summary = [], {}
    for k in folds:
        tr_idx, va_idx = select_fold(building_ids, k, args.n_splits, args.seed)
        tr, va = np.isin(b, tr_idx), np.isin(b, va_idx)
        model = fit(df.loc[tr, feats], y[tr], args.seed + k)
        prob = model.predict_proba(df.loc[va, feats])[:, 1]
        res = {
            "raw": hourly_metrics(y[va], prob),
            "postprocess": hourly_metrics(y[va], postprocess(prob, raw[va], b[va])),
        }
        out = args.out / f"fold{k}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(res, indent=2))
        np.savez_compressed(out / "val_prob.npz", prob=prob, row=np.flatnonzero(va))
        imp = pd.Series(model.booster_.feature_importance("gain"), feats).sort_values(ascending=False)
        imp.to_csv(out / "importance.csv", header=["gain"])
        summary[k] = res
        models.append(model)
        r, p = res["raw"], res["postprocess"]
        print(f"fold {k}: raw roc {r['roc_auc']:.4f} pr {r['pr_auc']:.4f} f1 {r['f1']:.4f} | "
              f"post roc {p['roc_auc']:.4f} pr {p['pr_auc']:.4f} f1 {p['f1']:.4f}")

    for kind in ("raw", "postprocess"):
        for m in ("roc_auc", "pr_auc", "f1"):
            v = [summary[k][kind][m] for k in folds]
            print(f"{kind:12s} {m:8s} {np.mean(v):.4f} ± {np.std(v):.4f}")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    if not args.no_test:
        del df
        tdf, traw, tb = build_frame(args.raw_dir, args.cache_dir, "test")
        prob = np.mean([m.predict_proba(tdf[feats])[:, 1] for m in models], axis=0)
        sample = pd.read_csv(Path(args.raw_dir) / "sample_submission.csv")
        for name, p in (("submission_raw.csv", prob), ("submission_post.csv", postprocess(prob, traw, tb))):
            sub = sample[["row_id"]].merge(pd.DataFrame({"row_id": tdf["row_id"], "anomaly": p}), on="row_id", how="left")
            assert len(sub) == len(sample) and not sub["anomaly"].isna().any()
            sub.to_csv(args.out / name, index=False)
            print(f"saved {args.out / name} mean {p.mean():.4f}")


if __name__ == "__main__":
    main()
