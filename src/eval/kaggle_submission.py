"""격자 예측 -> Kaggle 제출 파일 (SPEC 10: 원래 Row 매핑, 1시간 단위 확률)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def grid_to_rows(prob: np.ndarray, row_id: np.ndarray, present: np.ndarray) -> pd.DataFrame:
    """`[nb, n_days, 24]` 예측을 원본 row_id에 매핑한다."""
    return pd.DataFrame({"row_id": row_id[present], "anomaly": prob[present]})


def make_submission(prob: np.ndarray, row_id: np.ndarray, present: np.ndarray, sample: pd.DataFrame) -> pd.DataFrame:
    """sample_submission의 Row 순서와 개수를 그대로 따른다. 누락·중복 Row가 있으면 실패한다."""
    rows = grid_to_rows(prob, row_id, present)
    if rows["row_id"].duplicated().any():
        raise ValueError("중복 row_id가 있습니다.")
    sub = sample[["row_id"]].merge(rows, on="row_id", how="left")
    if sub["anomaly"].isna().any() or len(sub) != len(sample):
        raise ValueError(f"예측이 없는 Row {int(sub['anomaly'].isna().sum())}개")
    return sub
