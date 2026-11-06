import numpy as np
import pandas as pd

from src.baseline.e0_lgbm import postprocess, value_change


def test_value_change_uses_time_shift():
    values = np.arange(2 * 2 * 24, dtype=float).reshape(2, 2, 24)  # 건물 2개 x 2일
    vc = value_change(values, shifts=[1, 24])
    flat = {k: v.reshape(2, -1) for k, v in vc.items()}
    assert np.isnan(flat["value_chg_diff_1"][:, 0]).all()  # 건물 경계를 넘지 않는다
    assert (flat["value_chg_diff_1"][:, 1:] == 1).all()
    assert (flat["value_chg_diff_24"][:, 24:] == 24).all()
    assert (flat["value_chg_diff_-1"][:, :-1] == -1).all() and np.isnan(flat["value_chg_diff_-1"][:, -1]).all()
    assert flat["value_chg_ratio_1"][0, 1] == (1 + 1) / (0 + 1)


def test_postprocess_rules():
    building = np.array([5, 7, 5, 7, 5, 7])  # 시간순으로 건물이 번갈아 나온다
    raw = pd.Series([np.nan, 3.0, 1.0, 2.0, 4.0, 1.0])
    out = postprocess(np.full(6, 0.3), raw, building)
    # 건물 5: Row 0(첫), 4(마지막) -> 0, Row 2(값 1) -> 1 / 건물 7: Row 1, 5 -> 0 (마지막이 1이어도 경계 규칙이 우선)
    assert out.tolist() == [0.0, 0.0, 1.0, 0.3, 0.0, 0.0]
