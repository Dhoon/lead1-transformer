"""(대상 건물, 종료 날짜) 단위 Window Dataset. SPEC 3.3의 Tensor 규약을 따른다."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


class WindowDataset(Dataset):
    """대상 건물의 연속 T일과, 같은 날짜의 Zone/Type Peer 24시간 벡터를 묶는다.

    x, valid: 전체 격자 `[n_all, n_days, 24]`. x는 정규화·결측 대체가 끝난 값이다.
    targets: 대상 건물 행 인덱스. zone_idx / type_idx: `[len(targets), G]` Peer 행 인덱스(-1 = 패딩), None이면 생략.
    end_days: Window의 마지막 날짜. 시작 이전 날짜는 결측 날짜로 채운다.

    Pretraining: `class_index`(대상별 클래스)와 `day_ok`(손실을 줄 날짜) `[n_all, n_days]`를 준다.
    Downstream: `labels`와 `label_mask` `[n_all, n_days, 24]`를 준다.
    Building ID는 정답(`building_target`)으로만 반환하며 입력 Tensor에는 들어가지 않는다.
    """

    def __init__(
        self,
        x: np.ndarray,
        valid: np.ndarray,
        targets: np.ndarray,
        window: int,
        end_days,
        zone_idx: np.ndarray | None = None,
        type_idx: np.ndarray | None = None,
        class_index: np.ndarray | None = None,
        day_ok: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        label_mask: np.ndarray | None = None,
        shuffle_pool: np.ndarray | None = None,
        seed: int = 0,
    ):
        pad = window - 1
        self.window = window
        self.x = np.concatenate([np.zeros_like(x[:, :pad]), x], axis=1)
        self.valid = np.concatenate([np.zeros_like(valid[:, :pad]), valid], axis=1)
        self.targets = np.asarray(targets)
        self.zone_idx, self.type_idx = zone_idx, type_idx
        self.class_index = class_index
        self.day_ok = None if day_ok is None else np.concatenate([np.zeros_like(day_ok[:, :pad]), day_ok], axis=1)
        self.labels = None if labels is None else np.concatenate([np.zeros_like(labels[:, :pad]), labels], axis=1)
        self.label_mask = (
            None if label_mask is None else np.concatenate([np.zeros_like(label_mask[:, :pad]), label_mask], axis=1)
        )
        # Context permutation 실험(SPEC 5.3, E7): Peer를 같은 수의 무작위 건물로 바꾼다.
        self.shuffle_pool = shuffle_pool
        self.rng = np.random.default_rng(seed)
        self.end_days = np.asarray(list(end_days))
        self.samples = [(i, int(e)) for i in range(len(self.targets)) for e in self.end_days]

    def __len__(self) -> int:
        return len(self.samples)

    def _peers(self, idx: np.ndarray, target: int, sl: slice):
        idx = idx.copy()
        if self.shuffle_pool is not None:
            n = int((idx >= 0).sum())
            pool = self.shuffle_pool[self.shuffle_pool != target]
            idx[:n] = self.rng.choice(pool, size=n, replace=False)
        safe = np.where(idx < 0, 0, idx)
        daily = self.x[safe, sl].transpose(1, 0, 2)  # [T, G, 24]
        mask = (idx >= 0)[None, :] & self.valid[safe, sl].any(-1).T  # [T, G]
        return torch.from_numpy(np.ascontiguousarray(daily)), torch.from_numpy(mask)

    def __getitem__(self, k: int) -> dict:
        i, e = self.samples[k]
        t = int(self.targets[i])
        sl = slice(e, e + self.window)  # 패딩 기준 인덱스: 원래 날짜 e-T+1..e
        item = {
            "target_daily": torch.from_numpy(self.x[t, sl]),
            "day_valid_mask": torch.from_numpy(self.valid[t, sl]),
            "target_row": t,
            "end_day": e,
        }
        if self.zone_idx is not None:
            item["zone_daily"], item["zone_mask"] = self._peers(self.zone_idx[i], t, sl)
        if self.type_idx is not None:
            item["type_daily"], item["type_mask"] = self._peers(self.type_idx[i], t, sl)
        if self.class_index is not None:
            ok = self.day_ok[t, sl]
            item["building_target"] = torch.from_numpy(np.where(ok, self.class_index[i], IGNORE_INDEX).astype(np.int64))
        if self.labels is not None:
            item["hour_labels"] = torch.from_numpy(self.labels[t, sl].astype(np.float32))
            item["label_mask"] = torch.from_numpy(self.label_mask[t, sl])
        return item
