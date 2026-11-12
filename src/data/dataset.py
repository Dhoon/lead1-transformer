"""가변 길이 T일 Window Dataset, 샘플러, Collate (SPEC 3.3, 7.3)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class WindowDataset(Dataset):
    """키 `(target_pos, start, T)` -> 대상 건물과 Zone/Type 그룹의 같은 연속 T일.

    x: 결측 대체·정규화가 끝난 전체 격자 `[n_all, n_days, 24]` (NaN 없음).
    targets: 대상 건물 행 인덱스. zone_idx / type_idx: `[len(targets), G]` (열 0 = 대상 건물, -1 = 패딩).
    labels / label_mask: Train에서만 준다. observed: `observation_mask` 보관용 (모델 입력 아님).
    """

    def __init__(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        zone_idx: np.ndarray | None,
        type_idx: np.ndarray | None,
        observed: np.ndarray,
        labels: np.ndarray | None = None,
        label_mask: np.ndarray | None = None,
        mask_missing_peers: bool = False,
    ):
        """mask_missing_peers: True면 하루 24시간이 모두 결측인 Peer를 그 날짜의 Attention에서 가린다
        (`{zone,type}_mask`에서 False). 대상 건물(그룹 열 0)은 결측이어도 가리지 않는다 (Zero-fill 값 그대로)."""
        self.x, self.observed = x, observed
        self.mask_missing_peers = mask_missing_peers
        self.targets = np.asarray(targets)
        self.zone_idx, self.type_idx = zone_idx, type_idx
        self.labels, self.label_mask = labels, label_mask

    @property
    def n_days(self) -> int:
        return self.x.shape[1]

    def __len__(self) -> int:
        return len(self.targets)

    def _group(self, idx: np.ndarray, sl: slice):
        members = idx[idx >= 0]
        return torch.from_numpy(np.ascontiguousarray(self.x[members, sl].transpose(1, 0, 2)))  # [T, G, 24]

    def _group_valid(self, idx: np.ndarray, sl: slice):
        """`[T, G]` True = 그날 관측이 하나라도 있는 Token. 열 0(대상 건물)은 항상 True."""
        members = idx[idx >= 0]
        valid = self.observed[members, sl].any(-1).T.copy()
        valid[:, 0] = True
        return torch.from_numpy(valid)

    def __getitem__(self, key) -> dict:
        i, start, t = key
        b = int(self.targets[i])
        sl = slice(start, start + t)
        item = {
            "target_daily": torch.from_numpy(self.x[b, sl]),
            "observation_mask": torch.from_numpy(self.observed[b, sl]),
            "target_row": b,
            "start": start,
        }
        for name, idx in (("zone", self.zone_idx), ("type", self.type_idx)):
            if idx is None:
                continue
            item[f"{name}_daily"] = self._group(idx[i], sl)
            if self.mask_missing_peers:
                item[f"{name}_valid"] = self._group_valid(idx[i], sl)
        if self.labels is not None:
            item["anomaly_target"] = torch.from_numpy(self.labels[b, sl].astype(np.float32))
            item["label_mask"] = torch.from_numpy(self.label_mask[b, sl])
        return item


def collate_windows(items: list[dict]) -> dict:
    """같은 T끼리 묶인 배치를 받는다. 그룹 크기 차이를 Padding하고 `[B, T×G]` 유효 Token Mask를 만든다.

    `{name}_valid`가 있으면(mask_missing_peers) 결측 Peer Token도 Mask에서 False로 둔다.
    """
    t = items[0]["target_daily"].shape[0]
    assert all(it["target_daily"].shape[0] == t for it in items), "배치 내 T가 같아야 한다."
    out = {k: torch.stack([it[k] for it in items]) for k in ("target_daily", "observation_mask")}
    out["target_row"] = torch.tensor([it["target_row"] for it in items])
    out["start"] = torch.tensor([it["start"] for it in items])
    for name in ("zone", "type"):
        if f"{name}_daily" not in items[0]:
            continue
        groups = [it[f"{name}_daily"] for it in items]
        g = max(x.shape[1] for x in groups)
        daily = torch.zeros(len(items), t, g, 24)
        member = torch.zeros(len(items), t, g, dtype=torch.bool)
        for k, x in enumerate(groups):
            daily[k, :, : x.shape[1]] = x
            member[k, :, : x.shape[1]] = items[k][f"{name}_valid"] if f"{name}_valid" in items[k] else True
        out[f"{name}_daily"] = daily
        out[f"{name}_mask"] = member.reshape(len(items), t * g)  # day-major
    if "anomaly_target" in items[0]:
        out["anomaly_target"] = torch.stack([it["anomaly_target"] for it in items])
        out["label_mask"] = torch.stack([it["label_mask"] for it in items])
    return out


def batch_size_for(t: int, group_size: int, batch_size: int, max_tokens: int | None) -> int:
    """긴 T에서 Encoder Token 수(T×G)가 커지므로 배치당 Token 예산으로 배치 크기를 줄인다."""
    if not max_tokens:
        return batch_size
    return max(1, min(batch_size, max_tokens // (t * group_size)))


class RandomWindowBatchSampler(Sampler):
    """배치마다 T ~ Uniform{min_T..max_T}를 뽑고, 대상 건물과 시작 날짜를 무작위로 고른다.

    한 배치 안의 샘플은 같은 T를 공유한다 (SPEC 7.3 구현 주의: 같은 길이끼리 배치 구성).
    """

    def __init__(
        self, n_targets: int, n_days: int, min_t: int, max_t: int, samples_per_epoch: int,
        batch_size: int, group_size: int, max_tokens: int | None, seed: int,
    ):
        self.n_targets, self.n_days = n_targets, n_days
        self.min_t, self.max_t = min_t, min(max_t, n_days)
        self.samples_per_epoch, self.batch_size = samples_per_epoch, batch_size
        self.group_size, self.max_tokens = group_size, max_tokens
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        seen = 0
        while seen < self.samples_per_epoch:
            t = int(self.rng.integers(self.min_t, self.max_t + 1))
            b = batch_size_for(t, self.group_size, self.batch_size, self.max_tokens)
            targets = self.rng.integers(0, self.n_targets, size=b)
            starts = self.rng.integers(0, self.n_days - t + 1, size=b)
            yield [(int(i), int(s), t) for i, s in zip(targets, starts)]
            seen += b


def inference_windows(n_days: int, window: int, stride: int) -> list[tuple[int, int]]:
    """(start, T) 목록. 모든 날짜를 최소 한 번 덮으며, 마지막 Window는 남은 날짜만큼 짧아질 수 있다."""
    if not 0 < stride <= window:
        raise ValueError("stride는 1 이상 window 이하여야 모든 날짜를 덮는다.")
    window = min(window, n_days)
    starts = list(range(0, n_days - window + 1, stride))
    if starts[-1] + window < n_days:
        starts.append(starts[-1] + stride if stride < window else starts[-1] + window)
    return [(s, min(window, n_days - s)) for s in starts]


def inference_batches(
    n_targets: int, n_days: int, window: int, stride: int, batch_size: int, group_size: int, max_tokens: int | None
) -> list[list[tuple[int, int, int]]]:
    """모든 대상 건물 x 추론 Window를 같은 T끼리 배치로 묶는다."""
    by_len: dict[int, list] = {}
    for s, t in inference_windows(n_days, window, stride):
        by_len.setdefault(t, []).extend((i, s, t) for i in range(n_targets))
    batches = []
    for t, keys in by_len.items():
        b = batch_size_for(t, group_size, batch_size, max_tokens)
        batches.extend(keys[k : k + b] for k in range(0, len(keys), b))
    return batches
