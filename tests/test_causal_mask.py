"""Decoder Causal Self-Attention 단독은 미래 날짜를 보지 않는다.

전체 모델은 Encoder가 Window T일 전체를 양방향으로 보므로 오프라인/비인과적 모델이다 (SPEC 4.5, 5.3).
"""
import torch

from src.models.dual_context import DualContextTransformer
from src.models.temporal_decoder import causal_mask
from tests.conftest import random_batch, small_cfg


def _change_after(batch, d):
    changed = {k: v.clone() for k, v in batch.items()}
    changed["target_daily"][:, d + 1 :] += 10.0
    changed["zone_daily"][:, d + 1 :] += 10.0
    changed["type_daily"][:, d + 1 :] += 10.0
    return changed


def test_mask_pattern():
    m = causal_mask(4)
    assert not m[2, :3].any() and m[2, 3:].all()


def test_decoder_self_attention_alone_is_causal():
    model = DualContextTransformer(small_cfg(cross_attention_order=[])).eval()
    batch, d = random_batch(), 2
    base = model(batch["target_daily"])[:, : d + 1]
    changed = _change_after(batch, d)
    assert torch.allclose(model(changed["target_daily"])[:, : d + 1], base, atol=1e-5)


def test_full_model_is_offline_non_causal(model):
    """Encoder가 이후 날짜를 보므로 초기 날짜 예측도 바뀔 수 있다. 의도된 동작임을 고정한다."""
    batch, d = random_batch(), 2
    base = model(**batch)[:, : d + 1]
    assert not torch.allclose(model(**_change_after(batch, d))[:, : d + 1], base, atol=1e-5)
