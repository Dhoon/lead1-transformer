import pytest

from src.train.train_supervised import lr_factor


def test_constant_is_one():
    assert all(lr_factor("constant", p, 0.02) == 1.0 for p in (0.0, 0.5, 1.0))


def test_cosine_warmup_then_decay():
    assert lr_factor("cosine", 0.0, 0.02) == 0.0
    assert lr_factor("cosine", 0.01, 0.02) == pytest.approx(0.5)
    assert lr_factor("cosine", 0.02, 0.02) == pytest.approx(1.0)
    assert lr_factor("cosine", 0.51, 0.02) == pytest.approx(0.5)
    assert lr_factor("cosine", 1.0, 0.02) == pytest.approx(0.0, abs=1e-12)
    xs = [lr_factor("cosine", i / 100, 0.02) for i in range(2, 101)]
    assert all(a >= b for a, b in zip(xs, xs[1:]))


def test_cosine_without_warmup_starts_at_one():
    assert lr_factor("cosine", 0.0, 0.0) == pytest.approx(1.0)
