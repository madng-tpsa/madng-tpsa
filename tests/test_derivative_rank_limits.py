"""Guard against accidentally constructing exponentially large TPSA tensors."""

import pytest

from madng_tpsa import Descriptor, TpsaTensor


def test_extreme_rank_is_rejected_before_allocating_components() -> None:
    d = Descriptor(variables=('rank_q', 'rank_p'), order=4)
    t = TpsaTensor([d.var(1), d.var(2)])
    with pytest.raises(ValueError, match='components|limit|large'):
        t.derivative_tensor(21)


def test_configurable_rank_limit_and_valid_small_tensor() -> None:
    d = Descriptor(variables=('rank_a', 'rank_b'), order=4)
    t = TpsaTensor([d.var(1), d.var(2)])
    with pytest.raises(ValueError, match='components|limit|large'):
        t.derivative_tensor(3, max_components=8)
    higher = t.derivative_tensor(3, max_components=16)
    assert higher.shape == (2, 2, 2, 2)
    assert all(series.is_zero() for series in higher.flat)
