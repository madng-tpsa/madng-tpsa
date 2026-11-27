"""Variadic mixed-derivative API and common scalar/tensor parsing."""

import numpy as np
import pytest

from madng_tpsa import ComplexTpsa, Descriptor, TpsaTensor


def test_variadic_directions_and_explicit_orders():
    d = Descriptor(var_names=('q', 'p'), param_names=('k',), order=5, param_order=3)
    q, p = d.vars()
    k = d.param('k')
    f = q**2 * k + p
    tensor = TpsaTensor([f])
    complex_q = ComplexTpsa.from_tpsa(q)
    assert f.derivative(('k', 1), ('q', 2)) == d.constant(2)
    assert f.derivative((3, 1), (1, 2)) == d.constant(2)
    assert f.derivative(q, complex_q, k) == d.constant(2)
    assert tensor.partial_derivative(('k', 1), ('q', 2))[0] == d.constant(2)
    assert tensor.partial_derivative((3, 1), (1, 2))[0] == d.constant(2)
    assert tensor.partial_derivative(q, complex_q, k)[0] == d.constant(2)
    assert f.derivative(orders=(2, 0, 1)) == d.constant(2)
    assert tensor.partial_derivative(orders=(2, 0, 1))[0] == d.constant(2)
    assert f.derivative(('k', 2), 'q').is_zero()
    assert f.derivative(('q', 2), 'q').is_zero()


def test_integer_pair_not_exponent_vector():
    d = Descriptor(var_names=('q', 'p'), order=4)
    q, p = d.vars()
    f = q**2 * p
    assert f.derivative((1, 2), 2) == d.constant(2)
    assert f.derivative(orders=(2, 1)) == d.constant(2)
    assert f.derivative((2, 1)) == q**2


def test_reject_invalid_unit_directions_and_specifications():
    d = Descriptor(var_names=('q', 'p'), order=3)
    q, p = d.vars()
    f = q**2 + p
    bad = [2 * q, q + p, q**2, d.constant(1), 2 * ComplexTpsa.from_tpsa(q)]
    for direction in bad:
        with pytest.raises(ValueError, match='unit first-order'):
            f.derivative(direction)
    with pytest.raises(ValueError, match='incompatible descriptor'):
        f.derivative(Descriptor(1, 3).var(1))
    for direction in (True, np.bool_(True), 1.5):  # noqa: FBT003
        with pytest.raises(TypeError):
            f.derivative(direction)  # type: ignore
    with pytest.raises(ValueError):
        f.derivative(orders=(0, 0))
    with pytest.raises(TypeError):
        f.derivative(q, orders=(1, 0))
    with pytest.raises(TypeError):
        f.derivative(('q', 1.5))  # type: ignore
    with pytest.raises(ValueError):
        f.derivative(('q', 0))
    assert f.derivative(orders=(4, 0)).is_zero()
