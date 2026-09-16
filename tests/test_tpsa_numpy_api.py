"""Tests for TPSA mathematical functions and NumPy ufunc dispatch."""

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import scipy.special

import madng_tpsa

if TYPE_CHECKING:
    from typing import Any


def _constant(value: float) -> madng_tpsa.Tpsa:
    return madng_tpsa.Descriptor(1, 3).constant(value)


@pytest.mark.parametrize(
    ('method_name', 'numpy_func', 'value', 'method_expected', 'numpy_expected'),
    [
        ('abs', np.abs, -2.0, 2.0, 2.0),
        ('sqrt', np.sqrt, 4.0, 2.0, 2.0),
        ('exp', np.exp, 0.5, math.exp(0.5), math.exp(0.5)),
        ('log', np.log, 2.0, math.log(2.0), math.log(2.0)),
        ('sin', np.sin, 0.5, math.sin(0.5), math.sin(0.5)),
        ('cos', np.cos, 0.5, math.cos(0.5), math.cos(0.5)),
        ('tan', np.tan, 0.5, math.tan(0.5), math.tan(0.5)),
        ('sinh', np.sinh, 0.5, math.sinh(0.5), math.sinh(0.5)),
        ('cosh', np.cosh, 0.5, math.cosh(0.5), math.cosh(0.5)),
        ('tanh', np.tanh, 0.5, math.tanh(0.5), math.tanh(0.5)),
        ('asin', np.arcsin, 0.5, math.asin(0.5), math.asin(0.5)),
        ('acos', np.arccos, 0.5, math.acos(0.5), math.acos(0.5)),
        ('atan', np.arctan, 0.5, math.atan(0.5), math.atan(0.5)),
        ('asinh', np.arcsinh, 0.5, math.asinh(0.5), math.asinh(0.5)),
        ('acosh', np.arccosh, 2.0, math.acosh(2.0), math.acosh(2.0)),
        ('atanh', np.arctanh, 0.5, math.atanh(0.5), math.atanh(0.5)),
        ('sinc', np.sinc, 0.5, math.sin(0.5) / 0.5, np.sinc(0.5)),
        ('sinhc', None, 0.5, math.sinh(0.5) / 0.5, None),
    ],
)
def test_unary_math_methods_and_numpy_ufuncs(
    method_name,
    numpy_func,
    value,
    method_expected,
    numpy_expected,
):
    t = _constant(value)
    method_result = getattr(t, method_name)()

    assert method_result.const_part == pytest.approx(method_expected)
    if numpy_func is not None:
        numpy_result = numpy_func(t)
        assert numpy_result.const_part == pytest.approx(numpy_expected)
        if numpy_expected == method_expected:
            assert numpy_result == method_result


def test_norm_and_unit():
    d = madng_tpsa.Descriptor(1, 2)
    t = d.var(1, -2.0)
    t.set((2,), -3.0)

    assert t.norm() == pytest.approx(6.0)

    unit = t.unit()
    assert unit.const_part == pytest.approx(-1.0)
    assert unit.grad() == pytest.approx([0.5])
    assert unit.get((2,)) == pytest.approx(-1.5)


def test_unit_rejects_zero_constant_part():
    with pytest.raises(ZeroDivisionError, match='zero constant part'):
        madng_tpsa.Descriptor(1, 2).var(1).unit()


def test_power_uses_tpsa_integer_and_float_paths():
    d = madng_tpsa.Descriptor(1, 3)
    x = d.var(1, 2.0)
    y = d.var(1, 3.0)

    assert (x**3).const_part == pytest.approx(8.0)
    assert (x**0.5).const_part == pytest.approx(math.sqrt(2.0))
    assert (x**y).const_part == pytest.approx(8.0)


def test_selected_math_derivatives():
    d = madng_tpsa.Descriptor(1, 3)
    x = d.var(1, 2.0)

    assert x.exp().grad() == pytest.approx([math.exp(2.0)])
    assert np.sin(x).grad() == pytest.approx([math.cos(2.0)])  # ty: ignore[unresolved-attribute, no-matching-overload, unused-ignore-comment]
    assert (x**3).grad() == pytest.approx([12.0])
    assert (x**0.5).grad() == pytest.approx([1.0 / (2.0 * math.sqrt(2.0))])


@pytest.mark.parametrize(
    ('method_name', 'special_func', 'value', 'expected'),
    [
        ('erf', scipy.special.erf, 0.5, cast('Any', scipy.special.erf)(0.5)),
        ('erfc', scipy.special.erfc, 0.5, cast('Any', scipy.special.erfc)(0.5)),
        ('erfcx', scipy.special.erfcx, 0.5, cast('Any', scipy.special.erfcx)(0.5)),
        ('erfi', scipy.special.erfi, 0.5, cast('Any', scipy.special.erfi)(0.5)),
    ],
)
def test_scipy_special_methods_and_ufuncs(method_name, special_func, value, expected):
    t = _constant(value)
    method_result = getattr(t, method_name)()

    assert method_result.const_part == pytest.approx(expected)
    assert special_func(t) == method_result


def test_wofz_of_a_real_tpsa_returns_a_complex_tpsa():
    t = _constant(0.5)
    expected = scipy.special.wofz(np.complex128(0.5))

    assert t.wofz_real().const_part == pytest.approx(expected.real)

    result = t.__array_ufunc__(scipy.special.wofz, '__call__', t)

    assert isinstance(result, madng_tpsa.ComplexTpsa)
    assert result.const_part == pytest.approx(expected)


@pytest.mark.parametrize(
    ('numpy_func', 'left', 'right', 'expected'),
    [
        (np.add, 'x', 3.0, 5.0),
        (np.subtract, 3.0, 'x', 1.0),
        (np.multiply, 'x', 3.0, 6.0),
        (np.divide, 6.0, 'x', 3.0),
        (np.power, 'x', 3, 8.0),
        (np.atan2, 'x', 3.0, math.atan2(2.0, 3.0)),
        (np.hypot, 'x', 3.0, math.hypot(2.0, 3.0)),
    ],
)
def test_numpy_binary_ufuncs(numpy_func, left, right, expected):
    d = madng_tpsa.Descriptor(1, 3)
    x = d.var(1, 2.0)

    left_arg = x if left == 'x' else left
    right_arg = x if right == 'x' else right

    assert numpy_func(left_arg, right_arg).const_part == pytest.approx(expected)


def test_numpy_scalar_left_binary_operations():
    x = madng_tpsa.Descriptor(1, 3).var(1, 2.0)
    scalar = cast('Any', np.float64(3.0))

    assert (scalar + x).const_part == pytest.approx(5.0)
    assert (scalar - x).const_part == pytest.approx(1.0)
    assert (scalar * x).const_part == pytest.approx(6.0)
    assert (scalar / x).const_part == pytest.approx(1.5)
    assert (scalar**x).const_part == pytest.approx(9.0)


def test_hypot3_method():
    d = madng_tpsa.Descriptor(1, 3)
    x = d.var(1, 2.0)

    assert x.hypot3(3.0, 6.0).const_part == pytest.approx(7.0)


@pytest.mark.parametrize(
    ('numpy_func', 'expected'),
    [
        (np.negative, -2.0),
        (np.positive, 2.0),
    ],
)
def test_numpy_sign_ufuncs(numpy_func, expected):
    x = madng_tpsa.Descriptor(1, 3).var(1, 2.0)

    assert numpy_func(x).const_part == pytest.approx(expected)
