"""Tests for complex TPSA series."""

from decimal import Decimal
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import scipy.special

import madng_tpsa
from madng_tpsa._cffi import lib

if TYPE_CHECKING:
    from typing import Any


def test_complex_variable_and_parameter_seeds():
    d = madng_tpsa.Descriptor(variables=['x', 'y'], order=3, params=['k'])
    x = d.var('x', 1 + 2j)
    y = d.var('y', 0j)
    k = d.param('k', 3 - 4j)

    assert x.const_part == pytest.approx(1 + 2j)
    assert x.grad() == pytest.approx([1, 0])
    assert y.grad() == pytest.approx([0, 1])
    assert k.const_part == pytest.approx(3 - 4j)
    assert k.param_grad() == pytest.approx([1])


def test_complex_coefficients_and_copy_are_independent():
    d = madng_tpsa.Descriptor(2, 2)
    t = d.complex_zero()
    t.set_const_part(1 + 2j)
    t[(1, 1)] = -3 + 4j

    copied = t.copy()
    t.set_const_part(9j)

    assert copied.to_dict() == pytest.approx({(0, 0): 1 + 2j, (1, 1): -3 + 4j})
    assert t.coefficient((1, 1)) == pytest.approx(-3 + 4j)
    np.testing.assert_allclose(t.coefficient([(0, 0), (1, 1)]), [9j, -3 + 4j])


def test_complex_series_coefficient_state_operations():
    d = madng_tpsa.Descriptor(2, 3)
    t = d.complex_zero()

    assert t.is_zero()
    assert t.is_constant()
    assert t.max_nonzero_order == 0

    t.from_dict({(0, 0): 1 + 2j, (2, 0): 3 - 4j})

    assert t[(2, 0)] == pytest.approx(3 - 4j)
    assert not t.is_zero()
    assert not t.is_constant()
    assert t.max_nonzero_order == 2
    assert t.norm() == pytest.approx(abs(1 + 2j) + abs(3 - 4j))

    t.clear()

    assert t.is_zero()
    assert t.is_constant()


def test_complex_unit_normalises_by_the_constant_part_magnitude():
    d = madng_tpsa.Descriptor(1, 2)
    t = d.constant(3 + 4j)
    assert isinstance(t, madng_tpsa.ComplexTpsa)
    t[(1,)] = 1 + 2j

    result = t.unit()

    assert result.const_part == pytest.approx((3 + 4j) / 5)
    assert result[(1,)] == pytest.approx((1 + 2j) / 5)

    with pytest.raises(ZeroDivisionError, match='zero constant part'):
        d.complex_zero().unit()


def test_from_tpsa_promotes_real_tpsas():
    d = madng_tpsa.Descriptor(1, 2)
    real = d.var(1, 1.0)
    imag = d.var(1, 2.0)

    promoted = madng_tpsa.ComplexTpsa.from_tpsa(real)
    combined = madng_tpsa.ComplexTpsa.from_tpsa(real, imag)

    assert promoted.real().equals(real)
    assert promoted.imag().is_zero()
    assert combined.real().equals(real)
    assert combined.imag().equals(imag)

    with pytest.raises(ValueError, match='Incompatible TPSA descriptors'):
        madng_tpsa.ComplexTpsa.from_tpsa(real, madng_tpsa.Descriptor(2, 2).var(1))


def test_complex_from_ptr_interns_and_infers_descriptor():
    d = madng_tpsa.Descriptor(1, 2)
    t = d.var(1, 0j)

    assert madng_tpsa.ComplexTpsa.from_ptr(t.ptr, d) is t

    ptr = lib.mad_ctpsa_newd(d.ptr, d.order)
    owned = madng_tpsa.ComplexTpsa.from_ptr(ptr, owns=True)

    assert owned.ptr == ptr
    assert owned.descriptor is d

    with pytest.raises(ValueError, match='A descriptor must be provided'):
        madng_tpsa.ComplexTpsa.from_ptr(ptr)


def test_complex_arithmetic_and_mixed_real_operands():
    d = madng_tpsa.Descriptor(1, 3)
    x = d.var(1, 1.0)
    z = d.var(1, 1 + 2j)
    assert isinstance(z, madng_tpsa.ComplexTpsa)

    assert (z + x).const_part == pytest.approx(2 + 2j)
    assert (x + z).const_part == pytest.approx(2 + 2j)
    assert (z - x).const_part == pytest.approx(2j)
    assert (x - z).const_part == pytest.approx(-2j)
    assert (z * x).const_part == pytest.approx(1 + 2j)
    assert (z / x).const_part == pytest.approx(1 + 2j)
    assert (x / z).const_part == pytest.approx((1 - 2j) / 5)
    assert (z * (2 - 3j)).const_part == pytest.approx(8 + 1j)
    assert (z / (2 - 3j)).const_part == pytest.approx((-4 + 7j) / 13)
    assert ((2 - 3j) / z).const_part == pytest.approx((-4 - 7j) / 5)


def test_complex_power_conjugate_and_real_imaginary_parts():
    d = madng_tpsa.Descriptor(1, 2)
    z = d.var(1, 1 + 2j)
    assert isinstance(z, madng_tpsa.ComplexTpsa)

    assert (z**2).const_part == pytest.approx(-3 + 4j)
    assert z.conjugate().const_part == pytest.approx(1 - 2j)
    assert z.real().const_part == pytest.approx(1)
    assert z.imag().const_part == pytest.approx(2)
    assert z.real().grad() == pytest.approx([1])
    assert z.imag().grad() == pytest.approx([0])


def test_complex_differential_algebra():
    d = madng_tpsa.Descriptor(variables=['x', 'y'], order=4)
    x, y = d.vars([0j, 0j])
    assert isinstance(x, madng_tpsa.ComplexTpsa)
    assert isinstance(y, madng_tpsa.ComplexTpsa)
    f = (1 + 2j) * x * x * y

    assert f.derivative('x').get((1, 1)) == pytest.approx(2 + 4j)
    assert f.derivative((2, 1)).const_part == pytest.approx(2 + 4j)
    assert f.integrate('x').get((3, 1)) == pytest.approx((1 + 2j) / 3)
    assert x.poisson_bracket(y).const_part == pytest.approx(1)


def test_complex_derivative_resolves_variable_and_parameter_specifications():
    d = madng_tpsa.Descriptor(variables=['x'], order=3, params=['k'], param_order=3)
    x = d.var('x', 0j)
    k = d.param('k', 0j)
    real_x = d.var('x')
    real_k = d.param('k')
    f = x * x * k

    for specification in (1, 'x', real_x):
        assert f.derivative(specification).equals(f.derivative(real_x))
    for specification in (2, 'k', real_k):
        assert f.derivative(specification).equals(f.derivative(real_k))
    assert f.derivative((1, 1)).get((1, 0)) == pytest.approx(2)
    assert f.derivative(real_x * real_x).equals(f.derivative((2, 0)))

    with pytest.raises(ValueError, match='coefficient 1'):
        f.derivative(2.0 * real_x)
    with pytest.raises(ValueError, match='identity variable'):
        f.integrate(real_x * real_x)


def test_complex_integrate_resolves_variable_specifications():
    d = madng_tpsa.Descriptor(variables=['x'], order=3, params=['k'], param_order=3)
    x = d.var('x', 0j)
    real_x = d.var('x')
    f = x * x

    for specification in (1, 'x', real_x):
        assert f.integrate(specification).equals(f.integrate(real_x))


@pytest.mark.xfail(reason='MAD-NG integration does not support parameters')
def test_complex_integrate_resolves_parameter_specifications():
    d = madng_tpsa.Descriptor(variables=['x'], order=3, params=['k'], param_order=3)
    x = d.var('x', 0j)
    k = d.param('k', 0j)
    real_k = d.param('k')
    f = x * x + k

    for specification in (2, 'k', real_k):
        assert f.integrate(specification).equals(f.integrate(real_k))


def test_complex_poisson_bracket_with_explicit_pair_count():
    d = madng_tpsa.Descriptor(4, 3)
    x, px, y, py = d.vars([0j, 0j, 0j, 0j])
    assert isinstance(x, madng_tpsa.ComplexTpsa)
    assert isinstance(px, madng_tpsa.ComplexTpsa)
    assert isinstance(y, madng_tpsa.ComplexTpsa)
    assert isinstance(py, madng_tpsa.ComplexTpsa)

    assert x.poisson_bracket(px, num_pairs=1).const_part == pytest.approx(1)
    assert y.poisson_bracket(py, num_pairs=1).is_zero()


def test_complex_elementary_functions_and_numpy_dispatch():
    d = madng_tpsa.Descriptor(1, 2)
    z = d.constant(1 + 2j)

    assert z.exp().const_part == pytest.approx(np.exp(1 + 2j))
    assert z.log().const_part == pytest.approx(np.log(1 + 2j))
    sin_z = cast('madng_tpsa.ComplexTpsa', np.sin(z))
    conjugate_z = cast('madng_tpsa.ComplexTpsa', np.conjugate(z))
    assert sin_z.const_part == pytest.approx(np.sin(1 + 2j))
    assert conjugate_z.const_part == pytest.approx(1 - 2j)
    assert cast('Any', scipy.special.erf)(z).const_part == pytest.approx(
        cast('Any', scipy.special.erf)(1 + 2j)
    )


def test_numpy_scalar_left_binary_operations_with_complex_tpsa():
    z = madng_tpsa.Descriptor(1, 3).constant(1 + 2j)
    scalar = cast('Any', np.float64(3.0))

    assert (scalar + z).const_part == pytest.approx(4 + 2j)
    assert (scalar - z).const_part == pytest.approx(2 - 2j)
    assert (scalar * z).const_part == pytest.approx(3 + 6j)
    assert (scalar / z).const_part == pytest.approx(0.6 - 1.2j)
    assert (scalar**z).const_part == pytest.approx(3.0 ** (1 + 2j))


@pytest.mark.parametrize(
    ('method_name', 'function_name'),
    [
        ('sqrt', 'sqrt'),
        ('sin', 'sin'),
        ('cos', 'cos'),
        ('tan', 'tan'),
        ('sinh', 'sinh'),
        ('cosh', 'cosh'),
        ('tanh', 'tanh'),
        ('asin', 'arcsin'),
        ('acos', 'arccos'),
        ('atan', 'arctan'),
        ('asinh', 'arcsinh'),
        ('acosh', 'arccosh'),
        ('atanh', 'arctanh'),
    ],
)
def test_complex_sqrt_and_trigonometric_functions(method_name, function_name):
    value = 0.5 + 0.25j
    t = madng_tpsa.Descriptor(1, 2).constant(value)

    result = getattr(t, method_name)()
    expected = getattr(np, function_name)(value)

    assert result.const_part == pytest.approx(expected)


@pytest.mark.parametrize('method_name', ('erf', 'erfc', 'erfcx', 'erfi', 'wofz'))
def test_complex_error_functions(method_name):
    value = 0.5 + 0.25j
    t = madng_tpsa.Descriptor(1, 2).constant(value)

    result = getattr(t, method_name)()
    expected = getattr(scipy.special, method_name)(value)

    assert result.const_part == pytest.approx(expected)


def test_complex_operations_reject_incompatible_descriptors():
    x = madng_tpsa.Descriptor(1, 2).var(1, 0j)
    y = madng_tpsa.Descriptor(2, 2).var(1, 0j)

    with pytest.raises(ValueError, match='Incompatible TPSA descriptors'):
        x + y


def test_tpsa_arithmetic_dunder_dispatches_are_exhaustive():
    d = madng_tpsa.Descriptor(1, 2)
    t = d.var(1, 2.0)
    other_t = d.constant(3.0)

    for method_name in ('__add__', '__sub__', '__mul__', '__truediv__'):
        method = getattr(t, method_name)
        assert isinstance(method(other_t), madng_tpsa.Tpsa)
        assert isinstance(method(Decimal('2')), madng_tpsa.Tpsa)
        assert isinstance(method(1j), madng_tpsa.ComplexTpsa)
        assert method(object()) is NotImplemented

    for method_name in ('__rsub__', '__rtruediv__'):
        method = getattr(t, method_name)
        assert isinstance(method(Decimal('2')), madng_tpsa.Tpsa)
        assert isinstance(method(1j), madng_tpsa.ComplexTpsa)
        assert method(object()) is NotImplemented

    assert isinstance(t.__pow__(other_t), madng_tpsa.Tpsa)
    assert isinstance(t.__pow__(2), madng_tpsa.Tpsa)
    assert isinstance(t.__pow__(Decimal('0.5')), madng_tpsa.Tpsa)
    assert isinstance(t.__pow__(1j), madng_tpsa.ComplexTpsa)
    assert t.__pow__(cast('Any', object())) is NotImplemented
    assert isinstance(t.__rpow__(Decimal('2')), madng_tpsa.Tpsa)
    assert isinstance(t.__rpow__(1j), madng_tpsa.ComplexTpsa)
    assert t.__rpow__(cast('Any', object())) is NotImplemented
    assert isinstance(+t, madng_tpsa.Tpsa)
    assert isinstance(-t, madng_tpsa.Tpsa)
    assert isinstance(abs(t), madng_tpsa.Tpsa)
    assert repr(t).startswith('Tpsa(')
    assert t == t.copy()
    assert t != object()
    assert t.__lt__(Decimal('3'))
    assert t.__le__(Decimal('2'))
    assert t.__gt__(Decimal('1'))
    assert t.__ge__(Decimal('2'))
    assert t.__array_ufunc__(np.add, 'reduce', t) is NotImplemented
    assert t.__array_ufunc__(np.floor, '__call__', t) is NotImplemented
    assert isinstance(np.sinc(cast('Any', t)), madng_tpsa.Tpsa)
    assert t.__array_function__(np.cos, (), (), {}) is NotImplemented

    with pytest.raises(ZeroDivisionError):
        t.__truediv__(0)
    with pytest.raises(ZeroDivisionError):
        d.zero().__rtruediv__(1)
    with pytest.raises(TypeError):
        float(cast('Any', t))
    with pytest.raises(ValueError, match='Incompatible TPSA descriptors'):
        t.__pow__(madng_tpsa.Descriptor(1, 3).constant(2.0))


def test_complex_tpsa_arithmetic_dunder_dispatches_are_exhaustive():
    d = madng_tpsa.Descriptor(1, 2)
    t = d.var(1, 2 + 1j)
    other_t = d.var(1, 3.0)
    other_c = d.constant(3 + 2j)
    assert isinstance(t, madng_tpsa.ComplexTpsa)
    assert isinstance(other_c, madng_tpsa.ComplexTpsa)

    for method_name in ('__add__', '__sub__', '__mul__', '__truediv__'):
        method = getattr(t, method_name)
        assert isinstance(method(other_c), madng_tpsa.ComplexTpsa)
        assert isinstance(method(other_t), madng_tpsa.ComplexTpsa)
        assert isinstance(method(Decimal('2')), madng_tpsa.ComplexTpsa)
        assert method(object()) is NotImplemented

    for method_name in ('__rsub__', '__rtruediv__'):
        method = getattr(t, method_name)
        assert isinstance(method(other_t), madng_tpsa.ComplexTpsa)
        assert isinstance(method(Decimal('2')), madng_tpsa.ComplexTpsa)
        assert method(object()) is NotImplemented

    assert isinstance(t.__pow__(other_c), madng_tpsa.ComplexTpsa)
    assert isinstance(t.__pow__(other_t), madng_tpsa.ComplexTpsa)
    assert isinstance(t.__pow__(2), madng_tpsa.ComplexTpsa)
    assert isinstance(t.__pow__(Decimal('0.5')), madng_tpsa.ComplexTpsa)
    assert t.__pow__(cast('Any', object())) is NotImplemented
    assert isinstance(t.__rpow__(other_t), madng_tpsa.ComplexTpsa)
    assert isinstance(t.__rpow__(Decimal('2')), madng_tpsa.ComplexTpsa)
    assert t.__rpow__(cast('Any', object())) is NotImplemented
    assert isinstance(+t, madng_tpsa.ComplexTpsa)
    assert isinstance(-t, madng_tpsa.ComplexTpsa)
    assert repr(t).startswith('ComplexTpsa(')
    assert t == t.copy()
    assert t != object()
    assert t.__array_ufunc__(np.add, 'reduce', t) is NotImplemented
    assert t.__array_ufunc__(np.floor, '__call__', t) is NotImplemented

    with pytest.raises(ZeroDivisionError):
        t.__truediv__(0)
    with pytest.raises(ZeroDivisionError):
        d.complex_zero().__rtruediv__(1)
    with pytest.raises(TypeError):
        complex(cast('Any', t))
