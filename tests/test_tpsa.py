"""Tests for TPSA series."""

import gc
import math
import weakref

import numpy as np
import pytest

import madng_tpsa
from madng_tpsa._cffi import lib


def test_var_is_an_identity_seed():
    d = madng_tpsa.Descriptor(6, 2)
    t = d.var(2, 0.25)  # variable indices are 1-based
    assert t.const_part == 0.25
    assert t.grad() == [0, 1, 0, 0, 0, 0]
    assert t.descriptor is d
    assert t.order == d.order
    assert t.max_nonzero_order == 1


def test_vars_unpack_all_identity_seeds():
    d = madng_tpsa.Descriptor(4, 2)
    x, px, y, py = d.vars()

    assert x.grad() == [1, 0, 0, 0]
    assert px.grad() == [0, 1, 0, 0]
    assert y.grad() == [0, 0, 1, 0]
    assert py.grad() == [0, 0, 0, 1]


def test_var_order_uses_descriptor_order_by_default():
    d = madng_tpsa.Descriptor(variables=['a', 'b'], order=4, max_orders=[2, 4])
    a, b = d.vars()

    assert a.order == 4
    assert b.order == 4
    assert a.max_nonzero_order == 1
    assert b.max_nonzero_order == 1


def test_var_zero_and_param_have_expected_orders():
    d = madng_tpsa.Descriptor(variables=['a', 'b'], order=4, max_orders=[2, 4, 1], params=['k'])
    zero = d.zero(order=2)
    a = d.var('a', order=2)
    k = d.param('k')

    assert zero.order == 2
    assert a.order == 2
    assert k.order == 1

    with pytest.raises(ValueError, match='Variable order must be positive'):
        d.var('a', order=0)


def test_var_and_param_labels_can_seed_series():
    d = madng_tpsa.Descriptor(variables=['x', 'y'], order=2, params=['k'])
    x = d.var('x', 0.5)
    y = d.var('y')
    k = d.param('k', 2.0)

    assert x.const_part == 0.5
    assert x.grad() == [1.0, 0.0]
    assert y.grad() == [0.0, 1.0]
    assert k.const_part == 2.0
    assert k.param_grad() == [1.0]


def test_vars_can_set_expansion_values():
    d = madng_tpsa.Descriptor(2, 2)
    x, y = d.vars(values=[0.5, 2.0])

    assert x.const_part == 0.5
    assert y.const_part == 2.0
    assert x.grad() == [1, 0]
    assert y.grad() == [0, 1]


def test_vars_values_must_match_num_vars():
    d = madng_tpsa.Descriptor(2, 2)

    with pytest.raises(ValueError, match='Values must contain one entry per variable'):
        d.vars(values=[0.5])


def test_constant_series():
    d = madng_tpsa.Descriptor(3, 2)
    zero = d.zero()
    const = d.constant(1.5)

    assert zero.monomial_coeffs() == {}
    assert zero.is_zero()
    assert zero.is_constant()
    assert zero.const_part == 0.0
    assert zero.order == d.order
    assert zero.max_nonzero_order == 0
    assert zero.grad() == [0.0] * d.num_vars
    assert const.const_part == 1.5
    assert const.order == d.order
    assert const.max_nonzero_order == 0
    assert not const.is_zero()
    assert const.is_constant()
    assert const.grad() == [0.0] * d.num_vars


def test_arithmetic_carries_derivatives():
    d = madng_tpsa.Descriptor(2, 3)
    x = d.var(1, 0.5)
    y = d.var(2, 2.0)

    f = x * x + 2.0 * y
    assert f.const_part == pytest.approx(0.25 + 4.0)
    assert f.order == d.order
    assert f.max_nonzero_order == 2
    assert f.grad() == pytest.approx([2 * 0.5, 2.0])
    assert f.monomial_coeffs()[(2, 0)] == pytest.approx(1.0)

    # scalars mix on either side, and the reflected ops are not the mirrored ones
    assert (3.0 - x).const_part == pytest.approx(2.5)
    assert (1.0 / y).const_part == pytest.approx(0.5)
    assert (-x).const_part == pytest.approx(-0.5)
    assert (x / 2.0).const_part == pytest.approx(0.25)
    assert (x**2).const_part == pytest.approx(0.25)


def test_addition_paths():
    d = madng_tpsa.Descriptor(2, 2)
    x = d.var(1, 0.5)
    y = d.var(2, 2.0)

    xy = x + y
    xs = x + 3.0
    sx = 3.0 + x

    assert xy.const_part == pytest.approx(2.5)
    assert xy.grad() == pytest.approx([1.0, 1.0])
    assert xs.const_part == pytest.approx(3.5)
    assert xs.grad() == pytest.approx([1.0, 0.0])
    assert sx.const_part == pytest.approx(3.5)
    assert sx.grad() == pytest.approx([1.0, 0.0])


def test_subtraction_paths():
    d = madng_tpsa.Descriptor(2, 2)
    x = d.var(1, 0.5)
    y = d.var(2, 2.0)

    xy = x - y
    xs = x - 3.0
    sx = 3.0 - x

    assert xy.const_part == pytest.approx(-1.5)
    assert xy.grad() == pytest.approx([1.0, -1.0])
    assert xs.const_part == pytest.approx(-2.5)
    assert xs.grad() == pytest.approx([1.0, 0.0])
    assert sx.const_part == pytest.approx(2.5)
    assert sx.grad() == pytest.approx([-1.0, 0.0])


def test_multiplication_paths():
    d = madng_tpsa.Descriptor(2, 2)
    x = d.var(1, 0.5)
    y = d.var(2, 2.0)

    xy = x * y
    xs = x * 3.0
    sx = 3.0 * x

    assert xy.const_part == pytest.approx(1.0)
    assert xy.grad() == pytest.approx([2.0, 0.5])
    assert xy.get((1, 1)) == pytest.approx(1.0)
    assert xs.const_part == pytest.approx(1.5)
    assert xs.grad() == pytest.approx([3.0, 0.0])
    assert sx.const_part == pytest.approx(1.5)
    assert sx.grad() == pytest.approx([3.0, 0.0])


def test_division_paths():
    d = madng_tpsa.Descriptor(1, 2)
    x = d.var(1, 2.0)

    quotient = (x * x) / x
    scaled = x / 2.0
    inverse = 1.0 / x

    assert quotient.const_part == pytest.approx(2.0)
    assert quotient.grad() == pytest.approx([1.0])
    assert scaled.const_part == pytest.approx(1.0)
    assert scaled.grad() == pytest.approx([0.5])
    assert inverse.const_part == pytest.approx(0.5)
    assert inverse.grad() == pytest.approx([-0.25])
    assert inverse.get((2,)) == pytest.approx(0.125)


def test_power_and_negation():
    d = madng_tpsa.Descriptor(1, 3)
    x = d.var(1, 0.5)

    square = x**2
    cube = x**3
    neg = -x

    assert square.const_part == pytest.approx(0.25)
    assert square.grad() == pytest.approx([1.0])
    assert square.get((2,)) == pytest.approx(1.0)
    assert cube.const_part == pytest.approx(0.125)
    assert cube.grad() == pytest.approx([0.75])
    assert cube.get((2,)) == pytest.approx(1.5)
    assert cube.get((3,)) == pytest.approx(1.0)
    assert neg.const_part == pytest.approx(-0.5)
    assert neg.grad() == pytest.approx([-1.0])


def test_division_by_series():
    d = madng_tpsa.Descriptor(1, 2)
    x = d.var(1, 2.0)
    q = (x * x) / x
    assert q.const_part == pytest.approx(2.0)
    assert q.grad() == pytest.approx([1.0])


def test_binary_ops_reject_incompatible_descriptors():
    x = madng_tpsa.Descriptor(1, 2).var(1)
    y = madng_tpsa.Descriptor(2, 2).var(1)

    with pytest.raises(ValueError, match='Incompatible TPSA descriptors'):
        x + y


def test_equality_checks_coefficients_and_descriptor_compatibility():
    d = madng_tpsa.Descriptor(1, 2)
    x = d.var(1, 0.5)

    assert x == x.copy()
    assert x != x + 1.0
    assert x.equals(x + 1e-15, tol=1e-14)
    assert x != madng_tpsa.Descriptor(2, 2).var(1)


def test_copy_is_independent():
    d = madng_tpsa.Descriptor(1, 1)
    x = d.var(1, 1.0)
    c = x.copy()
    x.set_const_part(9.0)
    assert (c.const_part, x.const_part) == (1.0, 9.0)


def test_clear_sets_series_to_zero_in_place():
    d = madng_tpsa.Descriptor(2, 2)
    x, y = d.vars(values=[1.0, 2.0])
    t = x * y + 3.0

    assert not t.is_zero()
    assert not t.is_constant()

    t.clear()

    assert t.is_zero()
    assert t.is_constant()
    assert t.to_dict() == {}


def test_set_and_get_coefficients():
    d = madng_tpsa.Descriptor(2, 2)
    t = d.zero()
    t.set_const_part(1.5)
    t.set((1, 1), -2.0)
    t[(2, 0)] = 3.0
    assert t.const_part == 1.5
    assert t.get((1, 1)) == -2.0
    assert t[(2, 0)] == 3.0
    assert t.coefficient((1, 1)) == -2.0
    np.testing.assert_allclose(t.coefficient([(0, 0), (1, 1)]), [1.5, -2.0])
    with pytest.raises(ValueError):
        t.coefficient(np.zeros((1, 2, 2), dtype=int))


def test_get_and_set_reject_invalid_or_out_of_order_monomials():
    d = madng_tpsa.Descriptor(variables=['a', 'b'], order=4, max_orders=[2, 4])
    a = d.var('a', order=2)

    assert d.is_valid_monomial((1, 3))
    with pytest.raises(ValueError, match='Monomial order exceeds TPSA order 2'):
        a[1, 3]

    with pytest.raises(ValueError, match='Monomial order exceeds TPSA order 2'):
        a[1, 3] = 1.0

    with pytest.raises(ValueError, match='not valid for this descriptor'):
        a[3, 0]

    full_order_a = d.var('a')
    assert full_order_a[1, 3] == 0.0


def test_monomial_coeffs_skips_zeros_and_tiny_terms():
    d = madng_tpsa.Descriptor(2, 2)
    t = d.zero()
    t.set((1, 0), 1e-20)
    t.set((0, 1), 1.0)
    assert t.monomial_coeffs() == {(0, 1): 1.0}
    assert (1, 0) in t.monomial_coeffs(tol=1e-30)


def test_to_dict_uses_monomial_coeffs():
    d = madng_tpsa.Descriptor(2, 2)
    t = d.zero()
    t.set((1, 0), 1e-20)
    t.set((0, 1), 1.0)

    assert t.to_dict() == {(0, 1): 1.0}
    assert t.to_dict(tol=1e-30) == t.monomial_coeffs(tol=1e-30)


def test_from_dict_replaces_coefficients_and_round_trips():
    d = madng_tpsa.Descriptor(2, 2)
    original = d.zero()
    original.set_const_part(1.5)
    original.set((1, 0), 2.0)
    original.set((0, 2), -3.0)

    restored = d.var(1, 10.0)
    restored.from_dict(original.to_dict())
    assert restored == original

    restored.from_dict({(0, 1): 4.0})
    assert restored.to_dict() == {(0, 1): 4.0}
    assert restored.const_part == 0.0
    assert restored.get((1, 0)) == 0.0


def test_param_seed_and_param_grad():
    d = madng_tpsa.Descriptor(2, 2, num_params=2, param_order=1)
    x = d.var(1, 0.5)
    k = d.param(1, 3.0)

    assert k.const_part == 3.0
    assert k.param_grad() == [1.0, 0.0]
    assert k.grad() == [0.0, 0.0]  # a parameter is not a variable

    f = k * x
    assert f.grad() == pytest.approx([3.0, 0.0])  # d(kx)/dx = k
    assert f.get((1, 0, 1, 0)) == pytest.approx(1.0)  # mixed d2/dx dk


def test_params_unpack_all_identity_seeds():
    d = madng_tpsa.Descriptor(2, 2, num_params=2, param_order=1)
    k1, k2 = d.params()

    assert k1.param_grad() == [1, 0]
    assert k2.param_grad() == [0, 1]


def test_from_ptr_returns_same_object():
    d = madng_tpsa.Descriptor(2, 3)
    t1 = d.var(1)
    t2 = madng_tpsa.Tpsa.from_ptr(t1._ptr, d)

    assert t2 is t1


def test_from_ptr_infers_descriptor_for_a_new_owned_pointer():
    d = madng_tpsa.Descriptor(2, 3)
    ptr = lib.mad_tpsa_newd(d.ptr, d.order)

    t = madng_tpsa.Tpsa.from_ptr(ptr, owns=True)

    assert t.descriptor is d
    assert t.ptr == ptr


def test_from_ptr_raises_for_unknown_pointer():
    d = madng_tpsa.Descriptor(2, 3)
    t1 = d.var(1)
    ptr = t1._ptr

    del t1
    gc.collect()

    with pytest.raises(ValueError, match='No live Tpsa found'):
        madng_tpsa.Tpsa.from_ptr(ptr, d)


def test_from_ptr_does_not_double_free():
    d = madng_tpsa.Descriptor(2, 3)
    t1 = d.var(1)
    t1_ref = weakref.ref(t1)
    t2 = madng_tpsa.Tpsa.from_ptr(t1._ptr, d)

    del t2
    gc.collect()

    assert t1_ref() is t1
    assert t1.grad() == [1.0, 0.0]

    del t1
    gc.collect()

    assert t1_ref() is None


def test_coerce_operand_returns_compatible_tpsa_unchanged():
    d = madng_tpsa.Descriptor(1, 2)
    t = d.var(1)
    other = d.constant(2)

    assert t._coerce_operand(other) is other


@pytest.mark.parametrize(
    ('method_name', 'expected'),
    [
        ('sincos', lambda x: (math.sin(x), math.cos(x))),
        ('sincosq', lambda x: (math.sin(math.sqrt(x)) / math.sqrt(x), math.cos(math.sqrt(x)))),
        (
            'sincosmq',
            lambda x: (
                (math.sin(math.sqrt(x)) / math.sqrt(x) - 1) / x,
                (math.cos(math.sqrt(x)) - 1) / x,
            ),
        ),
        ('sincosh', lambda x: (math.sinh(x), math.cosh(x))),
        ('sincoshq', lambda x: (math.sinh(math.sqrt(x)) / math.sqrt(x), math.cosh(math.sqrt(x)))),
        (
            'sincoshmq',
            lambda x: (
                (math.sinh(math.sqrt(x)) / math.sqrt(x) - 1) / x,
                (math.cosh(math.sqrt(x)) - 1) / x,
            ),
        ),
    ],
)
def test_two_output_unary_operations(method_name, expected):
    value = 0.25
    t = madng_tpsa.Descriptor(1, 2).constant(value)

    first, second = getattr(t, method_name)()

    expected_first, expected_second = expected(value)
    assert first.const_part == pytest.approx(expected_first)
    assert second.const_part == pytest.approx(expected_second)
