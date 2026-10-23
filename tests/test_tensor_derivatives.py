"""Mixed partial and arbitrary-rank derivative tensor regression tests."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from madng_tpsa import ComplexTpsa, Descriptor, TpsaMap, TpsaTensor


def _map() -> TpsaMap:
    d = Descriptor(
        variables=('der_q', 'der_p'),
        parameters=('der_k',),
        order=5,
        param_order=3,
    )
    q, p = d.vars()
    k = d.param('der_k')
    return TpsaMap(
        [
            q**3 + 2 * q * p + k * q**2 + 3 * k * p + 5 * k**2,
            q * p**2 + k * q * p + 2 * k**2 * p + 4 * q,
        ]
    )


def _tensor() -> TpsaTensor:
    d = Descriptor(variables=('ten_q', 'ten_p'), parameters=('ten_k',), order=5, param_order=3)
    q, p = d.vars()
    k = d.param('ten_k')
    return TpsaTensor([[q**2 * k, q * p], [k * p**2, q + k]])


def test_partial_derivative_preserves_map_type_shape_and_values() -> None:
    m = _map()
    result = m.partial_derivative((('der_q', 2), 'der_k'))
    assert isinstance(result, TpsaMap)
    assert result.shape == m.shape
    assert result.component_names == m.component_names
    np.testing.assert_allclose(result.evaluate((2, 3), parameters=(4,)), [2, 0])
    np.testing.assert_allclose(
        m.partial_derivative((2, 0, 1)).evaluate((2, 3), parameters=(4,)),
        [2, 0],
    )
    np.testing.assert_allclose(m.evaluate((2, 3), parameters=(4,)), [152, 146])


def test_partial_derivative_accepts_labels_indices_repetition_and_identity_tpsa() -> None:
    m = _map()
    q = m.descriptor.var('der_q')
    k = m.descriptor.param('der_k')
    for spec in (1, 'der_q', q, ('der_q',)):
        np.testing.assert_allclose(
            m.partial_derivative(spec).evaluate((2, 3), parameters=(4,)),
            [34, 25],
        )
    for spec in (3, 'der_k', k, (('der_k', 1),)):
        np.testing.assert_allclose(
            m.partial_derivative(spec).evaluate((2, 3), parameters=(4,)),
            [53, 54],
        )
    for spec in ((('der_q', 2), 'der_k'), ((1, 2), 3), (2, 0, 1)):
        np.testing.assert_allclose(
            m.partial_derivative(spec).evaluate((2, 3), parameters=(4,)),
            [2, 0],
        )


def test_partial_derivative_preserves_rank_two_tensor_and_view() -> None:
    t = _tensor()
    result = t.partial_derivative((('ten_q', 2), 'ten_k'))
    assert type(result) is TpsaTensor
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result.evaluate((2, 3), parameters=(4,)), [[2, 0], [0, 0]])

    view = t[0]
    assert isinstance(view, TpsaTensor)
    assert view.shape == (2,)
    out = view.partial_derivative('ten_q')
    assert out.shape == (2,)
    assert view.derivative_tensor(2).shape == (2, 2, 2)
    np.testing.assert_allclose(out.evaluate((2, 3), parameters=(4,)), [16, 3])


def test_derivative_tensor_appends_axes_independent_of_original_rank() -> None:
    t = _tensor()
    d1 = t.derivative_tensor(1)
    assert d1.shape == (2, 2, 2)
    d2 = t.derivative_tensor(2, wrt='all')
    assert d2.shape == (2, 2, 3, 3)
    at = d2.evaluate((2, 3), parameters=(4,))
    assert at[0, 0, 0, 0] == pytest.approx(8)  # d^2(q^2*k)/dq^2
    assert at[0, 0, 0, 2] == pytest.approx(4)  # d^2(q^2*k)/dq dk
    assert at[0, 0, 2, 0] == pytest.approx(4)
    assert at[1, 0, 1, 2] == pytest.approx(6)  # d^2(k*p^2)/dp dk
    assert at[1, 0, 2, 1] == pytest.approx(6)
    assert t.derivative_tensor(2, wrt='parameters').shape == (2, 2, 1, 1)
    assert t.derivative_tensor(2).shape == (2, 2, 2, 2)
    assert isinstance(d2, TpsaTensor)


def test_hessian_wrapper_and_symmetry() -> None:
    m = _map()
    h = m.hessian()
    assert h.shape == (2, 2, 2)
    assert isinstance(h, TpsaTensor)
    expected = [[[20, 2], [2, 0]], [[0, 10], [10, 4]]]
    np.testing.assert_allclose(h.evaluate((2, 3), parameters=(4,)), expected)
    np.testing.assert_allclose(m.evaluate_hessian((2, 3), parameters=(4,)), expected)
    np.testing.assert_allclose(m.evaluate_hessian(), [[[0, 2], [2, 0]], [[0, 0], [0, 0]]])
    assert h[0, 0, 1] == h[0, 1, 0]
    assert m.derivative_tensor(2).shape == h.shape


def test_full_parameter_hessian_and_parameter_only_selection() -> None:
    m = _map()
    expected = [
        [[20, 2, 4], [2, 0, 3], [4, 3, 10]],
        [[0, 10, 3], [10, 4, 18], [3, 18, 12]],
    ]
    h = m.hessian(wrt='all')
    assert h.shape == (2, 3, 3)
    np.testing.assert_allclose(h.evaluate((2, 3), parameters=(4,)), expected)
    np.testing.assert_allclose(m.evaluate_hessian((2, 3), parameters=(4,), wrt='all'), expected)
    only_params = m.derivative_tensor(2, wrt='parameters')
    assert only_params.shape == (2, 1, 1)
    np.testing.assert_allclose(only_params.evaluate((2, 3), parameters=(4,)), [[[10]], [[12]]])


def test_jacobian_operator_matches_numerical_jacobian_and_parameters() -> None:
    m = _map()
    assert m.jacobian().shape == (2, 2)
    np.testing.assert_allclose(m.evaluate_jacobian((2, 3), parameters=(4,)), [[34, 16], [25, 52]])
    np.testing.assert_allclose(
        m.derivative_tensor(1).evaluate((2, 3), parameters=(4,)),
        [[34, 16], [25, 52]],
    )
    np.testing.assert_allclose(
        m.derivative_tensor(1, wrt='parameters').evaluate((2, 3), parameters=(4,)),
        [[53], [54]],
    )


def test_third_derivative_tensor_contains_all_permutations() -> None:
    m = _map()
    third = m.derivative_tensor(3, wrt='all')
    assert third.shape == (2, 3, 3, 3)
    data = third.evaluate((2, 3), parameters=(4,))
    assert data[0, 0, 0, 0] == pytest.approx(6)
    assert data[0, 0, 0, 2] == pytest.approx(2)
    assert data[0, 0, 2, 0] == pytest.approx(2)
    assert data[0, 2, 0, 0] == pytest.approx(2)
    assert data[1, 0, 1, 1] == pytest.approx(2)
    assert data[1, 1, 2, 2] == pytest.approx(4)


def test_complex_derivatives_and_no_mutation() -> None:
    d = Descriptor(variables=('cx', 'cp'), parameters=('ck',), order=4, param_order=2)
    q, p = d.vars()
    k = d.param('ck')
    m = TpsaMap(((1 + 2j) * q**2 + (2 - 1j) * k * q * p, 1j * p**2 + k**2 * q))
    h = m.derivative_tensor(2, wrt='all')
    assert h.is_complex
    assert all(isinstance(x, ComplexTpsa) for x in h.flat)
    evaluated = h.evaluate((2, 3), parameters=(4,))
    assert evaluated[0, 0, 0] == pytest.approx(2 + 4j)
    assert evaluated[0, 0, 1] == pytest.approx(8 - 4j)
    assert evaluated[0, 0, 2] == pytest.approx(6 - 3j)
    assert evaluated[1, 1, 1] == pytest.approx(2j)
    assert evaluated[1, 0, 2] == pytest.approx(8)
    assert evaluated[1, 2, 2] == pytest.approx(4)
    before = m.evaluate((2, 3), parameters=(4,))
    derived = m.partial_derivative('cx')
    first = derived.flat[0]
    assert isinstance(first, ComplexTpsa)
    first.const_part = 123j
    np.testing.assert_allclose(m.evaluate((2, 3), parameters=(4,)), before)


def test_order_beyond_descriptor_returns_independent_zero_components() -> None:
    d = Descriptor(variables=('hq', 'hp'), order=3)
    q, p = d.vars()
    m = TpsaMap((q**3, p))
    fourth = m.derivative_tensor(4)
    assert fourth.shape == (2, 2, 2, 2, 2)
    np.testing.assert_array_equal(fourth.evaluate((2, 3)), np.zeros(fourth.shape))
    zero_map = m.partial_derivative((4, 0))
    np.testing.assert_array_equal(zero_map.evaluate((2, 3)), (0, 0))
    assert zero_map.flat[0] is not m.flat[0]


@pytest.mark.parametrize(
    'spec, error',
    [
        ('not_in_descriptor', KeyError),
        (4, ValueError),
        (0, ValueError),
        ((1, 2), ValueError),  # wrong number of exponent entries
        ((-1, 0, 1), ValueError),
        ((True, 0, 1), TypeError),
        ((0, 0, 0), ValueError),
        ((('der_q', 0),), ValueError),
        ((('der_q', -1),), ValueError),
        ((('der_q', 1.5),), TypeError),
        (True, TypeError),
        ((('missing', 2),), KeyError),
        (((5, 2),), ValueError),
    ],
)
def test_partial_rejects_invalid_specifications(spec: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        _map().partial_derivative(spec)


@pytest.mark.parametrize('order', [0, -1, True, cast('Any', 1.5)])
def test_derivative_tensor_rejects_invalid_order(order: int) -> None:
    with pytest.raises(ValueError, match='positive integer'):
        _map().derivative_tensor(order)


def test_derivative_tensor_rejects_unselected_or_unavailable_axes() -> None:
    d = Descriptor(variables=('np_q', 'np_p'), order=3)
    with pytest.raises(ValueError, match='Select at least one'):
        TpsaMap(d).derivative_tensor(1, wrt='parameters')


def test_mixed_parameter_derivative_tensor_is_symmetric_but_not_aliased() -> None:
    d = Descriptor(
        variables=('review_q2', 'review_p2'),
        parameters=('review_k2',),
        order=5,
        param_order=2,
    )
    q, p = d.vars()
    k = d.param('review_k2')
    matrix = TpsaTensor([[q * k**2, p], [q**2 * k, p * k]])
    hessian = matrix.derivative_tensor(2, wrt='all')
    assert hessian.shape == (2, 2, 3, 3)
    values = hessian.evaluate((2, 3), parameters=(4,))
    assert values[0, 0, 0, 2] == 8
    assert values[0, 0, 2, 0] == 8
    assert values[0, 0, 2, 2] == 4
    assert values[0, 0, 0, 2] == values[0, 0, 2, 0]

    # Symmetry is mathematical, but mutable TPSA components must be independent.
    first = hessian[0, 0, 0, 2]
    second = hessian[0, 0, 2, 0]
    assert first is not second
    before = second.const_part
    first.const_part = 123.0
    assert second.const_part == before
    np.testing.assert_allclose(matrix.evaluate((2, 3), parameters=(4,)), [[32, 3], [16, 12]])


def test_tensor_partial_and_full_derivative_consistency() -> None:
    d = Descriptor(variables=('review_q3', 'review_p3'), order=5)
    q, p = d.vars()
    m = TpsaMap([q**3 + q * p**2, q * p**3])
    hessian = m.hessian()
    selected = m.partial_derivative((1, 1))
    for i, component in enumerate(selected.components):
        assert component == hessian[i, 0, 1]


def test_real_parameter_multiindex_matches_successive_derivatives() -> None:
    d = Descriptor(
        variables=('multi_q', 'multi_p'),
        parameters=('multi_k',),
        order=5,
        param_order=3,
    )
    q, p = d.vars()
    k = d.param('multi_k')
    f = 2 * k**2 * p + 3 * k**2 * q + 5 * k**3

    assert f.derivative((0, 1, 1)) == 4 * k
    assert f.derivative((0, 0, 2)) == 4 * p + 6 * q + 30 * k
    assert f.derivative((0, 0, 3)) == d.constant(30)
    assert f.derivative((0, 0, 2)) == f.derivative('multi_k').derivative('multi_k')
    assert f.derivative((0, 1, 1)) == f.derivative('multi_p').derivative('multi_k')


def test_complex_parameter_multiindex_matches_successive_derivatives() -> None:
    d = Descriptor(
        variables=('cmulti_q', 'cmulti_p'),
        parameters=('cmulti_k',),
        order=5,
        param_order=3,
    )
    q, p = d.vars()
    k = d.param('cmulti_k')
    f = (1 + 2j) * k**2 * q + (2 - 1j) * k * q * p + (3 - 4j) * k**3

    assert f.derivative((1, 0, 1)) == (2 + 4j) * k + (2 - 1j) * p
    assert f.derivative((0, 0, 2)) == (2 + 4j) * q + (18 - 24j) * k
    assert f.derivative((0, 0, 2)) == f.derivative('cmulti_k').derivative('cmulti_k')


def test_mixed_multiple_parameters() -> None:
    d = Descriptor(
        variables=('two_q', 'two_p'),
        parameters=('two_k', 'two_l'),
        order=5,
        param_order=3,
    )
    q, _ = d.vars()
    k = d.param('two_k')
    ell = d.param('two_l')
    f = (2 + 1j) * k**2 * ell * q

    assert f.derivative((0, 0, 2, 1)) == (4 + 2j) * q
    assert f.derivative((0, 0, 2, 1)) == (
        f.derivative('two_k').derivative('two_k').derivative('two_l')
    )


def test_tensor_parameter_hessian_inherits_scalar_fix() -> None:
    d = Descriptor(
        variables=('hpar_q', 'hpar_p'),
        parameters=('hpar_k',),
        order=5,
        param_order=3,
    )
    _, p = d.vars()
    k = d.param('hpar_k')
    m = TpsaMap([5 * k**2, 2 * k**2 * p])
    h = m.derivative_tensor(2, wrt='parameters')
    assert h.shape == (2, 1, 1)
    np.testing.assert_allclose(
        h.evaluate((2, 3), parameters=(4,)),
        [[[10]], [[12]]],
    )
    np.testing.assert_allclose(
        m.partial_derivative((0, 0, 2)).evaluate((2, 3), parameters=(4,)),
        [10, 12],
    )


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
