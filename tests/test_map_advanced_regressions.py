"""Regression tests for descriptors, formal inversion, and canonical ordering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from madng_tpsa import Descriptor, TpsaMap

if TYPE_CHECKING:
    from madng_tpsa import TpsaTensor


def _assert_coefficients_small(mapping: TpsaTensor, degree: int = 3, atol: float = 1e-9) -> None:
    for component in mapping.flat:
        for monomial, coefficient in component.monomial_coeffs(tol=0).items():
            if sum(monomial) <= degree:
                assert abs(coefficient) < atol, (monomial, coefficient)


def test_inverse_with_parameter_only_terms_has_formal_identities() -> None:
    d = Descriptor(variables=('inv_q', 'inv_p'), parameters=('inv_k',), order=6, param_order=3)
    q, p = d.vars()
    k = d.param('inv_k')
    mapping = TpsaMap([q + k + 0.1 * p**2, p + 0.02 * k * q])
    assert np.all(mapping.const_part == 0)
    inverse = mapping.inverse()
    identity = TpsaMap(d)
    _assert_coefficients_small(mapping.compose(inverse) - identity)
    _assert_coefficients_small(inverse.compose(mapping) - identity)
    for values, param in [((0.01, 0.02), 0.03), ((-0.02, 0.04), -0.02)]:
        np.testing.assert_allclose(
            mapping.compose(inverse).evaluate(values, parameters=(param,)),
            values,
            atol=1e-9,
        )


def test_partial_inverse_mixed_selected_and_unselected_variables() -> None:
    d = Descriptor(
        variables=('part_q', 'part_p', 'part_y', 'part_py'),
        parameters=('part_k',),
        order=5,
        param_order=2,
    )
    q, p, y, py = d.vars()
    k = d.param('part_k')
    mapping = TpsaMap([q + p**2 + k, p, y + py**2 + 0.02 * k, py])
    selected = [True, False, True, False]
    partial = mapping.partial_inverse(selected)
    selection_map = TpsaMap([mapping.part_q, p, mapping.part_y, py])
    identity = TpsaMap(d)
    _assert_coefficients_small(selection_map.compose(partial) - identity)
    _assert_coefficients_small(partial.compose(selection_map) - identity)


def test_canonical_labels_are_positional_not_inferred() -> None:
    canonical = Descriptor(variables=('physical_q', 'physical_p'), order=4)
    reversed_order = Descriptor(variables=('physical_p', 'physical_q'), order=4)
    q, p = canonical.vars()
    p_rev, q_rev = reversed_order.vars()
    assert q.poisson_bracket(p).const_part == 1
    assert q_rev.poisson_bracket(p_rev).const_part == -1
    # Core algebra deliberately does NOT validate names or guess conjugacy.
    assert TpsaMap.from_generator(q**2 / 2).physical_p == q
    assert TpsaMap.from_generator(q_rev**2 / 2).physical_p == -q_rev


def test_complex_descriptor_relabelling_with_reordered_variables_and_two_parameters() -> None:
    src = Descriptor(
        variables=('old_q', 'old_p'), parameters=('old_a', 'old_b'), order=5, param_order=2
    )
    q, p = src.vars()
    a, b = src.params()
    mapping = TpsaMap([q + (1 + 2j) * a * p + b**2, p - 1j * b * q])
    dst = Descriptor(
        variables=('new_p', 'new_q'), parameters=('new_b', 'new_a'), order=6, param_order=3
    )
    converted = mapping.change_descriptor(
        dst,
        variable_map={'old_q': 'new_q', 'old_p': 'new_p'},
        parameter_map={'old_a': 'new_a', 'old_b': 'new_b'},
    )
    assert converted.is_complex
    before = mapping.evaluate((0.2, 0.3), parameters=(0.4, 0.5))
    after = converted.evaluate((0.3, 0.2), parameters=(0.5, 0.4))
    np.testing.assert_allclose(after, before[::-1])
    with pytest.raises(ValueError):
        mapping.change_descriptor(
            dst,
            variable_map={'old_q': 'new_p', 'old_p': 'new_p'},
            parameter_map={'old_a': 'new_a', 'old_b': 'new_b'},
        )
    with pytest.raises(ValueError, match='same target'):
        mapping.change_descriptor(
            dst,
            variable_map={'old_q': 'new_q', 'old_p': 'new_p'},
            parameter_map={'old_a': 'new_a', 'old_b': 'new_a'},
        )


def test_complex_descriptor_rejects_unsafe_downwards_order_conversion() -> None:
    src = Descriptor(variables=('ord_q', 'ord_p'), order=5)
    q, p = src.vars()
    mapping = TpsaMap([q + 1j * q**4, p])
    dst = Descriptor(variables=('ord_q', 'ord_p'), order=2)
    with pytest.raises((ValueError, TypeError)):
        mapping.change_descriptor(dst)
    truncated = mapping.change_descriptor(dst, truncate=True)
    assert truncated.is_complex
    assert truncated.ord_q.degree <= 2
