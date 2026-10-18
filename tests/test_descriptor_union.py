"""Tests for Descriptor.union().

Two known limitations of native-pointer-only descriptor interning are
recorded as strict xfails at the bottom of this module. Remove the marks
once labelled descriptors can share a native algebra without losing labels.
"""

import pytest

from madng_tpsa.descriptor import Descriptor


def test_union_is_idempotent():
    descriptor = Descriptor(variables=('r', 'pr'), order=10)

    assert descriptor.union(descriptor) is descriptor


def test_union_merges_variables_parameters_and_limits():
    left = Descriptor(
        variables=('x', 'px', 'y', 'py'),
        parameters=('k',),
        order=3,
        param_order=1,
    )
    right = Descriptor(
        variables=('x', 'px', 'zeta', 'delta'),
        parameters=('epsilon',),
        order=5,
        param_order=2,
    )

    left_before = left.to_dict()
    right_before = right.to_dict()
    merged = left.union(right)

    assert merged.variables == ('x', 'px', 'y', 'py', 'zeta', 'delta')
    assert merged.parameters == ('k', 'epsilon')
    assert merged.num_vars == 6
    assert merged.num_params == 2
    assert merged.monomial_length == 8
    assert merged.order == 5
    assert merged.param_order == 2
    assert merged.max_orders == (5, 5, 3, 3, 5, 5, 1, 2)

    # Check the resulting native descriptor, not just Python metadata.
    assert merged.is_valid_monomial((5, 0, 0, 0, 0, 0, 0, 0))
    assert not merged.is_valid_monomial((0, 0, 4, 0, 0, 0, 0, 0))
    assert merged.is_valid_monomial((0, 0, 0, 0, 0, 0, 1, 0))
    assert not merged.is_valid_monomial((0, 0, 0, 0, 0, 0, 2, 0))
    assert merged.is_valid_monomial((0, 0, 0, 0, 0, 0, 0, 2))
    assert not merged.is_valid_monomial((0, 0, 0, 0, 0, 0, 1, 2))

    # Union must not modify either operand.
    assert left.to_dict() == left_before
    assert right.to_dict() == right_before


def test_union_merges_nonuniform_limits_by_label():
    left = Descriptor(
        variables=('alpha', 'beta'),
        order=6,
        max_orders=(2, 4),
    )
    right = Descriptor(
        variables=('alpha', 'gamma'),
        order=7,
        max_orders=(5, 2),
    )

    merged = left.union(right)

    assert merged.variables == ('alpha', 'beta', 'gamma')
    assert merged.parameters == ()
    assert merged.order == 7
    assert merged.max_orders == (5, 4, 2)
    assert merged.is_valid_monomial((5, 0, 0))
    assert not merged.is_valid_monomial((6, 0, 0))
    assert not merged.is_valid_monomial((0, 0, 3))


def test_union_of_subset_reuses_superset_descriptor():
    superset = Descriptor(variables=('v', 'pv', 'w'), order=11)
    subset = Descriptor(variables=('v', 'pv'), order=3)

    merged = superset.union(subset)

    assert merged is superset
    assert merged.variables == ('v', 'pv', 'w')
    assert merged.max_orders == (11, 11, 11)


def test_union_rejects_variable_parameter_collisions():
    left = Descriptor(
        variables=('u', 'pu'),
        parameters=('k',),
        order=8,
        param_order=1,
    )
    right = Descriptor(
        variables=('u', 'k'),
        parameters=('h',),
        order=9,
        param_order=2,
    )

    with pytest.raises(ValueError, match='both variables and parameters'):
        left.union(right)


def test_descriptors_with_same_native_structure_retain_distinct_labels():
    left = Descriptor(variables=('first', 'second'), order=13)
    right = Descriptor(variables=('third', 'fourth'), order=13)

    assert right.variables == ('third', 'fourth')
    merged = left.union(right)
    assert merged.variables == ('first', 'second', 'third', 'fourth')


def test_union_respects_left_label_order_on_native_descriptor_collision():
    left = Descriptor(variables=('left', 'right'), order=14)
    right = Descriptor(variables=('right', 'left'), order=15)

    merged = left.union(right)

    assert merged.variables == ('left', 'right')
    assert merged.order == 15
