"""Tests for Descriptor.union().

Two known limitations of native-pointer-only descriptor interning are
recorded as strict xfails at the bottom of this module. Remove the marks
once labelled descriptors can share a native algebra without losing labels.
"""

import pytest

from madng_tpsa.descriptor import Descriptor


def test_union_is_idempotent():
    descriptor = Descriptor(var_names=('r', 'pr'), order=10)

    assert descriptor.union(descriptor) is descriptor


def test_union_merges_variables_parameters_and_limits():
    left = Descriptor(
        var_names=('x', 'px', 'y', 'py'),
        param_names=('k',),
        order=3,
        param_order=1,
    )
    right = Descriptor(
        var_names=('x', 'px', 'zeta', 'delta'),
        param_names=('epsilon',),
        order=5,
        param_order=2,
    )

    left_before = left.to_dict()
    right_before = right.to_dict()
    merged = left.union(right)

    assert merged.var_names == ('x', 'px', 'y', 'py', 'zeta', 'delta')
    assert merged.param_names == ('k', 'epsilon')
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
        var_names=('alpha', 'beta'),
        order=6,
        max_orders=(2, 4),
    )
    right = Descriptor(
        var_names=('alpha', 'gamma'),
        order=7,
        max_orders=(5, 2),
    )

    merged = left.union(right)

    assert merged.var_names == ('alpha', 'beta', 'gamma')
    assert merged.param_names == ()
    assert merged.order == 7
    assert merged.max_orders == (5, 4, 2)
    assert merged.is_valid_monomial((5, 0, 0))
    assert not merged.is_valid_monomial((6, 0, 0))
    assert not merged.is_valid_monomial((0, 0, 3))


def test_union_of_subset_reuses_superset_descriptor():
    superset = Descriptor(var_names=('v', 'pv', 'w'), order=11)
    subset = Descriptor(var_names=('v', 'pv'), order=3)

    merged = superset.union(subset)

    assert merged is superset
    assert merged.var_names == ('v', 'pv', 'w')
    assert merged.max_orders == (11, 11, 11)


def test_union_rejects_variable_parameter_collisions():
    left = Descriptor(
        var_names=('u', 'pu'),
        param_names=('k',),
        order=8,
        param_order=1,
    )
    right = Descriptor(
        var_names=('u', 'k'),
        param_names=('h',),
        order=9,
        param_order=2,
    )

    with pytest.raises(ValueError, match='both variables and parameters'):
        left.union(right)


def test_descriptors_with_same_native_structure_retain_distinct_labels():
    left = Descriptor(var_names=('first', 'second'), order=13)
    right = Descriptor(var_names=('third', 'fourth'), order=13)

    assert right.var_names == ('third', 'fourth')
    merged = left.union(right)
    assert merged.var_names == ('first', 'second', 'third', 'fourth')


def test_union_respects_left_label_order_on_native_descriptor_collision():
    left = Descriptor(var_names=('left', 'right'), order=14)
    right = Descriptor(var_names=('right', 'left'), order=15)

    merged = left.union(right)

    assert merged.var_names == ('left', 'right')
    assert merged.order == 15


def test_union_of_distinct_label_sets_is_correct():
    x = Descriptor(var_names=('merge_x', 'merge_px'), order=12)
    y = Descriptor(var_names=('merge_y', 'merge_py'), order=12)

    result = x.union(y)
    assert result.var_names == ('merge_x', 'merge_px', 'merge_y', 'merge_py')
    assert result.order == 12
    assert x.var_names == ('merge_x', 'merge_px')
    assert y.var_names == ('merge_y', 'merge_py')


def test_union_preserves_left_hand_order_on_native_collision():
    x = Descriptor(var_names=('reorder_x', 'reorder_y'), order=13)
    y = Descriptor(var_names=('reorder_y', 'reorder_x'), order=14)
    merged = x.union(y)

    assert merged.var_names == ('reorder_x', 'reorder_y')
    assert merged.order == 14
