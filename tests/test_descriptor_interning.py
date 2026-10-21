"""Regression tests for label-aware Descriptor interning and native ownership.

Run with: pytest -q tests/test_descriptor_interning.py

Do not attempt to dereference native pointers after their last Descriptor owner
has been released. These tests only use live descriptors.
"""

from __future__ import annotations

import gc
import weakref

import pytest

from madng_tpsa.descriptor import Descriptor


def test_same_algebra_different_variable_labels():
    x = Descriptor(variables=('inter_x', 'inter_px'), order=7)
    y = Descriptor(variables=('inter_y', 'inter_py'), order=7)

    assert x is not y
    assert x.ptr == y.ptr  # MAD-NG interns the numerical algebra
    assert x.variables == ('inter_x', 'inter_px')
    assert y.variables == ('inter_y', 'inter_py')
    assert x != y
    assert len({x, y}) == 2


def test_same_algebra_and_same_labels_is_same_python_object():
    x = Descriptor(variables=('intern_x', 'intern_px'), order=8)
    y = Descriptor(variables=('intern_x', 'intern_px'), order=8)

    assert x is y
    assert x == y
    assert hash(x) == hash(y)


def test_parameter_labels_participate_in_interning():
    first = Descriptor(
        variables=('a', 'b'),
        parameters=('knob_1',),
        order=5,
        param_order=1,
    )
    second = Descriptor(
        variables=('a', 'b'),
        parameters=('knob_2',),
        order=5,
        param_order=1,
    )

    assert first.ptr == second.ptr
    assert first is not second
    assert first != second
    assert first.parameters == ('knob_1',)
    assert second.parameters == ('knob_2',)


def test_from_ptr_with_labels_is_unambiguous():
    first = Descriptor(variables=('lab_x', 'lab_y'), order=9)
    second = Descriptor(variables=('lab_y', 'lab_x'), order=9)

    assert (
        Descriptor.from_ptr(first.ptr, variables=first.variables, parameters=first.parameters)
        is first
    )
    assert (
        Descriptor.from_ptr(second.ptr, variables=second.variables, parameters=second.parameters)
        is second
    )

    with pytest.raises(ValueError, match='[Aa]mbiguous'):
        Descriptor.from_ptr(first.ptr)


def test_from_ptr_single_live_variant():
    first = Descriptor(variables=('lone_x', 'lone_y'), order=10)
    assert Descriptor.from_ptr(first.ptr) is first


def test_native_descriptor_stays_alive_while_alias_is_alive():
    first = Descriptor(variables=('life_x', 'life_px'), order=11)
    second = Descriptor(variables=('life_y', 'life_py'), order=11)
    assert first.ptr == second.ptr
    wr = weakref.ref(first)

    del first
    gc.collect()
    assert wr() is None  # first labelled view was collected

    # The other view still owns the underlying C descriptor.
    assert second.num_vars == 2
    assert second.max_orders == (11, 11)
    assert second.var('life_y').descriptor is second


def test_from_dict_recovers_requested_labels():
    source = Descriptor(variables=('serial_a', 'serial_b'), order=15)
    other = Descriptor(variables=('serial_c', 'serial_d'), order=15)
    assert source.ptr == other.ptr

    loaded = Descriptor.from_dict(source.to_dict())
    assert loaded is source
    assert loaded.variables == ('serial_a', 'serial_b')


def test_construct_rejects_mismatched_variable_and_parameter_labels():
    with pytest.raises(ValueError, match='both variable and parameter'):
        Descriptor(
            variables=('reused_name',),
            parameters=('reused_name',),
            order=4,
        )


def test_tpsa_retains_original_labelled_descriptor():
    first = Descriptor(variables=('retained_x', 'retained_px'), order=16)
    second = Descriptor(variables=('different_x', 'different_px'), order=16)
    assert first.ptr == second.ptr

    tpsa = first.var('retained_x')
    assert tpsa.descriptor is first
    reference = weakref.ref(first)
    del first
    gc.collect()

    assert reference() is not None  # TPSA retains its labelled descriptor.
    assert tpsa.descriptor.variables == ('retained_x', 'retained_px')
    assert second.variables == ('different_x', 'different_px')
