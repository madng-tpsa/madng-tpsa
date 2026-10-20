"""Core tests for TpsaTensor: shape, views, ownership, coefficients and conversion.

Place in tests/test_tpsa_tensor.py. The tests describe intended semantics and may
fail until the issues identified in the accompanying review have been corrected.
"""

from __future__ import annotations

import gc
import weakref
from typing import Any, cast

import numpy as np
import pytest

from madng_tpsa import ComplexTpsa, Descriptor, Tpsa, TpsaMap, TpsaTensor
from madng_tpsa._cffi import ffi


def _descriptor(*, params: bool = False) -> Descriptor:
    return Descriptor(
        variables=('tensor_x', 'tensor_px'),
        parameters=('tensor_k',) if params else (),
        order=4,
        param_order=2,
    )


def _components(d: Descriptor, n: int) -> list[Tpsa]:
    result: list[Tpsa] = []
    for i in range(n):
        t = d.zero()
        t.const_part = float(i + 1)
        result.append(t)
    return result


def _assert_ptrs(pointer_array: Any, values: tuple[Tpsa | ComplexTpsa, ...]) -> None:
    for i, value in enumerate(values):
        assert pointer_array[i] == ffi.cast('void*', value.ptr)


@pytest.mark.parametrize('kind', ['flat', 'nested', 'ndarray', 'mixed'])
def test_constructor_infers_shape_and_preserves_row_major_order(kind: str) -> None:
    d = _descriptor()
    a, b, c, e = _components(d, 4)
    if kind == 'flat':
        tensor = TpsaTensor([a, b, c, e])
        expected = (4,)
    elif kind == 'nested':
        tensor = TpsaTensor([[a, b], [c, e]])
        expected = (2, 2)
    elif kind == 'ndarray':
        tensor = TpsaTensor(cast('Any', np.array([[a, b], [c, e]], dtype=object)))
        expected = (2, 2)
    else:
        tensor = TpsaTensor(cast('Any', [[a, b], np.array([c, e], dtype=object)]))
        expected = (2, 2)

    assert tensor.shape == expected
    assert tensor.size == 4
    assert tensor.rank == len(expected)
    assert tuple(tensor.flat) == (a, b, c, e)
    assert tensor.descriptor is d
    np.testing.assert_array_equal(tensor.const_part, np.array([1, 2, 3, 4]).reshape(expected))


def test_constructor_accepts_flat_input_with_explicit_shape() -> None:
    d = _descriptor()
    a, b, c, e = _components(d, 4)
    tensor = TpsaTensor([a, b, c, e], shape=(2, 2))
    assert tensor.shape == (2, 2)
    assert tensor[1, 0] is c
    assert tensor[0, 1] is b


@pytest.mark.parametrize(
    'values',
    [
        [[1], [2, 3]],
        [[[1]], [2]],
    ],
)
def test_constructor_rejects_ragged_input(values: list[object]) -> None:
    d = _descriptor()
    components = _components(d, 3)

    def replace_ids(items: object) -> object:
        if isinstance(items, list):
            return [replace_ids(v) for v in items]
        assert isinstance(items, int)
        return components[items - 1]

    with pytest.raises(ValueError, match='[Rr]agged'):
        TpsaTensor(cast('Any', replace_ids(values)))


@pytest.mark.parametrize('shape', [(3,), (2, 3), (2, 0), (2, -1), (True, 2)])
def test_constructor_rejects_bad_explicit_shape(shape: tuple[int, ...]) -> None:
    d = _descriptor()
    with pytest.raises(ValueError):
        TpsaTensor(_components(d, 4), shape=shape)


def test_constructor_rejects_different_labelled_descriptors() -> None:
    dx = Descriptor(variables=('tensor_d1_x', 'tensor_d1_px'), order=4)
    dy = Descriptor(variables=('tensor_d2_x', 'tensor_d2_px'), order=4)
    assert dx.ptr == dy.ptr
    assert dx is not dy
    with pytest.raises(ValueError, match='Descriptor'):
        TpsaTensor([dx.var(1), dy.var(1)])


def test_constructor_promotes_mixed_real_and_complex_and_preserves_inputs() -> None:
    d = _descriptor(params=True)
    real = d.var('tensor_x')
    complex_component = d.complex_zero()
    complex_component.set((0, 1, 0), 2 + 3j)
    tensor = TpsaTensor([real, complex_component])

    assert tensor.is_complex
    assert tensor.descriptor is d
    assert all(isinstance(t, ComplexTpsa) for t in tensor.flat)
    assert tensor[0] is not real
    assert tensor[1] is complex_component
    assert tensor.flat[0].get((1, 0, 0)) == 1
    assert isinstance(real, Tpsa)


def test_full_and_partial_indexing_and_iteration() -> None:
    d = _descriptor()
    parts = _components(d, 12)
    tensor = TpsaTensor(parts, shape=(2, 2, 3))
    view = tensor[1]
    row = view[0]

    assert view.shape == (2, 3)
    assert row.shape == (3,)
    assert view.descriptor is d
    assert tensor[1, 0, 2] is parts[8]
    assert tensor[1][0][2] is parts[8]
    assert tensor[-1, -1, -1] is parts[-1]
    assert tensor[cast('Any', (1, 0, np.int64(2)))] is parts[8]
    assert len(tensor) == 2
    assert tuple(tensor[0].flat) == tuple(parts[:6])
    assert tuple(tensor[1].flat) == tuple(parts[6:])
    assert all(isinstance(v, TpsaTensor) for v in tensor)
    assert parts[8] in view
    assert parts[0] not in view


@pytest.mark.parametrize(
    'key,exc',
    [
        (True, TypeError),
        ((0, np.bool_(0)), TypeError),
        ((0, 2), IndexError),
        ((-3, 0), IndexError),
        ((0, 0, 0), IndexError),
        ((), IndexError),
        (0.5, TypeError),
    ],
)
def test_indexing_rejects_bad_indices(key: object, exc: type[Exception]) -> None:
    d = _descriptor()
    tensor = TpsaTensor(_components(d, 4), shape=(2, 2))
    with pytest.raises(exc):
        _ = tensor[cast('Any', key)]


def test_setitem_updates_existing_views_and_native_pointer_array() -> None:
    d = _descriptor()
    tensor = TpsaTensor(_components(d, 6), shape=(2, 3))
    view = tensor[1]
    old = tensor[1, 1]
    replacement = d.var('tensor_x')

    tensor[1, 1] = replacement
    assert tensor[1, 1] is replacement
    assert view[1] is replacement
    assert tensor[1, 1] is not old
    _assert_ptrs(tensor.pointer_array, tuple(tensor.flat))
    _assert_ptrs(view.pointer_array, tuple(view.flat))


def test_setitem_partial_and_overlapping_assignments() -> None:
    d = _descriptor()
    tensor = TpsaTensor(_components(d, 6), shape=(2, 3))
    original_second = tuple(tensor[1].flat)
    tensor[0] = tensor[1]
    assert tuple(tensor[0].flat) == original_second
    assert tensor[0, 0] is tensor[1, 0]
    _assert_ptrs(tensor.pointer_array, tuple(tensor.flat))


def test_setitem_rejects_incompatible_values_without_mutation() -> None:
    d = _descriptor()
    tensor = TpsaTensor(_components(d, 4), shape=(2, 2))
    original = tuple(tensor.flat)
    with pytest.raises(TypeError):
        tensor[0, 0] = cast('Any', d.complex_zero())
    with pytest.raises(ValueError, match='[Dd]escriptor'):
        different = Descriptor(variables=('tensor_wrong_x', 'tensor_wrong_px'), order=4)
        tensor[0, 0] = different.zero()
    with pytest.raises(ValueError, match='[Ss]hape'):
        tensor[0] = TpsaTensor(_components(d, 3))
    assert tuple(tensor.flat) == original
    _assert_ptrs(tensor.pointer_array, original)


def test_views_hold_backing_tpsas_alive() -> None:
    d = _descriptor()
    tensor = TpsaTensor(_components(d, 4), shape=(2, 2))
    view = tensor[1]
    referenced = view[0]
    weak = weakref.ref(referenced)
    del tensor
    del referenced
    gc.collect()
    assert weak() is view[0]
    _assert_ptrs(view.pointer_array, tuple(view.flat))


def test_copy_of_partial_view_is_independent_and_compact() -> None:
    d = _descriptor()
    tensor = TpsaTensor(_components(d, 6), shape=(2, 3))
    view = tensor[1]
    copied = view.copy()
    assert copied.shape == (3,)
    assert copied.descriptor is d
    assert copied._offset == 0
    assert len(copied._flat) == 3
    assert copied[0] is not view[0]
    copied.flat[0].const_part = 42
    assert view[0].const_part != 42


@pytest.mark.parametrize('already_complex', [False, True])
def test_to_complex_returns_deep_copy_even_when_already_complex(*, already_complex: bool) -> None:
    d = _descriptor()
    values = [d.complex_zero(), d.complex_zero()] if already_complex else _components(d, 2)
    tensor = TpsaTensor(values)
    promoted = tensor.to_complex()
    assert promoted is not tensor
    assert promoted.is_complex
    assert all(isinstance(t, ComplexTpsa) for t in promoted.flat)
    assert all(a is not b for a, b in zip(tensor.flat, promoted.flat, strict=True))
    promoted.flat[0].const_part = 20 + 5j
    assert tensor[0].const_part != 20 + 5j


def test_order_and_degree_properties_and_heterogeneous_orders() -> None:
    d = _descriptor()
    a = d.zero(order=2)
    b = d.zero(order=3)
    a.set((1, 0), 7)
    tensor = TpsaTensor([a, b])
    assert tensor.degree == 1
    np.testing.assert_array_equal(tensor.truncation_orders, [2, 3])
    with pytest.raises(ValueError, match='different truncation orders'):
        _ = tensor.truncation_order


def test_homogeneous_truncate_and_clear_order_return_independent_tensors() -> None:
    d = _descriptor()
    poly = d.zero()
    poly.set((1, 0), 2)
    poly.set((2, 0), 3)
    tensor = TpsaTensor([poly, poly.copy()], shape=(1, 2))
    homogeneous = tensor.homogeneous(2)
    truncated = tensor.truncate(1)
    cleared = tensor.clear_order(2)
    for result in (homogeneous, truncated, cleared):
        assert result.shape == tensor.shape
        assert result[0, 0] is not poly
    assert homogeneous.flat[0].monomial_coeffs(tol=0) == {(2, 0): 3.0}
    assert truncated.flat[0].monomial_coeffs(tol=0) == {(1, 0): 2.0}
    assert cleared.flat[0].monomial_coeffs(tol=0) == {(1, 0): 2.0}
    assert poly.get((2, 0)) == 3


def test_const_part_scalar_and_shaped_assignment() -> None:
    d = _descriptor()
    tensor = TpsaTensor(_components(d, 6), shape=(2, 3))
    tensor.const_part = 8.0
    np.testing.assert_array_equal(tensor.const_part, np.full((2, 3), 8.0))
    tensor.const_part = cast('Any', [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(tensor.const_part, [[1, 2, 3], [4, 5, 6]])
    assert tensor[1, 0].const_part == 4.0


def test_const_part_rejects_shape_and_complex_before_mutation() -> None:
    d = _descriptor()
    tensor = TpsaTensor(_components(d, 4), shape=(2, 2))
    before = tensor.const_part.copy()
    with pytest.raises(ValueError, match='[Ss]hape'):
        tensor.const_part = [1, 2, 3, 4]
    with pytest.raises((TypeError, ValueError), match='complex'):
        tensor.const_part = cast('Any', [[9, 10], [11, 12 + 1j]])
    np.testing.assert_array_equal(tensor.const_part, before)


def test_complex_const_part_assignment_preserves_imaginary_part() -> None:
    d = _descriptor()
    tensor = TpsaTensor([d.complex_zero(), d.complex_zero()])
    tensor.const_part = [1 + 2j, 3 - 4j]
    np.testing.assert_array_equal(tensor.const_part, [1 + 2j, 3 - 4j])


def test_serialization_round_trip_of_partial_view() -> None:
    d = _descriptor(params=True)
    values = _components(d, 6)
    tensor = TpsaTensor(values, shape=(2, 3))
    view = tensor[1]
    encoded = view.to_dict(tol=0)
    recreated = TpsaTensor.from_dict(encoded)
    assert recreated.shape == (3,)
    assert recreated.descriptor is d
    assert recreated._offset == 0
    for original, restored in zip(view.flat, recreated.flat, strict=True):
        assert restored is not original
        assert restored.monomial_coeffs(tol=0) == original.monomial_coeffs(tol=0)


def test_tensor_change_descriptor_converts_each_component_and_copies() -> None:
    old = Descriptor(variables=('tensor_conv_x', 'tensor_conv_px'), order=4)
    new = Descriptor(variables=('tensor_conv_px', 'tensor_conv_x'), order=4)
    first = old.zero()
    first.set((2, 1), 5)
    tensor = TpsaTensor([first, old.zero()], shape=(1, 2))
    converted = tensor.change_descriptor(new)
    assert converted is not tensor
    assert converted.shape == (1, 2)
    assert converted.descriptor is new
    assert converted.flat[0].get((1, 2)) == 5
    assert first.get((2, 1)) == 5
    same = tensor.change_descriptor(old)
    assert same is not tensor
    assert same[0, 0] is not first


def test_change_descriptor_honours_explicit_remapping_with_same_descriptor() -> None:
    d = Descriptor(variables=('tensor_remap_x', 'tensor_remap_px'), order=4)
    t = d.var('tensor_remap_x')
    tensor = TpsaTensor([t])
    swapped = tensor.change_descriptor(
        d,
        variable_map={
            'tensor_remap_x': 'tensor_remap_px',
            'tensor_remap_px': 'tensor_remap_x',
        },
    )
    assert swapped is not tensor
    assert swapped.flat[0].get((1, 0)) == 0
    assert swapped.flat[0].get((0, 1)) == 1
    assert t.get((1, 0)) == 1


def test_evaluate_rank_two_tensor_preserves_output_shape() -> None:
    d = _descriptor()
    x = d.var('tensor_x')
    px = d.var('tensor_px')
    t = TpsaTensor([[x, px], [x + 2 * px, x * px]])
    result = t.evaluate([2.0, 3.0])
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, [[2, 3], [8, 6]])


def test_evaluate_with_parameters() -> None:
    d = _descriptor(params=True)
    f = d.zero()
    f.set((1, 0, 0), 2)
    f.set((0, 0, 1), 3)
    g = d.zero()
    g.set((0, 1, 0), -1)
    t = TpsaTensor([[f], [g]])
    np.testing.assert_allclose(
        t.evaluate([5, 6], parameters=[7]),
        [[31], [-6]],
    )
    with pytest.raises(ValueError, match='[Pp]arameter'):
        t.evaluate([5, 6], parameters=[1, 2])
    with pytest.raises(ValueError, match='[Cc]oordinate'):
        t.evaluate([5], parameters=[7])


def test_evaluate_complex_coordinates_without_mutating_real_tensor() -> None:
    d = _descriptor()
    x, px = d.vars()
    tensor = TpsaTensor([x * x, px])
    with pytest.warns(UserWarning, match='Assigning complex coordinates to a real TpsaTensor'):
        result = tensor.evaluate([1 + 2j, 3.0])
    assert result.dtype.kind == 'c'
    np.testing.assert_allclose(result, [-3 + 4j, 3])
    assert not tensor.is_complex
    assert tensor[0].descriptor is d


def test_evaluate_complex_tensor() -> None:
    d = _descriptor()
    x, px = d.vars()
    tensor = TpsaTensor([x + px, x * px]).to_complex()
    tensor.flat[0].const_part = 1 + 1j
    np.testing.assert_allclose(tensor.evaluate([2.0, 3.0]), [6 + 1j, 6])


def test_const_part_assignment_on_view_leaves_siblings_unchanged() -> None:
    d = _descriptor()
    series = [d.zero() for _ in range(6)]
    tensor = TpsaTensor(series, shape=(2, 3))
    view = tensor[1]
    view.const_part = [10, 20, 30]

    np.testing.assert_array_equal(tensor.const_part, [[0, 0, 0], [10, 20, 30]])
    assert all(tensor.flat[i] is series[i] for i in range(6))


def test_const_part_real_accepts_zero_imaginary_part() -> None:
    d = _descriptor()
    t = TpsaTensor([d.zero(), d.zero()])
    t.const_part = [1 + 0j, 2 + 0j]
    np.testing.assert_array_equal(t.const_part, [1, 2])
    assert not t.is_complex


def test_const_part_rejection_is_atomic() -> None:
    d = _descriptor()
    t = TpsaTensor([d.zero(), d.zero()])
    t.const_part = [3, 4]
    before = t.const_part.copy()

    with pytest.raises((TypeError, ValueError), match='complex'):
        t.const_part = [10, 20 + 2j]
    np.testing.assert_array_equal(t.const_part, before)


def test_change_descriptor_of_view_creates_compact_independent_result() -> None:
    old = _descriptor()
    new = Descriptor(variables=('tensor_px', 'tensor_x'), order=4)
    x, px = old.vars()
    t = TpsaTensor([x, px, 2 * x, 3 * px], shape=(2, 2))
    view = t[1]
    converted = view.change_descriptor(new)

    assert converted.shape == (2,)
    assert converted.descriptor is new
    assert converted._offset == 0
    assert len(converted.flat) == 2
    assert converted.flat[0] is not view.flat[0]
    np.testing.assert_allclose(converted.evaluate((7, 5)), (10, 21))


def test_replace_tpsa_in_view_updates_original_native_pointer() -> None:
    d = _descriptor()
    t = TpsaTensor([d.zero() for _ in range(4)], shape=(2, 2))
    view = t[1]
    replacement = d.var('tensor_x')
    view[0] = replacement

    assert t[1, 0] is replacement
    assert t.pointer_array[2] == ffi.cast('void*', replacement.ptr)
    assert view.pointer_array[0] == ffi.cast('void*', replacement.ptr)


def test_compose_rank_two_with_params_and_complex_substitution() -> None:
    d = _descriptor(params=True)
    x, px = d.vars()
    k = d.param('tensor_k')
    outer = TpsaTensor([[x + k, px], [x * px, x * x]])
    inner = TpsaMap((x, px)).to_complex()
    inner.flat[0].const_part = 1 + 1j

    result = outer.compose(inner)
    assert isinstance(result, TpsaTensor)
    assert result.shape == (2, 2)
    assert result.is_complex
    np.testing.assert_allclose(
        result.evaluate((2, 3), parameters=(4,)),
        [[7 + 1j, 3], [9 + 3j, 8 + 6j]],
    )
    assert not outer.is_complex


def test_evaluate_large_view_uses_only_selected_component_block() -> None:
    d = _descriptor()
    x, px = d.vars()
    t = TpsaTensor([x, px, x + px, x * px, x * x, px * px], shape=(3, 2))
    view = t[1]  # Prefix views are supported; slicing is not.
    np.testing.assert_allclose(view.evaluate((2, 3)), (5, 6))


def test_tensor_composition_does_not_add_output_coordinates() -> None:
    d = _descriptor()
    x, px = d.vars()
    tensor = TpsaTensor([[x + 1, px], [x * x, px * px]])
    inner = TpsaMap((x + 1, px + 2))
    result = tensor.compose(inner)
    assert type(result) is TpsaTensor
    assert result.shape == tensor.shape
    np.testing.assert_allclose(result.evaluate((2, 3)), [[4, 5], [9, 25]])


def test_evaluate_tensor_more_components_than_variables_with_params():
    """Evaluation must batch by num_vars, not num_vars + num_params."""
    d = _descriptor(params=True)
    x, px = d.vars()
    k = d.param('tensor_k')

    tensor = TpsaTensor(
        [
            [x + k, px, x + px],
            [x * px, x * x, k * x],
        ]
    )

    assert tensor.shape == (2, 3)
    assert tensor.size > d.num_vars

    np.testing.assert_allclose(
        tensor.evaluate([2, 3], parameters=[4]),
        [[6, 3, 5], [6, 4, 8]],
    )


def test_evaluate_partial_view_with_parameters():
    d = _descriptor(params=True)
    x, px = d.vars()
    k = d.param('tensor_k')

    tensor = TpsaTensor(
        [
            [x + k, px, x + px],
            [x * px, x * x, k * x],
        ]
    )

    view = tensor[1]

    assert view.shape == (3,)
    np.testing.assert_allclose(
        view.evaluate([2, 3], parameters=[4]),
        [6, 4, 8],
    )


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
def test_numpy_array_of_tpsas(*, complex_series: bool) -> None:
    d = Descriptor(variables=('array_x', 'array_px'), order=4)
    x, px = d.vars()

    components = [x, px, x + px, x * px]
    if complex_series:
        components = [ComplexTpsa.from_tpsa(t) for t in components]

    # NumPy must treat TPSAs as scalar objects, not sequences.
    array = np.array(
        [[components[0], components[1]], [components[2], components[3]]],
        dtype=object,
    )

    assert array.shape == (2, 2)
    assert array.dtype == np.dtype(object)

    # Verify that NumPy preserved the objects themselves.
    for actual, expected in zip(array.flat, components, strict=True):
        assert actual is expected

    # Verify that the tensor constructor accepts the NumPy array.
    tensor = TpsaTensor(cast('Any', array))

    assert tensor.shape == (2, 2)
    assert tensor.is_complex is complex_series

    for actual, expected in zip(tensor.flat, components, strict=True):
        assert actual is expected

    np.testing.assert_allclose(
        tensor.evaluate([2, 3]),
        [[2, 3], [5, 6]],
    )


def test_numpy_array_of_mixed_tpsas() -> None:
    d = Descriptor(variables=('mixed_x', 'mixed_px'), order=4)
    x, px = d.vars()

    complex_px = ComplexTpsa.from_tpsa(px)

    array = np.array([[x, complex_px]], dtype=object)

    assert array.shape == (1, 2)
    assert array[0, 0] is x
    assert array[0, 1] is complex_px

    tensor = TpsaTensor(cast('Any', array))

    assert tensor.shape == (1, 2)
    assert tensor.is_complex
    assert all(isinstance(t, ComplexTpsa) for t in tensor.flat)

    # The real component is promoted, the complex component is reused.
    assert tensor[0, 0] is not x
    assert tensor[0, 1] is complex_px

    # Original TPSAs must remain unchanged.
    assert isinstance(x, Tpsa)

    np.testing.assert_allclose(
        tensor.evaluate([2, 3]),
        [[2, 3]],
    )
