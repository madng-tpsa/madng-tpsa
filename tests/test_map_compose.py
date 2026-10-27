from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from madng_tpsa import Descriptor, TpsaMap, TpsaTensor
from madng_tpsa._cffi import ffi


def _d(*, params: bool = False) -> Descriptor:
    return Descriptor(
        variables=('native_tensor_x', 'native_tensor_px'),
        parameters=('native_tensor_k',) if params else (),
        order=4,
        param_order=2,
    )


def test_compose_full_real_map_with_parameter_identity() -> None:
    d = _d(params=True)
    x = d.var('native_tensor_x')
    px = d.var('native_tensor_px')
    k = d.param('native_tensor_k')

    outer = TpsaTensor([x + k, px])
    inner = TpsaMap([2 * x, px + k])
    result = outer.compose(inner)

    assert result.shape == (2,)
    assert result.descriptor is d
    np.testing.assert_allclose(
        result.evaluate([3, 4], parameters=[1]),
        [7, 5],
    )
    assert outer[0].descriptor is d


def test_compose_high_rank_tensor_componentwise() -> None:
    d = _d()
    x, px = d.vars()
    outer = TpsaTensor([[x, px], [x + px, x * px]])
    inner = TpsaMap([x + 1, 2 * px])

    composed = outer.compose(inner)
    assert composed.shape == (2, 2)
    np.testing.assert_allclose(
        composed.evaluate([2, 3]),
        [[3, 6], [9, 18]],
    )


def test_compose_reorders_substitution_coordinates_by_label() -> None:
    old = Descriptor(variables=('perm_x', 'perm_px'), order=4)
    reverse = Descriptor(variables=('perm_px', 'perm_x'), order=4)

    # The outer map is represented in (x, px) order.
    x_old = old.var('perm_x')
    px_old = old.var('perm_px')
    outer = TpsaTensor([x_old + 2 * px_old, px_old])

    # The inner map is represented in (px, x) order:
    # px -> 3px; x -> x + 1.
    x_reverse = reverse.var('perm_x')
    px_reverse = reverse.var('perm_px')
    inner = TpsaMap([3 * px_reverse, x_reverse + 1])

    composed = outer.compose(inner)
    assert composed.descriptor.variables == old.variables
    np.testing.assert_allclose(composed.evaluate([2, 4]), [27, 12])


def test_compose_automatically_unions_parameter_sets() -> None:
    with_parameter = Descriptor(
        variables=('common_x', 'common_px'),
        parameters=('common_k',),
        order=4,
        param_order=2,
    )
    without_parameter = Descriptor(
        variables=('common_x', 'common_px'),
        order=4,
    )

    outer = TpsaTensor(
        [
            with_parameter.var('common_x') + with_parameter.param('common_k'),
            with_parameter.var('common_px'),
        ]
    )
    inner = TpsaMap(
        [
            2 * without_parameter.var('common_x'),
            without_parameter.var('common_px'),
        ]
    )
    composed = outer.compose(inner)
    assert composed.descriptor.parameters == ('common_k',)
    np.testing.assert_allclose(
        composed.evaluate([3, 4], parameters=[5]),
        [11, 4],
    )


def test_compose_mixed_real_and_complex() -> None:
    d = _d()
    x, px = d.vars()
    left = TpsaTensor([x + 1, px])
    right = TpsaMap([x, px]).to_complex()
    right[0].const_part = 2 + 1j
    result = left.compose(right)
    assert result.is_complex
    np.testing.assert_allclose(result.evaluate([3, 4]), [6 + 1j, 4])
    assert not left.is_complex


def test_compose_rejects_incompatible_variable_sets() -> None:
    d1 = Descriptor(variables=('co_x', 'co_px'), order=3)
    d2 = Descriptor(variables=('co_y', 'co_py'), order=3)
    outer = TpsaTensor(d1.vars())
    inner = TpsaMap(d2.vars())
    with pytest.raises(ValueError, match='variable'):
        outer.compose(inner)


def test_compose_rejects_nonvector_substitution() -> None:
    d = _d()
    coords = d.vars()
    outer = TpsaTensor(coords)
    with pytest.raises(TypeError, match='Can only compose with a TpsaMap!'):
        outer.compose(cast('Any', TpsaTensor([[coords[0], coords[1]]])))


def test_compose_rejects_wrong_number_of_coordinates() -> None:
    d = _d()
    x, px = d.vars()
    outer = TpsaTensor([x, px])
    with pytest.raises(ValueError, match='components'):
        outer.compose(TpsaMap([x]))


def test_compose_allocates_order_for_new_higher_degree_terms() -> None:
    """Composing two quadratic polynomials can produce quartic terms."""
    d = _d()  # Descriptor supports total order four.
    outer_component = d.zero(order=2)
    outer_component.set((2, 0), 1)  # F_x = x^2, stored at order 2.
    inner_component = d.zero(order=2)
    inner_component.set((2, 0), 1)  # G_x = x^2, stored at order 2.
    inner_px = d.var('native_tensor_px', order=2)

    outer = TpsaTensor([outer_component])
    inner = TpsaMap([inner_component, inner_px])
    composed = outer.compose(inner)

    assert composed.flat[0].get((4, 0)) == 1
    assert composed[0].truncation_order >= 4
    np.testing.assert_allclose(composed.evaluate([2, 3]), [16])


def test_empty_index_tuple_rejected_but_ellipsis_kept() -> None:
    d = Descriptor(variables=('quirk_q', 'quirk_p'), order=4)
    q, p = d.vars()
    tensor = TpsaTensor([[q, p], [q + p, q - p]])
    with pytest.raises(IndexError, match='Empty index tuple'):
        _ = tensor[()]
    all_view = tensor[...]
    assert isinstance(all_view, TpsaTensor)
    assert all_view.flat == tensor.flat


def test_pointer_arrays_retain_length_and_refresh_from_sibling_mutation() -> None:
    d = Descriptor(variables=('quirk_q', 'quirk_p'), order=4)
    q, p = d.vars()
    tensor = TpsaTensor([[q, p], [q + p, q - p]])
    row = tensor[0]
    assert isinstance(row, TpsaTensor)
    assert len(tensor.pointer_array) == 4
    assert len(row.pointer_array) == 2
    transposed = tensor.T
    assert len(transposed.pointer_array) == 4
    original = transposed.pointer_array
    assert original[1] == ffi.cast('void*', tensor.flat[2].ptr)
    replacement = d.var('quirk_q') + 2 * p
    tensor[1, 0] = replacement
    assert len(transposed.pointer_array) == 4
    assert transposed.pointer_array is not original
    assert transposed.pointer_array[1] == ffi.cast('void*', replacement.ptr)
    assert tensor.pointer_array[2] == ffi.cast('void*', replacement.ptr)


def test_composition_aligns_permuted_variables_by_label() -> None:
    d = Descriptor(variables=('quirk_q', 'quirk_p'), order=4)
    swapped = Descriptor(variables=('quirk_p', 'quirk_q'), order=4)
    q, p = d.vars()
    ps, qs = swapped.vars()
    inner = TpsaMap([ps, qs + ps])
    outer = TpsaMap([q + p**2, p])
    same_order = TpsaMap([q + p, p])
    assert outer.compose(inner).components == outer.compose(same_order).components
    tensor = TpsaTensor([[q**2, p], [q * p, q - p]])
    assert tensor.compose(inner).flat == tensor.compose(same_order).flat
    # A scalar observable with the permuted descriptor is also matched by label.
    observable = swapped.var('quirk_q') + swapped.var('quirk_p') ** 2
    expected_observable = q + p**2
    assert outer.pullback(observable) == outer.pullback(expected_observable)
    # Arithmetic deliberately remains stricter than composition.
    with pytest.raises(ValueError, match='variable'):
        _ = tensor + TpsaTensor([[qs, ps], [qs, ps]])


def test_conventional_and_dragt_operator_order_for_maps_tensors_scalars() -> None:
    d = Descriptor(variables=('quirk_q', 'quirk_p'), order=4)
    q, p = d.vars()
    m = TpsaMap([q + p, p])
    n = TpsaMap([q, p + q])
    tensor = TpsaTensor([[q**2, p], [q - p, q * p]])
    scalar = q**2 + p

    assert (m << n).components == m.compose(n).components
    assert (m >> n).components == n.compose(m).components
    left = tensor << m
    right = m >> tensor
    assert isinstance(left, TpsaTensor)
    assert isinstance(right, TpsaTensor)
    assert left.flat == tensor.compose(m).flat
    assert right.flat == tensor.compose(m).flat
    assert (scalar << m) == m.pullback(scalar)
    assert (m >> scalar) == m.pullback(scalar)
    assert (scalar << m) == (m >> scalar)
    with pytest.raises(TypeError):
        _ = cast('Any', scalar) >> m


def test_operator_composition_keeps_parameter_substitution() -> None:
    d = Descriptor(variables=('quirk_q', 'quirk_p'), parameters=('k',), order=4)
    q, p = d.vars()
    k = d.param('k')
    m = TpsaMap([q + k, p])
    tensor = TpsaTensor([q * p + k, p + k])
    scalar = q + k
    expected = tensor.compose(m)
    left = tensor << m
    right = m >> tensor
    assert isinstance(left, TpsaTensor)
    assert isinstance(right, TpsaTensor)
    assert left.flat == expected.flat == right.flat
    assert scalar << m == m >> scalar == m.pullback(scalar)
