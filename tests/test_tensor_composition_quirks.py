"""Regressions for strided pointer views and the two composition conventions."""

from typing import TYPE_CHECKING, cast

import pytest

from madng_tpsa import Descriptor, TpsaMap, TpsaTensor
from madng_tpsa._cffi import ffi

if TYPE_CHECKING:
    from typing import Any


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
