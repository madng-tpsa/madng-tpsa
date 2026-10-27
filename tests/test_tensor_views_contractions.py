"""Rank-independent TPSA tensor algebra and shared non-contiguous views."""

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from madng_tpsa import Descriptor, TpsaMap, TpsaTensor
from madng_tpsa._cffi import ffi

if TYPE_CHECKING:
    from typing import Any


def _example():
    d = Descriptor(variables=('q', 'p'), order=5)
    q, p = d.vars()
    t = TpsaTensor([q, p, q + p, q - p, q * p, q**2], shape=(2, 3))
    return d, q, p, t


def test_transpose_is_view_and_flat_is_logical_order():
    _d, q, p, t = _example()
    u = t.T
    assert u.shape == (3, 2)
    assert u[0, 0] is q
    assert u[0, 1] is t[1, 0]
    assert u[1, 0] is p
    assert u.flat == (q, q - p, p, q * p, q + p, q**2)
    assert u.T.flat == t.flat
    assert u.T._storage is t._storage


def test_slices_ellipsis_reverse_and_new_axis():
    _d, q, p, t = _example()
    assert t[:, 1].flat == (p, q * p)
    assert t[1, ::-1].flat == (q**2, q * p, q - p)
    assert t[..., 1].flat == (p, q * p)
    assert t[None, ..., 2].shape == (1, 2)
    assert t[1, -1] is t[1, 2]
    with pytest.raises(IndexError):
        _ = t[3]
    with pytest.raises(IndexError):
        _ = t[:, 5]
    with pytest.raises(ValueError):
        t.transpose(0, 0)


def test_transpose_native_pointer_cache_invalidation_and_overlap():
    d, q, p, t = _example()
    u = t.T
    before = u.pointer_array
    assert before[1] == ffi.cast('void*', t[1, 0].ptr)
    replacement = d.var('q') + 2 * p
    t[1, 0] = replacement
    assert u[0, 1] is replacement
    assert u.pointer_array[1] == ffi.cast('void*', replacement.ptr)
    assert u.pointer_array is not before
    t[:, :] = t[:, ::-1]
    assert t[0, 0] == q + p
    assert t[0, 2] == q


def test_assignment_through_strided_view_reflects_in_original():
    d, q, _p, t = _example()
    col = t[:, 0]
    replacement = TpsaTensor([q + 3, q + 4])
    col[:] = replacement
    assert t[0, 0] == q + 3
    assert t[1, 0] == q + 4
    with pytest.raises(ValueError):
        col[:] = TpsaTensor([q])
    with pytest.raises(ValueError):
        col[:] = TpsaTensor([Descriptor(2, 5).var(1), Descriptor(2, 5).var(2)])


def test_matmul_symplectic_identity_and_numeric_s():
    d = Descriptor(variables=('q', 'p'), order=5)
    q, p = d.vars()
    m = TpsaMap([q + p**2, p])
    j = m.jacobian()
    s = np.array([[0.0, 1.0], [-1.0, 0.0]])
    intermediate = j.T @ s
    assert isinstance(intermediate, TpsaTensor)
    residue = intermediate @ j
    assert isinstance(residue, TpsaTensor)
    residue = residue - s
    assert residue.shape == (2, 2)
    assert all(component.is_zero() for component in residue.flat)


def test_einsum_trace_inner_and_tensordot():
    d = Descriptor(variables=('q', 'p'), order=4)
    q, p = d.vars()
    a = TpsaTensor([[q, p], [q + p, q - p]])
    b = TpsaTensor([[p, q], [q, p]])
    assert a.trace() == 2 * q - p
    inner = a.inner(b)
    assert isinstance(inner, TpsaTensor)
    assert inner.shape == (2, 2)
    c = a.tensordot(b, axes=([1], [0]))
    matmul = a @ b
    assert isinstance(c, TpsaTensor)
    assert isinstance(matmul, TpsaTensor)
    assert c.shape == (2, 2)
    assert matmul.flat == c.flat
    einsum_result = a.einsum('ij,jk->ik', a, b)
    assert isinstance(einsum_result, TpsaTensor)
    assert einsum_result.flat == c.flat
    assert (a[0] @ b[:, 0]) == q * p + p * q
    assert a.einsum('ij,ij->', a, b) == (q * p + p * q + (q + p) * q + (q - p) * p)


def test_component_identity_not_polynomial_equality():
    _d, q, _p, t = _example()
    assert q in t
    assert q.copy() not in t


def test_map_components_names_and_composition_operators():
    d = Descriptor(variables=('q', 'p'), order=4)
    q, p = d.vars()
    m = TpsaMap([q + p**2, p])
    n = TpsaMap([q, p + q**2])
    assert m.component_names == ('q', 'p')
    assert m.components == (m[0], m[1])
    assert m.named_components['q'] is m[0]
    for removed in ('coord_names', 'coords', 'coordinates', 'keys', 'values', 'items'):
        assert not hasattr(m, removed)
    assert (m << n).components == m.compose(n).components
    assert (m >> n).components == n.compose(m).components
    assert (q << m) == m.pullback(q)
    assert (m >> q) == m.pullback(q)
    with pytest.raises(TypeError):
        _ = cast('Any', q) >> m
    assert (m ^ 0).components == TpsaMap(d).components
    assert (m ^ 5).components == m.composition_power(5).components
    assert (m ^ 5).components == (m << m << m << m << m).components
    with pytest.raises(ValueError):
        _ = m ^ (-1)


def test_map_slice_returns_plain_tensor():
    d = Descriptor(variables=('q', 'p'), order=4)
    m = TpsaMap(d)
    assert isinstance(m[:], TpsaTensor)
    assert not isinstance(m[:], TpsaMap)


def test_arithmetic_requires_same_variable_order_and_allows_parameters():
    d = Descriptor(variables=('q', 'p'), order=4)
    q, p = d.vars()
    t = TpsaTensor([q, p])
    flipped = Descriptor(variables=('p', 'q'), order=4)
    with pytest.raises(ValueError, match='variable'):
        _ = t + TpsaTensor([flipped.var('q'), flipped.var('p')])
    with pytest.raises(ValueError, match='variable'):
        _ = t * flipped.var('q')
    with pytest.raises(ValueError, match='variable'):
        _ = t + flipped.var('q')
    d2 = Descriptor(variables=('q', 'p'), parameters=('k',), order=4)
    k = d2.param('k')
    combined = t + k
    assert combined.descriptor.parameters == ('k',)
    assert combined[0] == d2.var('q') + k
    assert (t + np.array([1, 2])).evaluate((3, 4)).tolist() == [4, 6]


def test_parameter_augmented_jacobian_has_horizontal_shape():
    d = Descriptor(variables=('q', 'p'), parameters=('k',), order=4)
    q, p = d.vars()
    k = d.param('k')
    m = TpsaMap([q + 2 * p + k, p + q * k])
    at_origin = m.evaluate_jacobian(wrt='all')
    assert at_origin.shape == (2, 3)
    np.testing.assert_allclose(at_origin, [[1, 2, 1], [0, 1, 0]])
    at_point = m.evaluate_jacobian(coordinates=(2, 3), parameters=(4,), wrt='all')
    assert at_point.shape == (2, 3)
    np.testing.assert_allclose(at_point, [[1, 2, 1], [4, 1, 2]])


def test_tensor_high_rank_contraction_and_trace():
    d = Descriptor(variables=('q', 'p'), order=3)
    q, p = d.vars()
    h = TpsaTensor([q, p, p, q, 2 * q, 3 * p, p, q], shape=(2, 2, 2))
    s = np.asarray([[0, 1], [-1, 0]])
    contracted = h.tensordot(s, axes=([2], [0]))
    assert isinstance(contracted, TpsaTensor)
    assert contracted.shape == (2, 2, 2)
    assert contracted[0, 0, 0] == -p
    assert contracted[0, 0, 1] == q
    traced = h.trace(axis1=0, axis2=2)
    assert isinstance(traced, TpsaTensor)
    assert traced.shape == (2,)
    assert traced[0] == q + 3 * p
    assert traced[1] == p + q


def test_invalid_contraction_labels_and_axes():
    _d, _q, _p, t = _example()
    with pytest.raises(ValueError, match='explicit'):
        t.einsum('ij,ij', t, t)
    with pytest.raises(ValueError, match='Incompatible dimensions'):
        t.einsum('ij,jk->ik', t, t)
    with pytest.raises(ValueError, match='axis'):
        t.tensordot(t, axes=([3], [0]))


def test_native_evaluate_and_composition_on_noncontiguous_view():
    d = Descriptor(variables=('q', 'p'), order=4)
    q, p = d.vars()
    original = TpsaTensor([[q + p, p**2], [q**2, q - p]])
    noncontiguous = original.T
    expected = noncontiguous.evaluate(coordinates=(2, 3))
    np.testing.assert_allclose(expected, [[5, 4], [9, -1]])
    substitutions = TpsaMap([q + 1, p + 2])
    composed = noncontiguous.compose(substitutions)
    np.testing.assert_allclose(composed.evaluate((2, 3)), [[8, 9], [25, -2]])
    independent = noncontiguous.copy()
    assert independent.shape == noncontiguous.shape
    assert all(a is not b for a, b in zip(independent.flat, noncontiguous.flat, strict=True))
    assert independent.flat == noncontiguous.flat


def test_empty_and_unsupported_advanced_slices_are_rejected():
    _d, _q, _p, t = _example()
    with pytest.raises(IndexError, match='Empty'):
        _ = t[:, 1:1]
    with pytest.raises(TypeError, match='Boolean'):
        _ = t[True]
    with pytest.raises(TypeError):
        _ = t[[0, 1]]
