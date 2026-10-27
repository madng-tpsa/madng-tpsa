"""TPSA native-pointer interning and ownership regression tests.

Requires the label-aware _NativeDescriptor owner and both revised from_ptr()
methods. All native pointer accesses here refer to live allocations.
"""

from __future__ import annotations

import gc
import weakref

import pytest

from madng_tpsa import ComplexTpsa, Descriptor, Tpsa
from madng_tpsa._cffi import lib


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
def test_unlabelled_from_ptr_reuses_wrapper_with_ambiguous_labels(complex_series):
    dx = Descriptor(variables=('lookup_x', 'lookup_px'), order=5)
    dy = Descriptor(variables=('lookup_y', 'lookup_py'), order=5)
    assert dx.ptr == dy.ptr and dx is not dy

    series = dx.complex_zero() if complex_series else dx.zero()
    cls = ComplexTpsa if complex_series else Tpsa

    # An existing *TPSA* has an unambiguous Python identity, even though the
    # native descriptor admits multiple labelled Python Descriptor views.
    assert cls.from_ptr(series.ptr) is series
    assert cls.from_ptr(series.ptr, owns=True) is series
    assert cls.from_ptr(series.ptr, descriptor=dx) is series
    assert cls.from_ptr(series.ptr, descriptor=dx, owns=True) is series
    assert cls.from_ptr(series.ptr).descriptor is dx


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
def test_alternate_labelled_descriptor_cannot_rewrap_existing_pointer(complex_series):
    dx = Descriptor(variables=('conflict_x', 'conflict_px'), order=5)
    dy = Descriptor(variables=('conflict_y', 'conflict_py'), order=5)
    assert dx.ptr == dy.ptr

    series = dx.complex_zero() if complex_series else dx.zero()
    cls = ComplexTpsa if complex_series else Tpsa

    for owns in (False, True):
        with pytest.raises(ValueError, match='label|descriptor|already'):
            cls.from_ptr(series.ptr, descriptor=dy, owns=owns)

    # The attempted rewrap did not damage the original object.
    assert cls.from_ptr(series.ptr) is series
    if isinstance(series, ComplexTpsa):
        series.set((1, 0), 2 + 3j)
    else:
        series.set((1, 0), 2.0)
    assert series.get((1, 0)) == (2 + 3j if complex_series else 2.0)


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
def test_fresh_native_pointer_requires_ownership_transfer(complex_series):
    descriptor = Descriptor(variables=('adopt_x', 'adopt_px'), order=5)
    cls = ComplexTpsa if complex_series else Tpsa
    new = lib.mad_ctpsa_newd if complex_series else lib.mad_tpsa_newd
    delete = lib.mad_ctpsa_del if complex_series else lib.mad_tpsa_del
    raw = new(descriptor.ptr, 3)
    adopted = False
    try:
        with pytest.raises(ValueError, match='own|live|wrapped'):
            cls.from_ptr(raw)
        with pytest.raises(ValueError, match='own|live|wrapped'):
            cls.from_ptr(raw, descriptor=descriptor)

        # Existing backwards-compatible behaviour: if there is one labelled
        # descriptor variant, the caller may adopt without spelling it out.
        wrapped = cls.from_ptr(raw, owns=True)
        adopted = True
        assert wrapped.ptr == raw
        assert wrapped.descriptor is descriptor
        assert wrapped.truncation_order == 3
        assert cls.from_ptr(raw) is wrapped
    finally:
        if not adopted:
            delete(raw)


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
def test_fresh_pointer_with_ambiguous_descriptor_requires_explicit_label(complex_series):
    dx = Descriptor(variables=('ambig_x', 'ambig_px'), order=6)
    dy = Descriptor(variables=('ambig_y', 'ambig_py'), order=6)
    assert dx.ptr == dy.ptr

    cls = ComplexTpsa if complex_series else Tpsa
    new = lib.mad_ctpsa_newd if complex_series else lib.mad_tpsa_newd
    delete = lib.mad_ctpsa_del if complex_series else lib.mad_tpsa_del
    raw = new(dx.ptr, 2)
    adopted = False
    try:
        with pytest.raises(ValueError, match='[Aa]mbiguous|[Dd]escriptor'):
            cls.from_ptr(raw, owns=True)

        wrapped = cls.from_ptr(raw, descriptor=dy, owns=True)
        adopted = True
        assert wrapped.descriptor is dy
        assert cls.from_ptr(raw) is wrapped
    finally:
        if not adopted:
            delete(raw)


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
def test_from_ptr_rejects_incorrect_native_descriptor(complex_series):
    dx = Descriptor(variables=('native_a', 'native_b'), order=5)
    different = Descriptor(variables=('native_a', 'native_b', 'native_c'), order=5)
    series = dx.complex_zero() if complex_series else dx.zero()
    cls = ComplexTpsa if complex_series else Tpsa

    with pytest.raises(ValueError, match='descriptor|native'):
        cls.from_ptr(series.ptr, descriptor=different)


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
def test_pointer_registry_is_weak_and_tpsa_keeps_labelled_descriptor(complex_series):
    dx = Descriptor(variables=('life_x', 'life_px'), order=7)
    dy = Descriptor(variables=('life_y', 'life_py'), order=7)
    series = dx.complex_zero() if complex_series else dx.zero()
    cls = ComplexTpsa if complex_series else Tpsa
    dref = weakref.ref(dx)
    sref = weakref.ref(series)

    del dx
    gc.collect()
    assert dref() is not None
    assert cls.from_ptr(series.ptr) is series
    assert series.descriptor.variables == ('life_x', 'life_px')
    assert dy.variables == ('life_y', 'life_py')

    del series
    gc.collect()
    assert sref() is None
    assert dref() is None  # Shared weak registry does not retain wrappers.


def test_labelled_descriptors_are_not_implicitly_compatible_in_arithmetic():
    dx = Descriptor(variables=('arith_x', 'arith_px'), order=4)
    dy = Descriptor(variables=('arith_y', 'arith_py'), order=4)
    assert dx.ptr == dy.ptr
    real_x = dx.var('arith_x')
    real_y = dy.var('arith_y')

    with pytest.raises(ValueError, match='descriptor|label|compatible'):
        _ = real_x + real_y

    complex_x = ComplexTpsa.from_tpsa(real_x)
    complex_y = ComplexTpsa.from_tpsa(real_y)
    with pytest.raises(ValueError, match='descriptor|label|compatible'):
        _ = complex_x + complex_y
    assert real_x != real_y
    assert complex_x != complex_y


def test_serialisation_restores_correct_descriptor_among_aliases():
    dx = Descriptor(variables=('serial_x', 'serial_px'), order=5)
    dy = Descriptor(variables=('serial_y', 'serial_py'), order=5)
    assert dx.ptr == dy.ptr
    t = dx.zero()
    t.set((2, 0), 3.25)
    copy = Tpsa.from_dict(t.to_dict(tol=0))
    assert copy.descriptor is dx
    assert copy.get((2, 0)) == pytest.approx(3.25)

    ct = dx.complex_zero()
    ct.set((1, 0), 2 + 4j)
    ccopy = ComplexTpsa.from_dict(ct.to_dict(tol=0))
    assert ccopy.descriptor is dx
    assert ccopy.get((1, 0)) == pytest.approx(2 + 4j)
