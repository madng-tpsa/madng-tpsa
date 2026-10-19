"""TPSA conversion / labelled-descriptor regression tests for real and complex TPSAs.

Run after integrating the label-aware descriptor owner and to_descriptor().
"""

from __future__ import annotations

import pytest

from madng_tpsa import Descriptor


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
def test_to_descriptor_reorders_variables_and_copies(complex_series):
    source = Descriptor(variables=('swap_x', 'swap_px'), order=5)
    target = Descriptor(variables=('swap_px', 'swap_x'), order=5)
    assert source.ptr == target.ptr and source is not target
    if complex_series:
        t = source.complex_zero()
        value = 1 + 2j
        t.set((2, 1), value)
    else:
        t = source.zero()
        value = 3.5
        t.set((2, 1), value)

    converted = t.to_descriptor(target)
    assert type(converted) is type(t)
    assert converted is not t
    assert converted.descriptor is target
    assert converted.truncation_order == t.truncation_order
    assert converted.get((1, 2)) == pytest.approx(value)
    assert converted.get((2, 1)) == 0
    assert t.get((2, 1)) == pytest.approx(value)
    converted.set((1, 2), 0)
    assert t.get((2, 1)) == pytest.approx(value)


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
def test_to_descriptor_renames_variables(complex_series):
    old = Descriptor(variables=('rename_x', 'rename_px'), order=4)
    new = Descriptor(variables=('rename_q', 'rename_p'), order=4)
    if complex_series:
        t = old.complex_zero()
        t.set((1, 2), (4 - 1j))
    else:
        t = old.zero()
        t.set((1, 2), 4.0)

    converted = t.to_descriptor(new, variable_map={'rename_x': 'rename_q', 'rename_px': 'rename_p'})
    assert converted.descriptor is new
    assert converted.get((1, 2)) == pytest.approx(4 - 1j if complex_series else 4.0)


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
def test_to_descriptor_reorders_and_extends_parameters(complex_series):
    old = Descriptor(
        variables=('lift_x', 'lift_px'), parameters=('lift_k',), order=5, param_order=2
    )
    new = Descriptor(
        variables=('lift_px', 'lift_y', 'lift_x'),
        parameters=('lift_eps', 'lift_k'),
        order=5,
        param_order=2,
    )
    if complex_series:
        t = old.complex_zero()
        value = 2 + 3j
        t.set((1, 0, 1), value)
    else:
        t = old.zero()
        value = 2.0
        t.set((1, 0, 1), value)

    u = t.to_descriptor(new)
    assert u.descriptor is new
    assert u.get((0, 0, 1, 0, 1)) == pytest.approx(value)
    assert u.get((1, 0, 0, 0, 0)) == 0


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
def test_to_descriptor_missing_unused_variable_is_allowed(complex_series):
    old = Descriptor(variables=('drop_x', 'drop_unused'), order=4)
    new = Descriptor(variables=('drop_x',), order=4)
    t = old.complex_zero() if complex_series else old.zero()
    t.set((2, 0), 3)
    u = t.to_descriptor(new)
    assert u.get((2,)) == 3


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
@pytest.mark.parametrize('truncate', [False, True])
def test_to_descriptor_rejects_missing_used_variable(complex_series, truncate):
    old = Descriptor(variables=('mustkeep_x', 'mustkeep_px'), order=4)
    new = Descriptor(variables=('mustkeep_x',), order=4)
    t = old.complex_zero() if complex_series else old.zero()
    t.set((0, 1), 2)
    with pytest.raises(ValueError, match='remove|depend'):
        t.to_descriptor(new, truncate=truncate)


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
@pytest.mark.parametrize('truncate', [False, True])
def test_to_descriptor_rejects_missing_used_parameter(complex_series, truncate):
    old = Descriptor(variables=('param_x',), parameters=('param_k',), order=4, param_order=2)
    new = Descriptor(variables=('param_x',), order=4)
    t = old.complex_zero() if complex_series else old.zero()
    t.set((1, 1), 2)
    with pytest.raises(ValueError, match='remove|depend'):
        t.to_descriptor(new, truncate=truncate)


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
def test_to_descriptor_truncation_requires_opt_in(complex_series):
    old = Descriptor(variables=('cut_x', 'cut_px'), order=5)
    new = Descriptor(variables=('cut_x', 'cut_px'), order=2)
    t = old.complex_zero(order=5) if complex_series else old.zero(order=5)
    t.set((1, 0), 2)
    t.set((3, 0), 7)

    with pytest.raises(ValueError, match='represent|order|valid'):
        t.to_descriptor(new)

    u = t.to_descriptor(new, truncate=True)
    assert u.truncation_order == 2
    assert u.get((1, 0)) == 2
    assert u.monomial_coeffs(tol=0) == {(1, 0): 2}
    assert t.get((3, 0)) == 7


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
def test_to_descriptor_per_variable_limit_requires_opt_in(complex_series):
    old = Descriptor(variables=('cap_x', 'cap_px'), order=5)
    new = Descriptor(
        variables=('cap_x', 'cap_px'),
        order=5,
        max_orders=(2, 5),
    )
    t = old.complex_zero() if complex_series else old.zero()
    t.set((1, 0), 2)
    t.set((3, 0), 9)

    with pytest.raises(ValueError, match='represent|order|valid'):
        t.to_descriptor(new)

    u = t.to_descriptor(new, truncate=True)
    assert u.get((1, 0)) == 2
    assert u.monomial_coeffs(tol=0) == {(1, 0): 2}


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
def test_to_descriptor_rejects_bad_label_maps(complex_series):
    old = Descriptor(variables=('mapping_x', 'mapping_px'), order=4)
    new = Descriptor(variables=('mapping_q', 'mapping_p'), order=4)
    t = old.complex_zero() if complex_series else old.zero()
    with pytest.raises(ValueError, match='Unknown source'):
        t.to_descriptor(new, variable_map={'missing': 'mapping_q'})
    with pytest.raises(ValueError, match='Unknown target'):
        t.to_descriptor(new, variable_map={'mapping_x': 'absent'})
    with pytest.raises(ValueError, match='same target'):
        t.to_descriptor(new, variable_map={'mapping_x': 'mapping_q', 'mapping_px': 'mapping_q'})


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
def test_to_descriptor_same_descriptor_returns_independent_copy(complex_series):
    d = Descriptor(variables=('copy_x', 'copy_px'), order=4)
    t = d.complex_zero() if complex_series else d.zero()
    t.set((1, 0), 3)

    u = t.to_descriptor(d)
    assert u is not t
    assert u.ptr != t.ptr
    assert u.descriptor is d
    assert u.get((1, 0)) == 3
    u.set((1, 0), 5)
    assert t.get((1, 0)) == 3


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
def test_to_descriptor_explicit_parameter_rename(complex_series):
    old = Descriptor(
        variables=('pmap_x',),
        parameters=('pmap_old',),
        order=4,
        param_order=2,
    )
    new = Descriptor(
        variables=('pmap_x',),
        parameters=('pmap_new',),
        order=4,
        param_order=2,
    )
    if complex_series:
        t = old.complex_zero()
        t.set((1, 1), 1 + 2j)
    else:
        t = old.zero()
        t.set((1, 1), 1.5)

    u = t.to_descriptor(new, parameter_map={'pmap_old': 'pmap_new'})
    assert u.descriptor is new
    assert u.get((1, 1)) == pytest.approx(1 + 2j if complex_series else 1.5)
    assert t.descriptor is old


@pytest.mark.parametrize('complex_series', [False, True], ids=['real', 'complex'])
def test_to_descriptor_does_not_expand_source_truncation_order(complex_series):
    old = Descriptor(variables=('allocation_x', 'allocation_px'), order=4)
    new = Descriptor(variables=('allocation_px', 'allocation_x', 'allocation_y'), order=6)
    t = old.complex_zero(order=2) if complex_series else old.zero(order=2)
    t.set((1, 1), 3)

    u = t.to_descriptor(new)
    assert u.truncation_order == 2
    assert u.get((1, 1, 0)) == 3
    assert u.descriptor.order == 6
