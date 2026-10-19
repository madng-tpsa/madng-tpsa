"""Regression tests for TpsaMap.

Place beside the existing tensor tests. These specify the intended map API:
full descriptor-ordered outputs, scalar offsets from identity, explicit
coordinate-space extension, and label-aware descriptor conversion.
"""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import pytest

from madng_tpsa import ComplexTpsa, Descriptor, Tpsa, TpsaMap, TpsaTensor
from madng_tpsa._cffi import ffi


def _descriptor(*, parameters: bool = False) -> Descriptor:
    return Descriptor(
        variables=('maptest_x', 'maptest_px'),
        parameters=('maptest_k',) if parameters else (),
        order=4,
        param_order=2,
    )


def test_identity_constructor_and_descriptor_order() -> None:
    d = _descriptor(parameters=True)
    m = TpsaMap(d)

    assert isinstance(m, TpsaTensor)
    assert m.shape == (2,)
    assert m.rank == 1
    assert m.descriptor is d
    assert m.coord_names == d.variables
    np.testing.assert_allclose(m.evaluate((3.0, 5.0), parameters=(7.0,)), (3.0, 5.0))
    assert m.coordinates['maptest_x'] is m.flat[0]
    assert m.coordinates['maptest_px'] is m.flat[1]


def test_positional_tpsas_and_mixed_offsets() -> None:
    d = _descriptor()
    x, px = d.vars()
    m = TpsaMap([x * x, -2.0], descriptor=d)

    np.testing.assert_allclose(m.evaluate((3.0, 5.0)), (9.0, 3.0))
    assert m.flat[0].descriptor is d
    assert m.flat[1].descriptor is d


def test_positional_numeric_components_are_offsets_not_constant_maps() -> None:
    d = _descriptor()
    m = TpsaMap((2.5, -1.0), descriptor=d)
    np.testing.assert_allclose(m.evaluate((3.0, 5.0)), (5.5, 4.0))
    np.testing.assert_allclose(TpsaMap((0.0, 0.0), descriptor=d).evaluate((3, 5)), (3, 5))


def test_named_components_infer_descriptor_and_preserve_other_identity() -> None:
    d = _descriptor()
    x = d.var('maptest_x')
    m = TpsaMap(maptest_x=x * x)

    assert m.descriptor is d
    assert m.coord_names == d.variables
    np.testing.assert_allclose(m.evaluate((3, 5)), (9, 5))
    assert m.maptest_x is m.flat[0]
    assert m.maptest_px is m.flat[1]


def test_named_numeric_offsets_use_explicit_descriptor() -> None:
    d = _descriptor()
    m = TpsaMap(descriptor=d, maptest_px=-3.0)
    np.testing.assert_allclose(m.evaluate((2, 7)), (2, 4))
    assert m.maptest_px is m.flat[1]


def test_complex_offset_promotes_entire_map() -> None:
    d = _descriptor()
    m = TpsaMap(descriptor=d, maptest_x=1 + 2j)

    assert m.is_complex
    assert all(isinstance(t, ComplexTpsa) for t in m.flat)
    np.testing.assert_allclose(m.evaluate((3, 5)), (4 + 2j, 5))


def test_mixed_real_and_complex_polynomial_components() -> None:
    d = _descriptor()
    x, px = d.vars()
    c = d.complex_zero()
    c.set((0, 1), 1 + 2j)
    m = TpsaMap([x, c])

    assert m.is_complex
    assert all(isinstance(t, ComplexTpsa) for t in m.flat)
    assert isinstance(x, Tpsa)
    np.testing.assert_allclose(m.evaluate((3, 5)), (3, 5 + 10j))


@pytest.mark.parametrize('components', [[], [0], [0, 0, 0]])
def test_rejects_wrong_number_of_components(components: list[int]) -> None:
    with pytest.raises(ValueError, match='components'):
        TpsaMap(components, descriptor=_descriptor())


def test_rejects_missing_descriptor_for_numeric_components() -> None:
    with pytest.raises(ValueError, match='descriptor'):
        TpsaMap((0.0, 1.0))


def test_rejects_unknown_named_coordinate() -> None:
    with pytest.raises(ValueError, match='Unknown coordinates'):
        TpsaMap(descriptor=_descriptor(), unrecognised_coordinate=1.0)


def test_rejects_mixing_positional_and_named_components() -> None:
    d = _descriptor()
    with pytest.raises(TypeError, match='positional'):
        cast('Any', TpsaMap)([d.var('maptest_x'), d.var('maptest_px')], descriptor=d, maptest_x=1.0)


def test_rejects_duplicate_descriptor() -> None:
    d = _descriptor()
    with pytest.raises(TypeError, match='twice'):
        cast('Any', TpsaMap)(d, descriptor=d)


def test_rejects_nested_map_components() -> None:
    d = _descriptor()
    x, px = d.vars()
    with pytest.raises(ValueError, match='nested'):
        TpsaMap(cast('Any', [[x, px]]), descriptor=d)


def test_rejects_mixed_labelled_descriptors() -> None:
    d = _descriptor()
    other = Descriptor(variables=('other_x', 'other_px'), order=4)
    with pytest.raises(ValueError, match='Descriptor'):
        TpsaMap([d.var('maptest_x'), other.var('other_px')])


def test_mapping_accessors_track_component_replacement() -> None:
    d = _descriptor()
    m = TpsaMap(d)
    replacement = d.var('maptest_x') * 2
    m[0] = replacement

    assert m.coord_names == d.variables
    assert tuple(m.keys()) == d.variables
    assert tuple(m.values()) == m.flat
    assert dict(m.items())['maptest_x'] is replacement
    assert m.coordinates['maptest_x'] is replacement
    assert m.maptest_x is replacement
    assert m[0] is replacement
    assert m.pointer_array[0] == ffi.cast('void*', replacement.ptr)


def test_repr_mentions_components_and_does_not_raise() -> None:
    m = TpsaMap(_descriptor())
    printed = repr(m)
    assert 'TpsaMap' in printed
    assert 'maptest_x' in printed
    assert 'maptest_px' in printed


def test_copy_is_map_with_independent_components() -> None:
    d = _descriptor()
    m = TpsaMap(d)
    copied = m.copy()

    assert isinstance(copied, TpsaMap)
    assert copied is not m
    assert copied.descriptor is d
    assert copied.coord_names == d.variables
    assert all(a is not b for a, b in zip(m.flat, copied.flat, strict=True))
    copied.flat[0].set_const_part(12)
    assert m.flat[0].const_part == 0


def test_to_complex_keeps_map_type_and_is_independent() -> None:
    d = _descriptor(parameters=True)
    m = TpsaMap(d)
    c = m.to_complex()

    assert isinstance(c, TpsaMap)
    assert c.is_complex
    assert c.coord_names == d.variables
    assert all(isinstance(t, ComplexTpsa) for t in c.flat)
    c.flat[0].set_const_part(1 + 2j)
    assert not m.is_complex
    np.testing.assert_allclose(c.evaluate((3, 4), parameters=(5,)), (4 + 2j, 4))


def test_serialise_round_trip_preserves_coordinates_and_parameter() -> None:
    d = _descriptor(parameters=True)
    m = TpsaMap(d, maptest_x=d.var('maptest_x') + d.param('maptest_k'))
    data = json.loads(json.dumps(m.to_dict(tol=0)))
    recreated = TpsaMap.from_dict(data)

    assert isinstance(recreated, TpsaMap)
    assert recreated.descriptor is d
    assert recreated.coord_names == d.variables
    np.testing.assert_allclose(recreated.evaluate((3, 4), parameters=(5,)), (8, 4))
    assert all(a is not b for a, b in zip(m.flat, recreated.flat, strict=True))


def test_parameter_identity_pointer_array_has_correct_entries() -> None:
    d = _descriptor(parameters=True)
    m = TpsaMap(d)
    ptrs = m.pointer_array_with_params

    assert len(ptrs) == d.num_vars + d.num_params
    for idx, t in enumerate(m.flat):
        assert ptrs[idx] == ffi.cast('void*', t.ptr)
    assert ptrs[d.num_vars] == ffi.cast('void*', m._params[0].ptr)


def test_extend_to_new_variables_and_parameters_fills_identities() -> None:
    d = _descriptor(parameters=True)
    x, px = d.vars()
    source = TpsaMap([x + px, px])
    target = Descriptor(
        variables=('maptest_zeta', 'maptest_px', 'maptest_x'),
        parameters=('maptest_k', 'maptest_epsilon'),
        order=4,
        param_order=2,
    )
    extended = source.extend_to(target)

    assert extended.descriptor is target
    assert extended.coord_names == target.variables
    assert extended.shape == (3,)
    np.testing.assert_allclose(
        extended.evaluate((7, 3, 2), parameters=(5, 6)),
        (7, 3, 5),
    )
    assert source.descriptor is d
    assert source.shape == (2,)


def test_extend_to_reorders_existing_variables_even_without_new_ones() -> None:
    old = Descriptor(variables=('order_map_x', 'order_map_px'), order=4)
    target = Descriptor(variables=('order_map_px', 'order_map_x'), order=4)
    x, px = old.vars()
    original = TpsaMap((x + 2 * px, px))
    converted = original.extend_to(target)

    assert converted.descriptor is target
    assert converted.coord_names == target.variables
    np.testing.assert_allclose(converted.evaluate((4, 2)), (4, 10))


def test_extend_to_rejects_target_missing_original_variable() -> None:
    source = TpsaMap(_descriptor())
    target = Descriptor(variables=('maptest_x',), order=4)
    with pytest.raises(ValueError, match='variables'):
        source.extend_to(target)


def test_extend_to_rejects_target_missing_original_parameter() -> None:
    source = TpsaMap(_descriptor(parameters=True))
    target = _descriptor()
    with pytest.raises(ValueError, match='parameters'):
        source.extend_to(target)


def test_map_to_descriptor_reorders_output_components() -> None:
    old = Descriptor(variables=('convert_map_x', 'convert_map_px'), order=4)
    target = Descriptor(variables=('convert_map_px', 'convert_map_x'), order=4)
    x, px = old.vars()
    source = TpsaMap((x + 2 * px, 3 * px))
    converted = source.to_descriptor(target)

    assert isinstance(converted, TpsaMap)
    assert converted.descriptor is target
    assert converted.coord_names == target.variables
    np.testing.assert_allclose(converted.evaluate((4, 2)), (12, 10))


def test_compose_two_maps_returns_a_map_with_correct_labels() -> None:
    d = _descriptor(parameters=True)
    x, px = d.vars()
    k = d.param('maptest_k')
    outer = TpsaMap((x + k, px))
    inner = TpsaMap((2 * x, px + k))
    composed = outer.compose(inner)

    assert isinstance(composed, TpsaMap)
    assert composed.coord_names == d.variables
    np.testing.assert_allclose(composed.evaluate((3, 4), parameters=(1,)), (7, 5))


def test_composition_does_not_mutate_either_operand() -> None:
    d = _descriptor()
    x, px = d.vars()
    outer = TpsaMap((x + 2 * px, px))
    inner = TpsaMap((x + 1, 3 * px))
    composed = outer.compose(inner)

    np.testing.assert_allclose(composed.evaluate((2, 4)), (27, 12))
    np.testing.assert_allclose(outer.evaluate((2, 4)), (10, 4))
    np.testing.assert_allclose(inner.evaluate((2, 4)), (3, 12))


def test_composition_reconciles_parameter_sets() -> None:
    d1 = Descriptor(
        variables=('union_map_x', 'union_map_px'),
        parameters=('union_map_k',),
        order=4,
        param_order=2,
    )
    d2 = Descriptor(variables=('union_map_x', 'union_map_px'), order=4)
    x, px = d1.vars()
    outer = TpsaMap((x + d1.param('union_map_k'), px))
    inner = TpsaMap((2 * d2.var('union_map_x'), d2.var('union_map_px')))
    result = outer.compose(inner)

    assert result.descriptor.parameters == ('union_map_k',)
    np.testing.assert_allclose(result.evaluate((3, 4), parameters=(5,)), (11, 4))


def test_descriptor_order_does_not_change_due_to_read_access() -> None:
    d = _descriptor()
    m = TpsaMap(d)
    assert m.descriptor is d
    _ = m.coordinates
    _ = tuple(m.items())
    _ = tuple(m.values())
    assert m.descriptor is d


def test_evaluate_tpsa_map_with_parameters():
    d = Descriptor(
        variables=('map_eval_x', 'map_eval_px'),
        parameters=('map_eval_k',),
        order=4,
        param_order=2,
    )
    x, px = d.vars()
    k = d.param('map_eval_k')

    mapping = TpsaMap([x + k, 2 * px])

    assert mapping.shape == (2,)
    np.testing.assert_allclose(
        mapping.evaluate([2, 3], parameters=[4]),
        [6, 6],
    )
