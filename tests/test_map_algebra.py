"""Regression tests for native map algebra on TpsaMap.

These tests use the public class API only; the lower-level wrapper tests
remain in test_map_operations.py.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from madng_tpsa import ComplexTpsa, Descriptor, Tpsa, TpsaMap, TpsaTensor


def _d(
    *, params: bool = False, parameters: tuple[str, ...] | None = None, order: int = 5
) -> Descriptor:
    if parameters is None:
        parameters = ('alg_k',) if params else ()
    return Descriptor(
        variables=('alg_q', 'alg_p'),
        parameters=parameters,
        order=order,
        param_order=2,
    )


def test_inverse_of_polynomial_shear_and_both_composition_orders() -> None:
    d = _d()
    q, p = d.vars()
    mapping = TpsaMap([q + p * p, p])
    inverse = mapping.inverse()
    assert isinstance(inverse, TpsaMap)
    np.testing.assert_allclose(inverse.evaluate((2, 3)), (-7, 3))
    np.testing.assert_allclose(mapping.compose(inverse).evaluate((2, 3)), (2, 3))
    np.testing.assert_allclose(inverse.compose(mapping).evaluate((2, 3)), (2, 3))
    np.testing.assert_allclose(mapping.evaluate((2, 3)), (11, 3))


def test_inverse_with_parameter_identity_and_no_operand_mutation() -> None:
    d = _d(params=True)
    q, p = d.vars()
    k = d.param('alg_k')
    mapping = TpsaMap([q + p + k, p])
    inv = mapping.inverse()
    np.testing.assert_allclose(inv.evaluate((4, 3), parameters=(2,)), (-1, 3))
    np.testing.assert_allclose(mapping.compose(inv).evaluate((4, 3), parameters=(2,)), (4, 3))
    np.testing.assert_allclose(mapping.evaluate((4, 3), parameters=(2,)), (9, 3))


def test_inverse_complex_map() -> None:
    d = _d()
    q, p = d.vars()
    mapping = TpsaMap([q + 1j * p * p, p])
    inv = mapping.inverse()
    assert inv.is_complex
    np.testing.assert_allclose(inv.evaluate((2, 3)), (2 - 9j, 3))
    np.testing.assert_allclose(mapping.compose(inv).evaluate((2, 3)), (2, 3))


def test_inverse_rejects_nonzero_constant_and_singular_linear_part() -> None:
    d = _d()
    q, p = d.vars()
    with pytest.raises(ValueError, match='zero constant'):
        TpsaMap([q + 1, p]).inverse()
    with pytest.raises(Exception, match='GTPSA|singular|inverse|invert'):
        TpsaMap([q, d.zero()]).inverse()


def test_partial_inverse_of_selected_shear() -> None:
    d = _d()
    q, p = d.vars()
    mapping = TpsaMap([q + p**2, p])
    result = mapping.partial_inverse([True, False])
    np.testing.assert_allclose(result.evaluate((2, 3)), (-7, 3))
    np.testing.assert_allclose(mapping.evaluate((2, 3)), (11, 3))


def test_partial_inverse_selection_validation() -> None:
    mapping = TpsaMap(_d())
    with pytest.raises(ValueError, match='selection'):
        mapping.partial_inverse([True])
    with pytest.raises(ValueError, match='0 or 1'):
        mapping.partial_inverse([1, 2])
    with pytest.raises(TypeError, match='integers'):
        mapping.partial_inverse(cast('Any', [True, 0.5]))
    with pytest.raises(ValueError, match='zero constant'):
        TpsaMap([_d().var('alg_q') + 1, _d().var('alg_p')]).partial_inverse([1, 0])


def test_translation_acts_on_inputs_not_map_outputs() -> None:
    d = _d()
    q, p = d.vars()
    mapping = TpsaMap([q * q + p, p])
    result = mapping.translate([2, -1])
    np.testing.assert_allclose(result.evaluate((3, 4)), (28, 3))
    np.testing.assert_allclose(mapping.evaluate((3, 4)), (13, 4))
    assert isinstance(result, TpsaMap)


def test_translation_retains_parameters_and_promotes_complex() -> None:
    d = _d(params=True)
    q, p = d.vars()
    k = d.param('alg_k')
    mapping = TpsaMap([q * q + k, p])
    result = mapping.translate((1j, 2))
    assert result.is_complex
    np.testing.assert_allclose(result.evaluate((2, 3), parameters=(5,)), (8 + 4j, 5))
    np.testing.assert_allclose(mapping.evaluate((2, 3), parameters=(5,)), (9, 3))
    with pytest.raises(ValueError, match='offsets'):
        mapping.translate([1])


def test_from_generator_native_sign_convention_and_inverse_conversion() -> None:
    d = _d()
    q, p = d.vars()
    generator = (q * q + p * p) / 2
    field = TpsaMap.from_generator(generator)
    np.testing.assert_allclose(field.evaluate((2, 3)), (-3, 2))
    recovered = field.to_generator()
    assert recovered == generator


def test_from_generator_and_field_to_vector_complex() -> None:
    d = _d()
    q, p = d.vars()
    generator = 1j * (q * q + p * p) / 2
    field = TpsaMap.from_generator(generator)
    assert field.is_complex
    np.testing.assert_allclose(field.evaluate((2, 3)), (-3j, 2j))
    assert field.to_generator() == generator


def test_generator_methods_reject_odd_variable_count() -> None:
    d = Descriptor(variables=('alg_a', 'alg_b', 'alg_c'), order=4)
    with pytest.raises(ValueError, match='even'):
        TpsaMap.from_generator(d.var('alg_a'))
    with pytest.raises(ValueError, match='even'):
        TpsaMap(d).to_generator()


def test_lie_bracket_antisymmetry_and_self_zero() -> None:
    d = _d()
    q, p = d.vars()
    a = TpsaMap([q * q, p])
    b = TpsaMap([p, q])
    ab = a.lie_bracket(b)
    ba = b.lie_bracket(a)
    assert isinstance(ab, TpsaMap)
    np.testing.assert_allclose(ab.evaluate((2, 3)), -ba.evaluate((2, 3)))
    np.testing.assert_allclose(a.lie_bracket(a).evaluate((2, 3)), (0, 0))


def test_lie_bracket_promotes_complex_and_rejects_wrong_descriptor() -> None:
    d = _d()
    q, p = d.vars()
    real = TpsaMap([q * p, p])
    complex_field = TpsaMap([1j * p, q])
    result = real.lie_bracket(complex_field)
    assert result.is_complex
    assert all(isinstance(t, ComplexTpsa) for t in result.flat)
    other = Descriptor(variables=('alg_diff_x', 'alg_diff_p'), order=5)
    with pytest.raises(ValueError, match='descriptor'):
        real.lie_bracket(TpsaMap(other))


def test_exponential_of_zero_field_is_identity_on_arbitrary_map() -> None:
    d = _d()
    q, p = d.vars()
    mapping = TpsaMap([q + p * p, p + q * q])
    zero = TpsaMap([d.zero(), d.zero()])
    result = mapping.exp_poisson(zero)
    np.testing.assert_allclose(result.evaluate((2, 3)), mapping.evaluate((2, 3)))


def test_scalar_hamiltonian_exp_inverse_action() -> None:
    d = _d(order=5)
    q, p = d.vars()
    scalar = q * q / 2
    identity = TpsaMap(d)
    transformed = identity.exp_poisson(scalar)
    restored = transformed.exp_poisson(-scalar)
    np.testing.assert_allclose(restored.evaluate((2, 3)), (2, 3))
    assert isinstance(transformed, TpsaMap)


def test_scalar_complex_exponential_promotes_map() -> None:
    d = _d()
    q, _ = d.vars()
    result = TpsaMap(d).exp_poisson(1j * q * q / 2)
    assert result.is_complex
    assert all(isinstance(t, ComplexTpsa) for t in result.flat)


def test_logarithm_with_near_identity_map_and_round_trip() -> None:
    d = _d(order=5)
    q, p = d.vars()
    # Nilpotent shear; logpb should return a polynomial vector field.
    mapping = TpsaMap([q + p * p, p])
    field = mapping.log_poisson()
    rebuilt = TpsaMap(d).exp_poisson(field)
    np.testing.assert_allclose(rebuilt.evaluate((2, 3)), mapping.evaluate((2, 3)))
    assert isinstance(mapping.log_generator(), Tpsa)


def test_logarithm_rejects_mismatched_guess() -> None:
    mapping = TpsaMap(_d())
    other = Descriptor(variables=('alg_wrong_q', 'alg_wrong_p'), order=5)
    with pytest.raises(ValueError, match='descriptor'):
        mapping.log_poisson(TpsaMap(other))


def test_pullback_polynomial_with_parameters() -> None:
    d = _d(params=True)
    q, p = d.vars()
    k = d.param('alg_k')
    mapping = TpsaMap([q + p, 2 * p])
    function = q * q + k * p
    pulled = mapping.pullback(function)
    np.testing.assert_allclose(TpsaTensor([pulled]).evaluate((2, 3), parameters=(4,)), [49])
    np.testing.assert_allclose(TpsaTensor([function]).evaluate((2, 3), parameters=(4,)), [16])


def test_pullback_complex_promotion_and_descriptor_validation() -> None:
    d = _d()
    q, p = d.vars()
    result = TpsaMap([q + 1j * p, p]).pullback(q * q)
    assert isinstance(result, ComplexTpsa)
    np.testing.assert_allclose(TpsaTensor([result]).evaluate((2, 3)), [-5 + 12j])
    other = Descriptor(variables=('alg_bad_q', 'alg_bad_p'), order=5)
    with pytest.raises(ValueError, match='descriptor'):
        TpsaMap(d).pullback(other.var('alg_bad_q'))


def test_add_subtract_negate_scale_are_componentwise() -> None:
    d = _d()
    q, p = d.vars()
    left = TpsaMap([q + 1, p * p])
    right = TpsaMap([2 * q, -p])
    np.testing.assert_allclose((left + right).evaluate((2, 3)), (7, 6))
    np.testing.assert_allclose((left - right).evaluate((2, 3)), (-1, 12))
    np.testing.assert_allclose((-left).evaluate((2, 3)), (-3, -9))
    np.testing.assert_allclose((2 * left).evaluate((2, 3)), (6, 18))
    np.testing.assert_allclose((left * 2).evaluate((2, 3)), (6, 18))
    np.testing.assert_allclose((1j * left).evaluate((2, 3)), (3j, 9j))
    assert left.is_complex is False


def test_componentwise_arithmetic_validates_descriptors() -> None:
    left = TpsaMap(_d())
    other = Descriptor(variables=('alg_other_q', 'alg_other_p'), order=5)
    with pytest.raises(ValueError, match='descriptor'):
        _ = left + TpsaMap(other)
    with pytest.raises(ValueError, match='descriptor'):
        _ = left - TpsaMap(other)


def test_native_norm_and_degree_properties() -> None:
    d = _d()
    q, p = d.vars()
    mapping = TpsaMap([q + 2, 3 * p * p])
    assert mapping.norm() >= 0
    assert mapping.degree == 2
    assert TpsaMap([d.zero(), d.zero()]).norm() == pytest.approx(0)


def test_operations_preserve_original_map_and_coefficient_types() -> None:
    d = _d()
    q, p = d.vars()
    m = TpsaMap([q + p * p, p])
    original = m.evaluate((2, 3)).copy()
    _ = m.inverse()
    _ = m.partial_inverse([1, 0])
    _ = m.translate((1, 0))
    _ = m.exp_poisson(d.zero())
    np.testing.assert_allclose(m.evaluate((2, 3)), original)
    assert not m.is_complex


def test_native_translation_shifts_inputs_not_outputs_and_keeps_parameter_terms() -> None:
    d = _d(params=True)
    q, p = d.vars()
    k = d.param('alg_k')
    mapping = TpsaMap([q * q + k * q + k * k, p + q * k])

    translated = mapping.translate((2, -1))

    assert isinstance(translated, TpsaMap)
    assert translated.descriptor is d
    assert translated.is_complex is False
    np.testing.assert_allclose(translated.evaluate((3, 4), parameters=(5,)), (75, 28))
    np.testing.assert_allclose(mapping.evaluate((3, 4), parameters=(5,)), (49, 19))


def test_native_translation_agrees_with_explicit_polynomial_composition() -> None:
    d = _d(params=True)
    q, p = d.vars()
    k = d.param('alg_k')
    mapping = TpsaMap([q**2 + k * p, p**2 + q * k])
    translated = mapping.translate((1, -2))
    substitution = TpsaMap([q + 1, p - 2])
    composed = mapping.compose(substitution)

    for point, param in [((0.0, 0.0), 2.0), ((2.0, 3.0), -1.0)]:
        np.testing.assert_allclose(
            translated.evaluate(point, parameters=(param,)),
            composed.evaluate(point, parameters=(param,)),
        )


def test_native_translation_preserves_two_independent_parameters() -> None:
    d = _d(parameters=('alg_k', 'native_reg_l'))
    q, p = d.vars()
    k, ell = d.params()
    mapping = TpsaMap([q + k * ell, p + k])
    result = mapping.translate((1, 2))

    np.testing.assert_allclose(result.evaluate((2, 3), parameters=(4, 5)), (23, 9))
    assert result.num_params == 2


def test_native_translation_promotes_complex_offsets_without_mutating_input() -> None:
    d = _d(params=True)
    q, p = d.vars()
    k = d.param('alg_k')
    mapping = TpsaMap([q * q + k * q, p + k])

    translated = mapping.translate((1j, 0))

    assert translated.is_complex
    assert all(isinstance(t, ComplexTpsa) for t in translated.flat)
    np.testing.assert_allclose(translated.evaluate((2, 3), parameters=(5,)), (13 + 9j, 8))
    assert not mapping.is_complex
    np.testing.assert_allclose(mapping.evaluate((2, 3), parameters=(5,)), (14, 8))


def test_native_translation_accepts_array_offsets_and_complex_map() -> None:
    d = _d(params=True)
    q, p = d.vars()
    k = d.param('alg_k')
    mapping = TpsaMap([1j * k * q + p, p])
    translated = mapping.translate(np.asarray([1, 2]))

    assert translated.is_complex
    np.testing.assert_allclose(translated.evaluate((2, 3), parameters=(4,)), (5 + 12j, 5))


def test_native_translation_rejects_bad_offset_count() -> None:
    mapping = TpsaMap(_d())
    with pytest.raises(ValueError, match='offsets'):
        mapping.translate((1,))
    with pytest.raises(ValueError, match='offsets'):
        mapping.translate((1, 2, 3))


def test_hamiltonian_field_to_generator_roundtrip_discards_parameter_only_term() -> None:
    d = _d(params=True)
    q, p = d.vars()
    k = d.param('alg_k')
    generator = q * p + k * q + k * k
    field = TpsaMap.from_generator(generator)

    recovered = field.to_generator()
    assert recovered.descriptor is d
    assert recovered.get((0, 0, 1)) == 0
    assert recovered.get((0, 0, 2)) == 0
    np.testing.assert_allclose(TpsaTensor([recovered]).evaluate((2, 3), parameters=(4,)), (14,))
    rebuilt = TpsaMap.from_generator(recovered)
    for original, restored in zip(field.flat, rebuilt.flat, strict=True):
        assert original == restored


def test_complex_hamiltonian_field_roundtrip_with_parameter() -> None:
    d = _d(params=True)
    q, p = d.vars()
    k = d.param('alg_k')
    generator = (1 + 1j) * q * p + 2j * k * q + k * k
    field = TpsaMap.from_generator(generator)
    recovered = field.to_generator()

    assert isinstance(recovered, ComplexTpsa)
    assert recovered.get((0, 0, 1)) == 0
    assert recovered.get((0, 0, 2)) == 0
    np.testing.assert_allclose(
        TpsaTensor([recovered]).evaluate((2, 3), parameters=(4,)), (6 + 22j,)
    )
    for original, restored in zip(field.flat, TpsaMap.from_generator(recovered).flat, strict=True):
        assert original == restored


def test_to_generator_clears_mixed_parameter_only_monomials() -> None:
    d = _d(parameters=('alg_k', 'alg_l'))
    q, p = d.vars()
    k, ell = d.params()
    generator = q * p + k * q + ell * p + k * ell

    recovered = TpsaMap.from_generator(generator).to_generator()
    assert recovered.get((0, 0, 1, 0)) == 0
    assert recovered.get((0, 0, 0, 1)) == 0
    assert recovered.get((0, 0, 1, 1)) == 0
    assert recovered.get((0, 0, 2, 0)) == 0
    assert recovered.get((0, 0, 0, 2)) == 0
    np.testing.assert_allclose(
        TpsaTensor([recovered]).evaluate((2, 3), parameters=(4, 5)),
        (6 + 8 + 15,),
    )


def test_to_generator_rejects_nonhamiltonian_field_by_default() -> None:
    d = _d()
    q, _ = d.vars()
    field = TpsaMap([q, d.zero()])  # Divergence != 0 in the (q,p) plane.

    with pytest.raises(ValueError, match='Hamiltonian|non-integrable'):
        field.to_generator()

    # Native fld2vec still computes a formal projection if requested.
    projected = field.to_generator(validate=False)
    assert projected.descriptor is d


def test_to_generator_rejects_complex_nonhamiltonian_field() -> None:
    d = _d()
    q, _ = d.vars()
    field = TpsaMap([1j * q, d.complex_zero()])

    with pytest.raises(ValueError, match='Hamiltonian|non-integrable'):
        field.to_generator()


def test_to_generator_detects_generator_order_beyond_descriptor() -> None:
    d = _d(parameters=(), order=3)
    _, p = d.vars()
    # A Hamiltonian primitive needs degree four, but the descriptor only
    # supports order three. The native integral cannot be a true round trip.
    field = TpsaMap([p**3, d.zero()])

    with pytest.raises(ValueError, match='truncation|higher order'):
        field.to_generator()
    assert field.to_generator(validate=False).descriptor is d


def test_to_generator_rejects_invalid_tolerances() -> None:
    m = TpsaMap(_d())
    with pytest.raises(ValueError, match='non-negative'):
        m.to_generator(atol=-1)
    with pytest.raises(ValueError, match='non-negative'):
        m.to_generator(rtol=-1)


def test_scalar_exponential_matches_negative_native_hamiltonian_field() -> None:
    d = _d(parameters=())
    q, _ = d.vars()
    generator = q * q / 2
    field = TpsaMap.from_generator(generator)
    identity = TpsaMap(d)
    scalar_action = identity.exp_poisson(generator)
    field_action = identity.exp_poisson(-field)
    opposite_action = identity.exp_poisson(field)

    np.testing.assert_allclose(scalar_action.evaluate((2, 3)), field_action.evaluate((2, 3)))
    assert not np.allclose(scalar_action.evaluate((2, 3)), opposite_action.evaluate((2, 3)))


def test_field_to_vector_compatibility_alias_uses_validation() -> None:
    d = _d(parameters=())
    q, _ = d.vars()
    m = TpsaMap([q, d.zero()])
    with pytest.raises(ValueError, match='Hamiltonian|non-integrable'):
        m.to_generator()
    assert m.to_generator(validate=False).descriptor is d


def test_align_descriptors_promotes_converted_operands() -> None:
    d = _d(params=True)
    q, p = d.vars()
    real = TpsaTensor([q + d.param('alg_k'), p])
    complex_map = TpsaMap([q + 1j * p, p])
    result = real.compose(complex_map)
    assert result.is_complex
    assert all(isinstance(tpsa, ComplexTpsa) for tpsa in result.flat)
    np.testing.assert_allclose(result.evaluate((2, 3), parameters=(4,)), (6 + 3j, 3))


def test_scalar_generator_promotion_is_not_lost_by_descriptor_alignment() -> None:
    d = _d(params=True)
    q, p = d.vars()
    k = d.param('alg_k')
    result = TpsaMap([q + k, p]).exp_poisson(1j * q * q / 2)
    assert result.is_complex
    assert all(isinstance(tpsa, ComplexTpsa) for tpsa in result.flat)


def test_tensor_arithmetic_preserves_complex_promotion_and_both_operands() -> None:
    d = _d()
    q, p = d.vars()
    left = TpsaMap([q, p])
    right = TpsaMap([1j * q, p])
    added = left + right
    subtracted = left - right
    reversed_subtraction = right.__rsub__(left)
    for candidate in (added, subtracted, reversed_subtraction):
        assert candidate.is_complex
    np.testing.assert_allclose(added.evaluate((2, 3)), (2 + 2j, 6))
    np.testing.assert_allclose(subtracted.evaluate((2, 3)), (2 - 2j, 0))
    np.testing.assert_allclose(reversed_subtraction.evaluate((2, 3)), (2 - 2j, 0))
    assert not left.is_complex
    assert right.is_complex


def test_translate_accepts_zero_imaginary_offsets_as_real() -> None:
    d = _d()
    q, p = d.vars()
    m = TpsaMap([q * q, p])
    translated = m.translate((2 + 0j, -1 + 0j))
    assert not translated.is_complex
    np.testing.assert_allclose(translated.evaluate((3, 4)), (25, 3))
