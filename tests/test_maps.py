import numpy as np
import pytest

import madng_tpsa
from madng_tpsa import (
    ComplexTpsa,
    Descriptor,
    TpsaMap,
    compose,
    evaluate,
    exp_poisson,
    field_to_vector,
    inverse,
    lie_bracket,
    log_generator,
    log_poisson,
    partial_inverse,
    translate,
    vector_to_field,
)


def assert_series_equal(actual, expected):
    assert actual.descriptor is expected.descriptor
    assert actual.to_dict() == pytest.approx(expected.to_dict())


def assert_map_equal(actual, expected):
    assert len(actual) == len(expected)
    for aa, ee in zip(actual, expected, strict=True):
        assert_series_equal(aa, ee)


@pytest.fixture
def descriptor():
    return Descriptor(2, 5)


@pytest.fixture
def nonlinear_map(descriptor):
    x, y = descriptor.vars()
    return TpsaMap(
        [
            x + 2 * y + x * y + y**2,
            y - x**2 + 3 * x * y,
        ],
        coord_names=('x', 'y'),
    )


# ---------------------------------------------------------------------------
# Construction and container behaviour
# ---------------------------------------------------------------------------


def test_map_constructs_from_sequence(descriptor):
    x, y = descriptor.vars()
    map_ = TpsaMap([x, y])
    assert len(map_) == 2
    assert map_.descriptor is descriptor
    assert map_.coord_names == descriptor.var_labels
    assert map_[0] is x
    assert map_[1] is y


def test_map_constructs_from_mapping(descriptor):
    x, y = descriptor.vars()
    map_ = TpsaMap({'qx': x, 'qy': y})
    assert map_.coord_names == ('qx', 'qy')
    assert map_['qx'] is x
    assert map_['qy'] is y


def test_map_mapping_can_reorder_with_coord_names(descriptor):
    x, y = descriptor.vars()
    map_ = TpsaMap({'x': x, 'y': y}, coord_names=('y', 'x'))
    assert map_.coord_names == ('y', 'x')
    assert map_[0] is y
    assert map_[1] is x


def test_map_default_names_follow_descriptor_labels():
    descriptor = Descriptor(variables=['q', 'p'], order=3)
    q, p = descriptor.vars()
    map_ = TpsaMap([q, p])
    assert map_.coord_names == ('q', 'p')


def test_map_rejects_empty_sequence():
    with pytest.raises(ValueError, match='cannot be empty'):
        TpsaMap([])


def test_map_rejects_non_series_component(descriptor):
    x, _ = descriptor.vars()
    with pytest.raises(TypeError, match='Map components must be'):
        TpsaMap([x, 1.0])  # ty:ignore[invalid-argument-type]


def test_map_rejects_different_descriptors():
    x = Descriptor(1, 3).var(1)
    y = Descriptor(2, 3).var(1)
    with pytest.raises(ValueError, match='same Descriptor'):
        TpsaMap([x, y])


def test_map_rejects_too_many_components(descriptor):
    x, y = descriptor.vars()
    with pytest.raises(ValueError, match='only 2 variables'):
        TpsaMap([x, y, x])


def test_map_rejects_wrong_number_of_names(descriptor):
    x, y = descriptor.vars()
    with pytest.raises(ValueError, match='one name per map component'):
        TpsaMap([x, y], coord_names=('x',))


def test_map_rejects_duplicate_names(descriptor):
    x, y = descriptor.vars()
    with pytest.raises(ValueError, match='must be unique'):
        TpsaMap([x, y], coord_names=('x', 'x'))


def test_map_identity(descriptor):
    identity = TpsaMap.identity(descriptor)
    assert_map_equal(identity.coords, descriptor.vars())


def test_map_identity_with_expansion_values(descriptor):
    identity = TpsaMap.identity(descriptor, values=[1.5, -2.0])
    np.testing.assert_allclose(identity.const_part, [1.5, -2.0])
    np.testing.assert_allclose(identity.jacobian(), np.eye(2))


def test_map_from_monomial_coeffs_sequence(descriptor):
    map_ = TpsaMap.from_monomial_coeffs(
        descriptor,
        [
            {(1, 0): 1.0, (0, 2): 2.0},
            {(0, 1): 1.0, (2, 0): -3.0},
        ],
        coord_names=('x', 'y'),
    )
    assert map_['x'].to_dict() == pytest.approx({(1, 0): 1.0, (0, 2): 2.0})
    assert map_['y'].to_dict() == pytest.approx({(0, 1): 1.0, (2, 0): -3.0})


def test_map_from_monomial_coeffs_mapping(descriptor):
    map_ = TpsaMap.from_monomial_coeffs(
        descriptor,
        {'qx': {(1, 0): 1.0}, 'qy': {(0, 1): 1.0}},
    )
    assert map_.coord_names == ('qx', 'qy')
    assert map_['qx'][(1, 0)] == 1.0
    assert map_['qy'][(0, 1)] == 1.0


def test_map_promotes_mixed_real_complex_components(descriptor):
    x, y = descriptor.vars()
    complex_y = ComplexTpsa.from_tpsa(y) + 1j
    map_ = TpsaMap([x, complex_y])
    assert map_.is_complex
    assert all(isinstance(component, ComplexTpsa) for component in map_)


def test_map_const_part_real(descriptor):
    x, y = descriptor.vars(values=[1.0, -2.0])
    map_ = TpsaMap([x, y])
    np.testing.assert_allclose(map_.const_part, [1.0, -2.0])


def test_map_const_part_complex(descriptor):
    x, y = descriptor.vars()
    map_ = TpsaMap(
        [
            ComplexTpsa.from_tpsa(x) + 1j,
            ComplexTpsa.from_tpsa(y) - 2j,
        ]
    )
    assert map_.is_complex
    np.testing.assert_allclose(map_.const_part, [1j, -2j])


def test_map_getitem_and_attribute_access():
    descriptor = Descriptor(variables=['q', 'p'], order=3)
    q, p = descriptor.vars()
    map_ = TpsaMap([q, p])
    assert map_[0] is q
    assert map_[-1] is p
    assert map_['q'] is q
    assert map_.q is q
    with pytest.raises(KeyError):
        map_['missing']
    with pytest.raises(AttributeError):
        _ = map_.missing


def test_map_copy_is_independent(nonlinear_map):
    copied = nonlinear_map.copy()
    copied[0].set_const_part(42.0)
    assert copied[0].const_part == 42.0
    assert nonlinear_map[0].const_part == 0.0


def test_map_repr_contains_coordinate_names(descriptor):
    map_ = TpsaMap.identity(descriptor, coord_names=('qx', 'qy'))
    representation = repr(map_)
    assert representation.startswith('TpsaMap(')
    assert 'qx=' in representation
    assert 'qy=' in representation


# ---------------------------------------------------------------------------
# Coefficient access
# ---------------------------------------------------------------------------


def test_map_monomial_coeffs_all_coordinates(nonlinear_map):
    coefficients = nonlinear_map.monomial_coeffs()
    assert tuple(coefficients) == ('x', 'y')
    assert coefficients['x'] == pytest.approx(nonlinear_map['x'].to_dict())
    assert coefficients['y'] == pytest.approx(nonlinear_map['y'].to_dict())


def test_map_monomial_coeffs_one_coordinate(nonlinear_map):
    assert nonlinear_map.monomial_coeffs('x') == pytest.approx(nonlinear_map['x'].to_dict())


def test_map_coefficient_and_set_coefficient(nonlinear_map):
    assert nonlinear_map.coefficient('x', (1, 1)) == pytest.approx(1.0)
    nonlinear_map.set_coefficient('x', (0, 3), 2.5)
    assert nonlinear_map['x'][(0, 3)] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_compose_uses_left_after_right(descriptor):
    x, y = descriptor.vars()
    left = (x + y**2, y)
    right = (x, y + x**2)
    result = compose(left, right)
    expected = (x + (y + x**2) ** 2, y + x**2)
    assert_map_equal(result, expected)


def test_map_compose_and_matmul_match_function(descriptor):
    x, y = descriptor.vars()
    left = TpsaMap([x + y**2, y])
    right = TpsaMap([x, y + x**2])
    expected = compose(left.coords, right.coords)
    assert_map_equal(left.compose(right).coords, expected)
    assert_map_equal((left @ right).coords, expected)


def test_compose_partial_left_map(descriptor):
    x, y = descriptor.vars()
    result = compose([x + y**2], [x, y + x])
    assert len(result) == 1
    assert_series_equal(result[0], x + (y + x) ** 2)


def test_compose_requires_full_right_map(descriptor):
    x, y = descriptor.vars()
    with pytest.raises(ValueError, match='Right map must have 2 components'):
        compose([x, y], [x])


def test_compose_rejects_different_descriptors():
    left = Descriptor(2, 3).vars()
    right = Descriptor(2, 4).vars()
    with pytest.raises(ValueError, match='same Descriptor'):
        compose(left, right)


def test_compose_preserves_parameters():
    descriptor = Descriptor(variables=['x', 'y'], order=4, params=['k'], param_order=2)
    x, y = descriptor.vars()
    k = descriptor.param('k')
    result = compose((x + k * y, y + k), (x + y, y))
    assert_map_equal(result, (x + y + k * y, y + k))


def test_compose_promotes_real_and_complex(descriptor):
    x, y = descriptor.vars()
    right = (
        ComplexTpsa.from_tpsa(x) + 1j,
        ComplexTpsa.from_tpsa(y),
    )
    result = compose((x + y, y), right)
    assert all(isinstance(component, ComplexTpsa) for component in result)
    assert result[0].const_part == pytest.approx(1j)


# ---------------------------------------------------------------------------
# Evaluation and Jacobian
# ---------------------------------------------------------------------------


def test_evaluate_real_map(descriptor):
    x, y = descriptor.vars()
    np.testing.assert_allclose(evaluate((x**2 + y, x * y), [2.0, 3.0]), [7.0, 6.0])


def test_map_evaluate_matches_function(descriptor):
    x, y = descriptor.vars()
    map_ = TpsaMap([x**2 + y, x * y])
    np.testing.assert_allclose(
        map_.evaluate([2.0, 3.0]),
        evaluate(map_.coords, [2.0, 3.0]),
    )


def test_evaluate_with_parameters():
    descriptor = Descriptor(variables=['x', 'y'], order=3, params=['k'])
    x, y = descriptor.vars()
    k = descriptor.param('k')
    result = evaluate((x + k * y, y + k), [2.0, 3.0], parameters=[4.0])
    np.testing.assert_allclose(result, [14.0, 7.0])


def test_evaluate_parameters_default_to_zero():
    descriptor = Descriptor(variables=['x'], order=3, params=['k'])
    x = descriptor.var('x')
    k = descriptor.param('k')
    np.testing.assert_allclose(evaluate([x + 2 * k], [3.0]), [3.0])


def test_evaluate_real_map_at_complex_point_promotes(descriptor):
    x, y = descriptor.vars()
    result = evaluate([x + y, x - y], [1 + 2j, 3 - 1j])
    np.testing.assert_allclose(result, [4 + 1j, -2 + 3j])


def test_evaluate_complex_map(descriptor):
    x, y = descriptor.vars()
    map_ = TpsaMap(
        [
            ComplexTpsa.from_tpsa(x) + 1j * y,
            ComplexTpsa.from_tpsa(y),
        ]
    )
    np.testing.assert_allclose(map_.evaluate([2.0, 3.0]), [2 + 3j, 3])


def test_evaluate_rejects_wrong_lengths(descriptor):
    x, y = descriptor.vars()
    with pytest.raises(ValueError, match='Expected 2 coordinate values'):
        evaluate([x, y], [1.0])

    param_descriptor = Descriptor(variables=['x'], order=2, params=['k'])
    px = param_descriptor.var('x')
    with pytest.raises(ValueError, match='Expected 1 parameter values'):
        evaluate([px], [0.0], parameters=[1.0, 2.0])


def test_jacobian_at_expansion_point(descriptor):
    x, y = descriptor.vars()
    map_ = TpsaMap(
        [
            2 * x + 3 * y + x * y,
            -x + 4 * y + x**2,
        ]
    )
    np.testing.assert_allclose(map_.jacobian(), [[2.0, 3.0], [-1.0, 4.0]])


def test_jacobian_at_numerical_point(descriptor):
    x, y = descriptor.vars()
    map_ = TpsaMap([x**2 + x * y, y**2 - x * y])
    np.testing.assert_allclose(map_.jacobian([2.0, 3.0]), [[7.0, 2.0], [-3.0, 4.0]])


def test_jacobian_with_parameter():
    descriptor = Descriptor(variables=['x', 'y'], order=3, params=['k'])
    x, y = descriptor.vars()
    k = descriptor.param('k')
    map_ = TpsaMap([k * x + y, x + k * y])
    np.testing.assert_allclose(
        map_.jacobian([2.0, 3.0], parameters=[4.0]),
        [[4.0, 1.0], [1.0, 4.0]],
    )


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def test_translate_identity_map(descriptor):
    result = translate(descriptor.vars(), [1.5, -2.0])
    assert result[0].const_part == pytest.approx(1.5)
    assert result[1].const_part == pytest.approx(-2.0)
    assert result[0].grad() == pytest.approx([1.0, 0.0])
    assert result[1].grad() == pytest.approx([0.0, 1.0])


def test_translate_nonlinear_map(descriptor):
    x, y = descriptor.vars()
    result = translate((x**2 + y, x * y), [1.0, 2.0])
    assert_map_equal(
        result,
        ((x + 1) ** 2 + (y + 2), (x + 1) * (y + 2)),
    )


def test_map_translate_matches_function(nonlinear_map):
    via_method = nonlinear_map.translate([0.5, -1.0])
    via_function = translate(nonlinear_map.coords, [0.5, -1.0])
    assert_map_equal(via_method.coords, via_function)


def test_translate_complex_offsets_promotes(descriptor):
    result = translate(descriptor.vars(), [1j, 0.0])
    assert all(isinstance(component, ComplexTpsa) for component in result)
    assert result[0].const_part == pytest.approx(1j)


def test_translate_preserves_parameter_identity():
    descriptor = Descriptor(variables=['x'], order=3, params=['k'])
    x = descriptor.var('x')
    k = descriptor.param('k')
    result = translate([x + k], [2.0])
    assert_series_equal(result[0], x + 2 + k)


def test_translate_rejects_wrong_offset_count(descriptor):
    with pytest.raises(ValueError, match='Expected 2 offsets'):
        translate(descriptor.vars(), [1.0])


# ---------------------------------------------------------------------------
# Inversion
# ---------------------------------------------------------------------------


def test_inverse_simple_triangular_map(descriptor):
    x, y = descriptor.vars()
    result = inverse((x + y**2, y))
    assert_map_equal(result, (x - y**2, y))


def test_inverse_composes_to_identity(descriptor):
    x, y = descriptor.vars()
    map_ = (x + y**2, y + x**3)
    inverse_map = inverse(map_)
    assert_map_equal(compose(map_, inverse_map), (x, y))


def test_map_inverse_matches_function(descriptor):
    x, y = descriptor.vars()
    map_ = TpsaMap([x + y**2, y])
    assert_map_equal(map_.inverse().coords, inverse(map_.coords))


def test_inverse_with_parameter():
    descriptor = Descriptor(variables=['x', 'y'], order=4, params=['k'], param_order=2)
    x, y = descriptor.vars()
    k = descriptor.param('k')
    result = inverse((x + k * y, y))
    assert_map_equal(result, (x - k * y, y))


def test_inverse_requires_full_map(descriptor):
    x, _ = descriptor.vars()
    with pytest.raises(ValueError, match='full 2-component map'):
        inverse([x])


def test_inverse_singular_map_raises_tpsa_error(descriptor):
    x, _ = descriptor.vars()
    with pytest.raises(madng_tpsa.TpsaError):
        inverse([x, descriptor.zero()])


# ---------------------------------------------------------------------------
# Partial inversion: validation only until select semantics are pinned down.
# ---------------------------------------------------------------------------


def test_partial_inverse_requires_full_map(descriptor):
    x, _ = descriptor.vars()
    with pytest.raises(ValueError, match='full 2-component map'):
        partial_inverse([x], [1, 0])


def test_partial_inverse_requires_one_selection_per_variable(descriptor):
    with pytest.raises(ValueError, match='select must contain 2 entries'):
        partial_inverse(descriptor.vars(), [1])


# ---------------------------------------------------------------------------
# Hamiltonian fields and Lie brackets
# ---------------------------------------------------------------------------


def test_vector_to_field_quadratic_generator(descriptor):
    q, p = descriptor.vars()
    field = vector_to_field((q**2 + p**2) / 2)
    assert_map_equal(field, (-p, q))


def test_field_to_vector_round_trip(descriptor):
    q, p = descriptor.vars()
    generator = q**3 / 3 + q * p**2 + p**4 / 4
    recovered = field_to_vector(vector_to_field(generator))
    assert_series_equal(recovered, generator)


def test_vector_to_field_complex(descriptor):
    q, p = descriptor.vars()
    generator = 1j * ComplexTpsa.from_tpsa(q * p)
    field = vector_to_field(generator)
    assert all(isinstance(component, ComplexTpsa) for component in field)


def test_vector_to_field_validation():
    with pytest.raises(TypeError, match='generator must be'):
        vector_to_field(1.0)  # ty:ignore[invalid-argument-type]

    descriptor = Descriptor(3, 3)
    with pytest.raises(ValueError, match='even number of variables'):
        vector_to_field(descriptor.var(1) ** 2)


def test_field_to_vector_requires_full_field(descriptor):
    q, _ = descriptor.vars()
    with pytest.raises(ValueError, match='Field must have 2 components'):
        field_to_vector([q])


def test_lie_bracket_with_itself_is_zero(descriptor):
    q, p = descriptor.vars()
    field = (q * p, q**2 - p**2)
    result = lie_bracket(field, field)
    assert all(component.is_zero() for component in result)


def test_lie_bracket_antisymmetry(descriptor):
    q, p = descriptor.vars()
    left = (q * p, q**2)
    right = (p**2, q * p)
    ab = lie_bracket(left, right)
    ba = lie_bracket(right, left)
    for aa, bb in zip(ab, ba, strict=True):
        assert_series_equal(aa, -bb)


def test_map_lie_bracket_matches_function(descriptor):
    q, p = descriptor.vars()
    left = TpsaMap([q * p, q**2])
    right = TpsaMap([p**2, q * p])
    assert_map_equal(
        left.lie_bracket(right).coords,
        lie_bracket(left.coords, right.coords),
    )


# ---------------------------------------------------------------------------
# Poisson exponential/logarithm
# ---------------------------------------------------------------------------


def test_exp_poisson_zero_generator_is_identity(descriptor):
    identity = descriptor.vars()
    result = exp_poisson(identity, descriptor.zero())
    assert_map_equal(result, identity)


def test_exp_poisson_zero_field_is_identity(descriptor):
    identity = descriptor.vars()
    zero = descriptor.zero()
    assert_map_equal(exp_poisson(identity, (zero, zero)), identity)


def test_exp_poisson_scalar_and_explicit_field_agree(descriptor):
    q, _ = descriptor.vars()
    identity = descriptor.vars()
    generator = q**3 / 3
    field = tuple(-component for component in vector_to_field(generator))
    assert_map_equal(
        exp_poisson(identity, generator),
        exp_poisson(identity, field),
    )


def test_map_exp_poisson_matches_function(descriptor):
    q, _ = descriptor.vars()
    map_ = TpsaMap.identity(descriptor)
    generator = q**3 / 3
    assert_map_equal(
        map_.exp_poisson(generator).coords,
        exp_poisson(map_.coords, generator),
    )


def test_exp_poisson_validation(descriptor):
    q, _ = descriptor.vars()
    with pytest.raises(ValueError, match='full 2-component map'):
        exp_poisson([q], q**2)

    other_generator = Descriptor(2, 4).var(1) ** 2
    with pytest.raises(ValueError, match='share the same Descriptor'):
        exp_poisson(descriptor.vars(), other_generator)


def test_log_poisson_identity_map_returns_zero_field(descriptor):
    result = log_poisson(descriptor.vars())
    assert all(component.is_zero() for component in result)


def test_log_generator_identity_map_is_zero(descriptor):
    assert log_generator(descriptor.vars()).is_zero()


def test_map_log_methods_match_functions(descriptor):
    map_ = TpsaMap.identity(descriptor)
    assert_map_equal(map_.log_poisson().coords, log_poisson(map_.coords))
    assert_series_equal(map_.log_generator(), log_generator(map_.coords))


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


def test_map_order_is_max_allocated_order(descriptor):
    map_ = TpsaMap(
        [
            descriptor.var(1, order=2),
            descriptor.var(2, order=4),
        ]
    )
    assert map_.order == 4


def test_map_max_nonzero_order(nonlinear_map):
    assert nonlinear_map.max_nonzero_order == 2


def test_map_homogeneous(nonlinear_map):
    quadratic = nonlinear_map.homogeneous(2)
    assert quadratic.order == 2
    assert quadratic['x'].to_dict() == pytest.approx({(1, 1): 1.0, (0, 2): 1.0})
    assert quadratic['y'].to_dict() == pytest.approx({(2, 0): -1.0, (1, 1): 3.0})


def test_map_truncate(nonlinear_map):
    linear = nonlinear_map.truncate(1)
    assert linear.order == 1
    assert linear['x'].to_dict() == pytest.approx({(1, 0): 1.0, (0, 1): 2.0})
    assert linear['y'].to_dict() == pytest.approx({(0, 1): 1.0})


def test_map_clear_order(nonlinear_map):
    without_quadratic = nonlinear_map.clear_order(2)
    assert without_quadratic.order == nonlinear_map.order
    assert without_quadratic['x'].to_dict() == pytest.approx({(1, 0): 1.0, (0, 1): 2.0})
    assert without_quadratic['y'].to_dict() == pytest.approx({(0, 1): 1.0})


# ---------------------------------------------------------------------------
# Arithmetic and norm
# ---------------------------------------------------------------------------


def test_map_negation(descriptor):
    x, y = descriptor.vars()
    result = -TpsaMap([x, y])
    assert_series_equal(result[0], -x)
    assert_series_equal(result[1], -y)


def test_map_addition_and_subtraction(descriptor):
    x, y = descriptor.vars()
    left = TpsaMap([x + y, y])
    right = TpsaMap([y, x])
    added = left + right
    subtracted = left - right
    assert_series_equal(added[0], x + 2 * y)
    assert_series_equal(added[1], x + y)
    assert_series_equal(subtracted[0], x)
    assert_series_equal(subtracted[1], y - x)


def test_map_scalar_multiplication(descriptor):
    x, y = descriptor.vars()
    map_ = TpsaMap([x, y])
    left = 3 * map_
    right = map_ * 3
    assert_series_equal(left[0], 3 * x)
    assert_series_equal(left[1], 3 * y)
    assert_map_equal(left.coords, right.coords)


def test_map_addition_promotes_complex(descriptor):
    x, y = descriptor.vars()
    real = TpsaMap([x, y])
    complex_ = TpsaMap(
        [
            ComplexTpsa.from_tpsa(x) + 1j,
            ComplexTpsa.from_tpsa(y),
        ]
    )
    result = real + complex_
    assert result.is_complex
    assert all(isinstance(component, ComplexTpsa) for component in result)


def test_map_addition_requires_same_length(descriptor):
    x, y = descriptor.vars()
    with pytest.raises(ValueError, match='same number of components'):
        TpsaMap([x, y]) + TpsaMap([x])


def test_map_norm_is_sum_of_component_norms(nonlinear_map):
    expected = sum(component.norm() for component in nonlinear_map)
    assert nonlinear_map.norm() == pytest.approx(expected)


def test_set_const_part_real_map(descriptor):
    map_ = TpsaMap.identity(descriptor)
    map_.set_const_part([1.0, -2.0])
    np.testing.assert_allclose(map_.const_part, [1.0, -2.0])


def test_set_const_part_complex_map(descriptor):
    x, y = descriptor.vars()
    map_ = TpsaMap(
        [
            ComplexTpsa.from_tpsa(x),
            ComplexTpsa.from_tpsa(y),
        ]
    )
    map_.set_const_part([1 + 2j, 3 - 4j])
    np.testing.assert_allclose(map_.const_part, [1 + 2j, 3 - 4j])


def test_set_const_part_rejects_complex_for_real_map(descriptor):
    map_ = TpsaMap.identity(descriptor)
    with pytest.raises(TypeError, match='complex'):
        map_.set_const_part([1j, 0.0])


def test_map_coefficient_rejects_wrong_monomial_length_with_parameters():
    d = Descriptor(variables=['x', 'y'], order=3, params=['k'])
    map_ = TpsaMap.identity(d)

    with pytest.raises(ValueError, match='length 3'):
        map_.coefficient('x', (1, 0))

    with pytest.raises(ValueError, match='length 3'):
        map_.set_coefficient('x', (1, 0), 2.0)
