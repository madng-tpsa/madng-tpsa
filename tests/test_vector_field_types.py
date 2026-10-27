"""Type semantics for TPSA coordinate maps and TPSA vector fields.

These tests assume ``TpsaVectorField(TpsaMap)`` is a minimal semantic subclass
and that ``from_generator``, ``lie_bracket``, and ``log_fgrad`` construct it.
They also exercise preservation of map subclasses through ``_new_like``.
"""

from __future__ import annotations

import numpy as np
import pytest

from madng_tpsa import ComplexTpsa, Descriptor, Tpsa, TpsaMap, TpsaVector, TpsaVectorField


class _CustomField(TpsaVectorField[Tpsa]):
    """Stand-in for a downstream subclass."""


class _CustomMap(TpsaMap[Tpsa]):
    """Stand-in for a downstream subclass such as Xtrack's ParticlesTpsa."""


def _descriptor() -> Descriptor:
    return Descriptor(variables=('vf_q', 'vf_p'), order=5)


def test_from_generator_returns_real_vector_field() -> None:
    d = _descriptor()
    q, p = d.vars()
    h = (q**2 + p**2) / 2

    field = TpsaVectorField.from_generator(h)

    assert type(field) is TpsaVectorField
    assert field.shape == (2,)
    assert field.component_names == d.variables
    assert not field.is_complex
    np.testing.assert_allclose(field.evaluate((2, 3)), (-3, 2))
    assert field.to_generator() == h


def test_from_generator_returns_complex_vector_field() -> None:
    d = _descriptor()
    q, p = d.vars()
    h = 1j * (q**2 + p**2) / 2

    field = TpsaVectorField.from_generator(h)

    assert type(field) is TpsaVectorField
    assert field.is_complex
    assert all(isinstance(component, ComplexTpsa) for component in field.components)
    np.testing.assert_allclose(field.evaluate((2, 3)), (-3j, 2j))
    assert field.to_generator() == h


def test_from_generator_does_not_inherit_arbitrary_map_subclass() -> None:
    d = _descriptor()
    q, p = d.vars()
    field = _CustomField.from_generator((q**2 + p**2) / 2)
    assert type(field) is TpsaVectorField
    assert not isinstance(field, TpsaMap)


def test_lie_bracket_returns_vector_field() -> None:
    d = _descriptor()
    q, p = d.vars()
    f = TpsaVectorField([q**2, p])
    g = TpsaVectorField([p, q])

    bracket_fg = f.lie_bracket(g)
    bracket_gf = g.lie_bracket(f)

    assert type(bracket_fg) is TpsaVectorField
    assert type(bracket_gf) is TpsaVectorField
    np.testing.assert_allclose(
        bracket_fg.evaluate((0.2, -0.3)),
        -bracket_gf.evaluate((0.2, -0.3)),
        atol=1e-12,
    )


def test_lie_bracket_promotes_to_complex_vector_field() -> None:
    d = _descriptor()
    q, p = d.vars()
    real = TpsaVectorField([q * p, p])
    complex_field = TpsaVectorField([1j * p, q])

    result = real.lie_bracket(complex_field)

    assert type(result) is TpsaVectorField
    assert result.is_complex
    assert all(isinstance(component, ComplexTpsa) for component in result.components)


def test_log_fgrad_returns_field_and_exponential_returns_map() -> None:
    d = _descriptor()
    q, p = d.vars()
    mapping = _CustomMap([q, p - 0.015 * q**2])

    field = mapping.log_fgrad()
    reconstructed = TpsaMap(d).exp_fgrad(field)

    assert type(field) is TpsaVectorField
    assert type(reconstructed) is TpsaMap
    np.testing.assert_allclose(
        reconstructed.evaluate((0.1, -0.2)),
        mapping.evaluate((0.1, -0.2)),
        atol=1e-10,
    )


def test_log_fgrad_preserves_complex_coefficients_in_field() -> None:
    d = _descriptor()
    q, p = d.vars()
    mapping = TpsaMap([q, p + (0.01 + 0.005j) * q**2])

    field = mapping.log_fgrad()

    assert type(field) is TpsaVectorField
    assert field.is_complex
    assert all(isinstance(component, ComplexTpsa) for component in field.components)


@pytest.mark.parametrize('operation', ('copy', 'truncate', 'homogeneous', 'to_complex'))
def test_vector_field_container_operations_preserve_type(operation: str) -> None:
    d = _descriptor()
    q, p = d.vars()
    field = TpsaVectorField([q + p**2, p - q**2])

    if operation == 'truncate':
        result = field.truncate(3)
    elif operation == 'homogeneous':
        result = field.homogeneous(2)
    else:
        result = getattr(field, operation)()

    assert type(result) is TpsaVectorField
    assert result.shape == (2,)
    if operation == 'to_complex':
        assert result.is_complex
        assert not field.is_complex


def test_vector_field_compose_is_componentwise_substitution() -> None:
    """This is F(M(z)) in coordinates, not the geometric pushforward M_*F."""
    d = _descriptor()
    q, p = d.vars()
    field = TpsaVectorField([q**2, p])
    mapping = TpsaMap([q + p, 2 * p])

    substituted = field.compose(mapping)

    assert type(substituted) is TpsaVector
    assert type(substituted) is not TpsaVectorField
    assert type(substituted) is not TpsaMap
    np.testing.assert_allclose(substituted.evaluate((2, 3)), (25, 6))


@pytest.mark.parametrize('operation', ('composition_power', 'translate', 'exp_poisson'))
def test_finite_map_operations_preserve_map_subclass(operation: str) -> None:
    d = _descriptor()
    q, p = d.vars()
    mapping = _CustomMap([q, p + 0.01 * q**2])

    if operation == 'composition_power':
        result = mapping.composition_power(2)
    elif operation == 'translate':
        result = mapping.translate((0.1, -0.2))
    else:
        result = mapping.exp_poisson(q**2 / 2)

    assert type(result) is _CustomMap
    assert not isinstance(result, TpsaVectorField)


def test_zero_composition_power_preserves_map_subclass() -> None:
    d = _descriptor()
    mapping = _CustomMap(d)

    result = mapping.composition_power(0)

    assert type(result) is _CustomMap
    assert not isinstance(result, TpsaVectorField)
    np.testing.assert_allclose(result.evaluate((0.2, -0.3)), (0.2, -0.3))


def test_descriptor_extension_distinguishes_map_and_field():
    d1 = Descriptor(variables=('x', 'px'), order=3)
    d2 = Descriptor(variables=('x', 'px', 'y', 'py'), order=3)

    map_ = TpsaMap(d1)
    field = TpsaVectorField(d1)

    map2 = map_.extend_descriptor(d2)
    field2 = field.extend_descriptor(d2)

    assert isinstance(map2, TpsaMap)
    assert isinstance(field2, TpsaVectorField)

    # Map: newly introduced outputs are identity coordinates.
    assert map2.y == d2.var('y')
    assert map2.py == d2.var('py')

    # Field: newly introduced outputs are zero.
    assert field2.y.is_zero()
    assert field2.py.is_zero()
