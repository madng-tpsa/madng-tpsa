"""Tests for simultaneous variable and parameter substitution."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from madng_tpsa import (
    ComplexTpsa,
    Descriptor,
    TpsaMap,
    TpsaTensor,
    TpsaVector,
    TpsaVectorField,
)


def _descriptor() -> Descriptor:
    return Descriptor(var_names=('x', 'px'), order=4, param_names=('k', 'h'), param_order=2)


@pytest.mark.parametrize('cls', [TpsaTensor, TpsaVector, TpsaMap, TpsaVectorField])
def test_substitute_positional_scalars(cls: type[TpsaTensor[Any]]) -> None:
    d = _descriptor()
    x, px = d.vars()
    k, h = d.params()
    tensor = cls([x + k, px + h])

    result = tensor.substitute(0.0)
    assert type(result) is cls
    assert result.shape == tensor.shape
    assert result.flat == (k, h)
    assert tensor.substitute(None, 0.0).flat == (x, px)
    assert tensor.substitute(0.0, 1.0).flat == (d.constant(1.0), d.constant(1.0))
    assert tensor.flat == (x + k, px + h)


def test_substitute_positional_sequences() -> None:
    d = _descriptor()
    x, px = d.vars()
    k, h = d.params()
    tensor = TpsaTensor([[x + k, px + h], [x * px, k * h]])

    assert tensor.substitute([px, x]).flat == (px + k, x + h, x * px, k * h)
    assert tensor.substitute(None, [h, k]).flat == (x + h, px + k, x * px, k * h)
    result = tensor.substitute((px, x), (h, k))
    assert result.shape == (2, 2)
    assert result.flat == (px + h, x + k, x * px, k * h)
    assert tensor.substitute(np.array([2.0, 3.0]), np.array([4.0, 5.0])).flat == (
        d.constant(6.0),
        d.constant(8.0),
        d.constant(6.0),
        d.constant(20.0),
    )


def test_substitute_broadcast_polynomial() -> None:
    d = _descriptor()
    x, px = d.vars()
    k, h = d.params()
    tensor = TpsaTensor([x + k, px + h])

    assert tensor.substitute(x + 1).flat == (x + 1 + k, x + 1 + h)
    assert tensor.substitute(None, k + 1).flat == (x + k + 1, px + k + 1)
    # Simultaneous, not recursive: the k in the replacement for x is retained.
    assert tensor.substitute(k, 0.0).flat == (k, k)


def test_substitute_positional_complex() -> None:
    d = _descriptor()
    x, px = d.vars()
    k, h = d.params()
    tensor = TpsaTensor([x + k, px + h])
    result = tensor.substitute([1j, px], 0.0)

    assert result.is_complex
    assert result.flat == (d.constant(1j), ComplexTpsa.from_tpsa(px))
    assert not tensor.is_complex


@pytest.mark.parametrize('var_subs, param_subs', [([0.0], None), (None, [0.0]), ([], None)])
def test_substitute_rejects_wrong_group_size(var_subs: Any, param_subs: Any) -> None:
    tensor = TpsaMap(_descriptor())
    with pytest.raises(ValueError, match='Expected 2 substitutions'):
        tensor.substitute(var_subs, param_subs)


def test_substitute_rejects_mixed_arguments() -> None:
    tensor = TpsaMap(_descriptor())
    with pytest.raises(TypeError, match='positional and named'):
        tensor.substitute(0.0, k=1.0)
    with pytest.raises(TypeError, match='positional and named'):
        tensor.substitute(None, 0.0, x=1.0)
    with pytest.raises(ValueError, match='one-dimensional'):
        tensor.substitute(np.zeros((2, 1)))


def test_substitute_empty_groups_and_none() -> None:
    d = Descriptor(var_names=('x', 'px'), order=3)
    tensor = TpsaMap(d)
    for result in (
        tensor.substitute(None, None),
        tensor.substitute(None, []),
        tensor.substitute(None, 0.0),
    ):
        assert result.flat == tensor.flat
        assert all(a is not b for a, b in zip(result.flat, tensor.flat, strict=True))


def test_substitute_group_argument_names_are_valid_labels() -> None:
    d = Descriptor(var_names=('var_subs', 'px'), param_names=('param_subs',), order=3)
    x, px = d.vars()
    k = d.param('param_subs')
    tensor = TpsaTensor([x + k, px])
    assert tensor.substitute(var_subs=2.0, param_subs=3.0).flat == (d.constant(5.0), px)


def test_substitute_zero_parameters() -> None:
    d = _descriptor()
    x, px = d.vars()
    k, h = d.params()
    m = TpsaMap([3 + x + k * x * px + h * x, px + k * h])

    result = m.substitute(**dict.fromkeys(d.param_names, 0))

    assert isinstance(result, TpsaMap)
    assert result.descriptor is d
    assert not result.is_complex
    assert result.components == (3 + x, px)
    assert m.components == (3 + x + k * x * px + h * x, px + k * h)
    assert all(a is not b for a, b in zip(result.flat, m.flat, strict=True))

    parameter_free = Descriptor(var_names=d.var_names, order=d.order)
    x0, px0 = parameter_free.vars()
    assert result.change_descriptor(parameter_free).components == (3 + x0, px0)


def test_substitute_single_parameter() -> None:
    d = _descriptor()
    x, px = d.vars()
    k, h = d.params()
    m = TpsaMap([x + k * x * px + h * x, px + k * h])

    result = m.substitute(k=0)

    assert result.components == (x + h * x, px)
    assert not result.is_complex


def test_substitute_parameter_shift() -> None:
    d = _descriptor()
    x, px = d.vars()
    k, h = d.params()
    m = TpsaMap([x + k * x * px, px + k * h])

    result = m.substitute(k=k + 0.125)

    assert result.components == (x + k * x * px + 0.125 * x * px, px + k * h + 0.125 * h)


def test_substitute_variables_and_parameters() -> None:
    d = _descriptor()
    x, px = d.vars()
    k, _ = d.params()
    m = TpsaMap([3 + x + k * x * px, px])

    result = m.substitute(x=2 * x, k=0.125)

    assert result.components == (3 + 2 * x + 0.25 * x * px, px)


def test_substitute_scalar_variable_is_constant() -> None:
    d = _descriptor()
    x, px = d.vars()
    k, _ = d.params()
    m = TpsaMap([x + k * x * px, px])

    assert m.substitute(x=2).components == (2 + 2 * k * px, px)
    assert m.substitute(x=0).components == (d.zero(), px)


def test_substitute_is_simultaneous() -> None:
    d = _descriptor()
    x, px = d.vars()
    k, h = d.params()
    m = TpsaMap([x + 2 * px + k, px + 2 * k])

    result = m.substitute(x=px, px=x, k=h, h=0)

    assert result.components == (px + 2 * x + h, x + 2 * h)


def test_substitute_parameter_with_variable() -> None:
    d = _descriptor()
    x, px = d.vars()
    k, _ = d.params()
    m = TpsaMap([x + k * x * px, px])

    assert m.substitute(k=x).components == (x + x**2 * px, px)


def test_substitute_no_arguments_returns_independent_copy() -> None:
    d = _descriptor()
    m = TpsaMap(d)

    result = m.substitute()

    assert isinstance(result, TpsaMap)
    assert result is not m
    assert result.components == m.components
    assert all(a is not b for a, b in zip(result.flat, m.flat, strict=True))


@pytest.mark.parametrize('cls', [TpsaTensor, TpsaVector, TpsaMap, TpsaVectorField])
def test_substitute_preserves_container_type(cls: type[TpsaTensor[Any]]) -> None:
    d = _descriptor()
    x, px = d.vars()
    k, _ = d.params()
    tensor = cls([x + k, px])

    result = tensor.substitute(k=0)

    assert type(result) is cls
    assert result.shape == tensor.shape
    assert result.flat == (x, px)


def test_substitute_high_rank_tensor() -> None:
    d = _descriptor()
    x, px = d.vars()
    k, h = d.params()
    tensor = TpsaTensor([[x + k, px, h * x], [x * px, k, h], [x**2, px**2, k * h]])

    result = tensor.substitute(x=2 * x, k=0)

    assert result.shape == (3, 3)
    assert result.descriptor is d
    assert result.flat == (
        2 * x,
        px,
        2 * h * x,
        2 * x * px,
        d.zero(),
        h,
        4 * x**2,
        px**2,
        d.zero(),
    )


def test_substitute_tensor_view() -> None:
    d = _descriptor()
    x, px = d.vars()
    k, _ = d.params()
    tensor = TpsaTensor([[x + k, px], [x * px, k * x]])

    result = tensor.T.substitute(k=0)

    assert result.shape == (2, 2)
    assert result.flat == (x, x * px, px, d.zero())
    assert tensor.flat == (x + k, px, x * px, k * x)


def test_substitute_promotes_to_complex() -> None:
    d = _descriptor()
    x, px = d.vars()
    k, _ = d.params()
    m = TpsaMap([x + k, px])

    result = m.substitute(k=1j)

    assert isinstance(result, TpsaMap)
    assert result.is_complex
    assert all(isinstance(t, ComplexTpsa) for t in result.flat)
    assert result.components == (ComplexTpsa.from_tpsa(x) + 1j, ComplexTpsa.from_tpsa(px))
    assert not m.is_complex


def test_substitute_complex_polynomial() -> None:
    d = _descriptor()
    x, px = d.vars()
    k, _ = d.params()
    m = TpsaMap([x + k, px]).to_complex()

    result = m.substitute(k=1j * x)

    assert result.is_complex
    assert result.components == (
        (1 + 1j) * ComplexTpsa.from_tpsa(x),
        ComplexTpsa.from_tpsa(px),
    )


def test_substitute_preserves_complex_type() -> None:
    d = _descriptor()
    m = TpsaMap(d).to_complex()

    result = m.substitute(k=0)

    assert result.is_complex
    assert result.components == m.components


def test_substitute_unions_and_reorders_descriptors() -> None:
    d1 = Descriptor(var_names=('x', 'px'), param_names=('k',), order=3)
    d2 = Descriptor(var_names=('px', 'x'), param_names=('h', 'k'), order=4, param_order=2)
    x, px = d1.vars()
    k = d1.param('k')
    h = d2.param('h')
    m = TpsaMap([x + k, px])

    # Include a variable and both parameters from the reordered descriptor.
    result = m.substitute(k=h**2 + d2.var('x') + d2.param('k'))

    common = d1.union(d2)
    x, px = common.vars()
    k, h = common.params()
    assert result.descriptor is common
    assert result.descriptor.var_names == ('x', 'px')
    assert result.descriptor.param_names == ('k', 'h')
    assert result.components == (2 * x + k + h**2, px)


def test_substitute_allocates_higher_order() -> None:
    d = _descriptor()
    x, px = d.vars()
    t = d.zero(order=2)
    t.set((2, 0, 0, 0), 1)
    m = TpsaMap([t, px])

    result = m.substitute(x=t)

    assert result.components == (x**4, px)
    assert result.flat[0].truncation_order == 4


def test_substitute_respects_parameter_order() -> None:
    d = _descriptor()
    x, px = d.vars()
    k, h = d.params()
    m = TpsaMap([x + k**2, px])

    assert m.substitute(k=k * h).components == (x, px)


def test_substitute_parameter_free_descriptor() -> None:
    d = Descriptor(var_names=('x', 'px'), order=3)
    x, px = d.vars()
    m = TpsaMap([x + px, px])

    assert m.substitute(x=2 * x).components == (2 * x + px, px)


def test_substitute_rejects_unknown_name() -> None:
    m = TpsaMap(_descriptor())

    with pytest.raises(KeyError, match='unknown_parameter'):
        m.substitute(unknown_parameter=0)


def test_substitute_rejects_incompatible_variables() -> None:
    m = TpsaMap(_descriptor())
    other = Descriptor(var_names=('y', 'py'), order=4)

    with pytest.raises(ValueError, match='variable'):
        m.substitute(x=other.var('y'))
