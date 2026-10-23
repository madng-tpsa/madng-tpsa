"""Tests for TPSA differential algebra operations."""

import pytest

import madng_tpsa


def test_integrate_accepts_index_or_identity_variable():
    d = madng_tpsa.Descriptor(2, 3)
    x, y = d.vars()
    f = x * x + 3.0 * y

    by_index = f.integrate(1)
    by_variable = f.integrate(x)

    assert by_index == by_variable
    assert by_index.get((3, 0)) == pytest.approx(1.0 / 3.0)
    assert by_index.get((1, 1)) == pytest.approx(3.0)


def test_integrate_resolves_variable_specifications():
    d = madng_tpsa.Descriptor(variables=['x'], order=3, parameters=['k'], param_order=3)
    x = d.var('x')
    k = d.param('k')
    f = x * x + k

    for specification in (1, 'x', x):
        assert f.integrate(specification) == f.integrate(x)


@pytest.mark.xfail(reason='MAD-NG integration does not support parameters')
def test_integrate_resolves_parameter_specifications():
    d = madng_tpsa.Descriptor(variables=['x'], order=3, parameters=['k'], param_order=3)
    x = d.var('x')
    k = d.param('k')
    f = x * x + k

    for specification in (2, 'k', k):
        assert f.integrate(specification) == f.integrate(k)


def test_integrate_rejects_non_identity_tpsa_variable():
    d = madng_tpsa.Descriptor(1, 2)
    x = d.var(1)

    with pytest.raises(ValueError, match='coefficient 1'):
        x.integrate(2.0 * x)


def test_derivative_accepts_index_identity_variable_tuple_or_single_monomial_tpsa():
    d = madng_tpsa.Descriptor(2, 3)
    x, y = d.vars()
    f = x * x * y

    assert f.derivative(1) == f.derivative(x)
    assert f.derivative(1).get((1, 1)) == pytest.approx(2.0)

    mixed = f.derivative((2, 1))
    assert mixed.const_part == pytest.approx(2.0)

    monomial = d.zero()
    monomial.set((2, 1), 1.0)
    assert f.derivative(monomial) == mixed


def test_derivative_resolves_variable_and_parameter_specifications():
    d = madng_tpsa.Descriptor(variables=['x'], order=3, parameters=['k'])
    x = d.var('x')
    k = d.param('k')
    f = x * x * k

    for specification in (1, 'x', x):
        assert f.derivative(specification) == f.derivative(x)
    for specification in (2, 'k', k):
        assert f.derivative(specification) == f.derivative(k)
    assert f.derivative((1, 1)).get((1, 0)) == pytest.approx(2.0)


@pytest.mark.parametrize('receiver_factory', ('zero', 'complex_zero'))
def test_resolve_single_monomial(receiver_factory):
    d = madng_tpsa.Descriptor(variables=['x'], order=3, parameters=['k'], param_order=3)
    receiver = getattr(d, receiver_factory)()
    x = d.var('x')
    k = d.param('k')
    monomial = d.zero()
    monomial[(2, 1)] = 1.0

    assert receiver._resolve_single_monomial((1, 0)) == (1, 0)
    assert receiver._resolve_single_monomial((1, 1)) == (1, 1)
    assert receiver._resolve_single_monomial((2, 1)) == (2, 1)
    assert receiver._resolve_single_monomial(1) == (1, 0)
    assert receiver._resolve_single_monomial('x') == (1, 0)
    assert receiver._resolve_single_monomial(x) == (1, 0)
    assert receiver._resolve_single_monomial(2) == (0, 1)
    assert receiver._resolve_single_monomial('k') == (0, 1)
    assert receiver._resolve_single_monomial(k) == (0, 1)
    assert receiver._resolve_single_monomial(monomial) == (2, 1)

    with pytest.raises(ValueError, match='positive degree'):
        receiver._resolve_single_monomial((0, 0))
    with pytest.raises(ValueError, match='Monomial must have length'):
        receiver._resolve_single_monomial((1,))
    with pytest.raises(ValueError, match='Monomial is not valid'):
        receiver._resolve_single_monomial((4, 0))
    with pytest.raises(ValueError, match='index out of range'):
        receiver._resolve_single_monomial(3)
    with pytest.raises(KeyError, match='unknown'):
        receiver._resolve_single_monomial('unknown')
    with pytest.raises(ValueError, match='coefficient 1'):
        receiver._resolve_single_monomial(2.0 * x)
    with pytest.raises(ValueError, match='exactly one'):
        receiver._resolve_single_monomial(x + k)
    with pytest.raises(ValueError, match='exactly one'):
        receiver._resolve_single_monomial(d.constant(1.0))
    with pytest.raises(ValueError, match='exactly one'):
        receiver._resolve_single_monomial(d.constant(1.0) + x)
    with pytest.raises(
        ValueError, match='TPSAs have incompatible descriptors or coordinate labels'
    ):
        receiver._resolve_single_monomial(madng_tpsa.Descriptor(1, 3).var(1))


def test_derivative_rejects_invalid_monomials():
    d = madng_tpsa.Descriptor(2, 2)
    x, y = d.vars()

    with pytest.raises(ValueError, match='positive degree'):
        x.derivative((0, 0))
    with pytest.raises(ValueError, match='Monomial must have length'):
        x.derivative((1,))
    with pytest.raises(ValueError, match='Monomial is not valid'):
        x.derivative((3, 0))
    with pytest.raises(ValueError, match='exactly one'):
        x.derivative(x + y)
    with pytest.raises(ValueError, match='coefficient 1'):
        x.derivative(2.0 * x)
    with pytest.raises(ValueError, match='identity variable'):
        x.integrate(x * x)


def test_poisson_bracket_uses_canonical_pairs():
    d = madng_tpsa.Descriptor(2, 2)
    q, p = d.vars()

    assert q.poisson_bracket(p).const_part == pytest.approx(1.0)
    assert p.poisson_bracket(q).const_part == pytest.approx(-1.0)
    assert q.poisson_bracket(p, num_pairs='all').const_part == pytest.approx(1.0)
    assert q.poisson_bracket(p, num_pairs=1).const_part == pytest.approx(1.0)


def test_poisson_bracket_validates_num_pairs():
    d = madng_tpsa.Descriptor(2, 2)
    q, p = d.vars()

    with pytest.raises(ValueError, match='num_pairs'):
        q.poisson_bracket(p, num_pairs=0)
    with pytest.raises(ValueError, match='num_pairs'):
        q.poisson_bracket(p, num_pairs=2)
    with pytest.raises(ValueError, match='num_pairs'):
        q.poisson_bracket(p, num_pairs='bad')  # ty: ignore[invalid-argument-type]
