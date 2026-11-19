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
    d = madng_tpsa.Descriptor(var_names=['x'], order=3, param_names=['k'], param_order=3)
    x = d.var('x')
    k = d.param('k')
    f = x * x + k

    for specification in (1, 'x', x):
        assert f.integrate(specification) == f.integrate(x)


@pytest.mark.xfail(reason='MAD-NG integration does not support parameters')
def test_integrate_resolves_parameter_specifications():
    d = madng_tpsa.Descriptor(var_names=['x'], order=3, param_names=['k'], param_order=3)
    x = d.var('x')
    k = d.param('k')
    f = x * x + k

    for specification in (2, 'k', k):
        assert f.integrate(specification) == f.integrate(k)


def test_integrate_rejects_non_identity_tpsa_variable():
    d = madng_tpsa.Descriptor(1, 2)
    x = d.var(1)

    with pytest.raises(ValueError, match='unit first-order'):
        x.integrate(2.0 * x)


def test_derivative_accepts_variadic_directions_and_explicit_orders():
    d = madng_tpsa.Descriptor(2, 3)
    x, y = d.vars()
    f = x * x * y

    assert f.derivative(1) == f.derivative(x)
    assert f.derivative(1).get((1, 1)) == pytest.approx(2.0)

    mixed = f.derivative((1, 2), y)
    assert mixed.const_part == pytest.approx(2.0)
    assert f.derivative(orders=(2, 1)) == mixed


def test_derivative_resolves_variable_and_parameter_specifications():
    d = madng_tpsa.Descriptor(var_names=['x'], order=3, param_names=['k'])
    x = d.var('x')
    k = d.param('k')
    f = x * x * k

    for specification in (1, 'x', x):
        assert f.derivative(specification) == f.derivative(x)
    for specification in (2, 'k', k):
        assert f.derivative(specification) == f.derivative(k)
    assert f.derivative(1, 2).get((1, 0)) == pytest.approx(2.0)


@pytest.mark.parametrize('receiver_factory', ('zero', 'complex_zero'))
def test_shared_direction_parser(receiver_factory):
    from madng_tpsa._derivatives import resolve_derivative_orders

    d = madng_tpsa.Descriptor(var_names=['x'], order=3, param_names=['k'], param_order=3)
    receiver = getattr(d, receiver_factory)()
    x = d.var('x')
    k = d.param('k')

    assert resolve_derivative_orders(d, 1) == (1, 0)
    assert resolve_derivative_orders(d, 'x') == (1, 0)
    assert resolve_derivative_orders(d, x) == (1, 0)
    assert resolve_derivative_orders(d, 2) == (0, 1)
    assert resolve_derivative_orders(d, 'k') == (0, 1)
    assert resolve_derivative_orders(d, k) == (0, 1)
    assert resolve_derivative_orders(d, ('x', 2), 'k') == (2, 1)
    assert resolve_derivative_orders(d, orders=(2, 1)) == (2, 1)
    assert receiver.derivative(('x', 2), 'k') == receiver.derivative(orders=(2, 1))

    with pytest.raises(ValueError, match='positive degree'):
        resolve_derivative_orders(d, orders=(0, 0))
    with pytest.raises(ValueError, match='length'):
        resolve_derivative_orders(d, orders=(1,))
    with pytest.raises(ValueError, match='index out of range'):
        resolve_derivative_orders(d, 3)
    with pytest.raises(KeyError, match='unknown'):
        resolve_derivative_orders(d, 'unknown')
    with pytest.raises(ValueError, match='unit first-order'):
        resolve_derivative_orders(d, 2.0 * x)
    with pytest.raises(ValueError, match='unit first-order'):
        resolve_derivative_orders(d, x + k)
    with pytest.raises(ValueError, match='unit first-order'):
        resolve_derivative_orders(d, d.constant(1.0))
    with pytest.raises(ValueError, match='incompatible descriptor'):
        resolve_derivative_orders(d, madng_tpsa.Descriptor(1, 3).var(1))


def test_derivative_rejects_invalid_directions():
    d = madng_tpsa.Descriptor(2, 2)
    x, y = d.vars()

    with pytest.raises(ValueError, match='positive degree'):
        x.derivative(orders=(0, 0))
    with pytest.raises(ValueError, match='length'):
        x.derivative(orders=(1,))
    assert x.derivative(orders=(3, 0)).is_zero()
    with pytest.raises(ValueError, match='unit first-order'):
        x.derivative(x + y)
    with pytest.raises(ValueError, match='unit first-order'):
        x.derivative(2.0 * x)
    with pytest.raises(ValueError, match='unit first-order'):
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
