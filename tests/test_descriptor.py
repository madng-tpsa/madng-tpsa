"""Tests for GTPSA descriptors."""

import gc
import weakref

import pytest

import madng_tpsa


def test_descriptor_is_reused():
    a = madng_tpsa.Descriptor(6, 2)
    b = madng_tpsa.Descriptor(6, 2)
    assert a is b  # GTPSA dedups equivalent descriptors
    assert a != madng_tpsa.Descriptor(6, 3)
    assert a != madng_tpsa.Descriptor(4, 2)
    assert a in madng_tpsa.Descriptor._instances_by_ptr.values()


@pytest.mark.parametrize('nv, order', [(1, 1), (2, 3), (6, 4)])
def test_descriptor_queries_come_from_c(nv, order):
    d = madng_tpsa.Descriptor(nv, order)
    assert (d.num_vars, d.order) == (nv, order)
    assert (d.num_params, d.monomial_length) == (0, nv)


def test_descriptor_order_must_be_positive():
    with pytest.raises(ValueError, match='Descriptor order must be positive'):
        madng_tpsa.Descriptor(6, 0)


def test_descriptor_with_parameters():
    d = madng_tpsa.Descriptor(6, 2, num_params=2, param_order=1)
    assert (d.num_vars, d.num_params, d.param_order) == (6, 2, 1)
    assert d.monomial_length == 8
    assert 'num_params=2' in repr(d)


def test_factories_dispatch_to_complex_tpsas():
    d = madng_tpsa.Descriptor(2, 2, num_params=2)

    assert isinstance(d.constant(1 + 2j), madng_tpsa.ComplexTpsa)
    assert isinstance(d.var(1, 1j), madng_tpsa.ComplexTpsa)
    assert isinstance(d.param(1, 1j), madng_tpsa.ComplexTpsa)
    variables = d.vars([0.0, 1j])
    parameters = d.params([0.0, 1j])
    assert isinstance(variables[0], madng_tpsa.Tpsa)
    assert isinstance(variables[1], madng_tpsa.ComplexTpsa)
    assert isinstance(parameters[0], madng_tpsa.Tpsa)
    assert isinstance(parameters[1], madng_tpsa.ComplexTpsa)
    assert [t.const_part for t in variables] == [0j, 1j]
    assert [t.const_part for t in parameters] == [0j, 1j]

    assert isinstance(d.constant(1.0), madng_tpsa.Tpsa)
    assert isinstance(d.var(1), madng_tpsa.Tpsa)
    assert isinstance(d.param(1), madng_tpsa.Tpsa)
    assert all(isinstance(t, madng_tpsa.Tpsa) for t in d.vars())
    assert all(isinstance(t, madng_tpsa.Tpsa) for t in d.params())


def test_descriptor_param_order_must_be_positive():
    with pytest.raises(ValueError, match='Descriptor parameter order must be positive'):
        madng_tpsa.Descriptor(6, 2, num_params=2, param_order=0)


def test_is_valid_monomial():
    # Never call get() beyond the order: GTPSA exit(1)s the interpreter, so this
    # predicate is the only safe way to ask.
    d = madng_tpsa.Descriptor(2, 2)
    assert d.is_valid_monomial((1, 1))
    assert not d.is_valid_monomial((2, 1))  # total order 3 > 2
    assert not d.is_valid_monomial((1, 1, 1))  # wrong length


def test_monomial_index_matches_the_stored_coefficient():
    # The index is what a C loop uses with mad_tpsa_geti instead of decoding a
    # monomial per read, so it has to select the same coefficient as get().
    d = madng_tpsa.Descriptor(6, 3)
    t = d.zero()
    monomials = [
        (0, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 2),
        (1, 1, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 3),
    ]
    for value, monomial in enumerate(monomials, start=1):
        t.set(monomial, float(value))
    for value, monomial in enumerate(monomials, start=1):
        index = d.monomial_index(monomial)
        assert madng_tpsa.lib.mad_tpsa_geti(t.ptr, index) == float(value)
        assert t.get(monomial) == float(value)


def test_monomial_index_of_the_constant_term_is_zero():
    d = madng_tpsa.Descriptor(6, 2)
    assert d.monomial_index((0, 0, 0, 0, 0, 0)) == 0


def test_monomial_index_with_parameters():
    d = madng_tpsa.Descriptor(6, 2, num_params=2, param_order=1)
    t = d.zero()
    monomial = (1, 0, 0, 0, 0, 0, 1, 0)  # one variable times one parameter
    t.set(monomial, 0.25)
    index = d.monomial_index(monomial)
    assert madng_tpsa.lib.mad_tpsa_geti(t.ptr, index) == 0.25
    # a parameter beyond param_order is not representable
    with pytest.raises(ValueError, match='Invalid monomial'):
        d.monomial_index((0, 0, 0, 0, 0, 0, 2, 0))


def test_monomial_index_rejects_invalid_monomials():
    d = madng_tpsa.Descriptor(2, 2)
    with pytest.raises(ValueError, match='Invalid monomial'):
        d.monomial_index((2, 1))  # total order 3 > 2
    with pytest.raises(ValueError, match='Invalid monomial'):
        d.monomial_index((1, 1, 1))  # wrong length


def test_monomial_index_of_a_low_order_series_is_still_readable():
    # A series can be shorter than the descriptor. geti has to return 0 there
    # rather than read past the coefficients.
    d = madng_tpsa.Descriptor(6, 4)
    t = d.zero(order=1)
    index = d.monomial_index((0, 0, 0, 0, 0, 4))
    assert madng_tpsa.lib.mad_tpsa_geti(t.ptr, index) == 0.0


def test_descriptor_instances_and_from_ptr():
    d = madng_tpsa.Descriptor(6, 4)
    live = list(madng_tpsa.Descriptor._instances_by_ptr.values())
    assert d in live
    assert all(isinstance(x, madng_tpsa.Descriptor) for x in live)
    # the same C pointer always wraps to the same Python object
    assert madng_tpsa.Descriptor.from_ptr(d.ptr) is d


def test_descriptor_accepts_variable_and_parameter_labels():
    d = madng_tpsa.Descriptor(variables=['x', 'px'], order=3, params=['delta'])

    assert d.num_vars == 2
    assert d.num_params == 1
    assert d.var_labels == ('x', 'px')
    assert d.param_labels == ('delta',)
    assert d.var('px').grad() == [0.0, 1.0]
    assert d.param('delta').param_grad() == [1.0]


def test_descriptor_generates_default_labels():
    d = madng_tpsa.Descriptor(2, 3, num_params=1)

    assert d.var_labels == ('v_1', 'v_2')
    assert d.param_labels == ('p_1',)


def test_descriptor_accepts_per_variable_max_orders():
    d = madng_tpsa.Descriptor(variables=['x', 'y'], order=3, max_orders=[1, 3])

    assert d.max_orders == (1, 3)
    assert d.is_valid_monomial((1, 0))
    assert not d.is_valid_monomial((2, 0))
    assert d.is_valid_monomial((0, 3))


def test_descriptor_accepts_parameter_max_orders():
    d = madng_tpsa.Descriptor(variables=['x'], order=3, params=['k'], max_orders=[3, 1])

    assert d.max_orders == (3, 1)
    assert d.is_valid_monomial((1, 1))
    assert not d.is_valid_monomial((0, 2))


def test_descriptor_warns_when_max_orders_raise_order():
    with pytest.warns(UserWarning, match='Order 2 was given, but it was overridden to 4'):
        d = madng_tpsa.Descriptor(variables=['x', 'y'], order=2, max_orders=[2, 4])

    assert d.order == 4


def test_descriptor_rejects_mismatched_label_counts():
    with pytest.raises(ValueError, match='Variable labels must contain 3 entries'):
        madng_tpsa.Descriptor(3, 2, variables=['x', 'y'])

    with pytest.raises(ValueError, match='Param labels must contain 2 entries'):
        madng_tpsa.Descriptor(2, 2, num_params=2, params=['k'])

    with pytest.raises(ValueError, match='Length of max_orders must have the length'):
        madng_tpsa.Descriptor(2, 2, num_params=1, max_orders=[1, 1])

    with pytest.raises(ValueError, match='All max_orders must be positive'):
        madng_tpsa.Descriptor(2, 2, max_orders=[1, 0])


def test_descriptor_rejects_duplicate_labels():
    with pytest.raises(ValueError, match='Variable labels must be unique'):
        madng_tpsa.Descriptor(variables=['x', 'x'], order=2)

    with pytest.raises(ValueError, match='Param labels must be unique'):
        madng_tpsa.Descriptor(1, 2, params=['k', 'k'])


def test_descriptor_unknown_labels_raise_key_error():
    d = madng_tpsa.Descriptor(variables=['x'], order=2, params=['k'])

    with pytest.raises(KeyError):
        d.var('y')
    with pytest.raises(KeyError):
        d.param('q')


def test_descriptor_reuses_existing_labels_and_warns_on_mismatch():
    d = madng_tpsa.Descriptor(variables=['x', 'y'], order=2)

    with pytest.warns(UserWarning, match='different variable labels'):
        same = madng_tpsa.Descriptor(variables=['a', 'b'], order=2)

    assert same is d
    assert same.var_labels == ('x', 'y')


def test_tpsa_keeps_descriptor_alive():
    d = madng_tpsa.Descriptor(variables=['x'], order=2)
    descriptor_ref = weakref.ref(d)
    t = d.var('x')

    del d
    gc.collect()

    assert descriptor_ref() is t.descriptor
    assert t.descriptor.var_labels == ('x',)
