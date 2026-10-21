"""Tests for GTPSA descriptors."""

import gc
import json
import weakref

import numpy as np
import pytest

from madng_tpsa import ComplexTpsa, Descriptor, Tpsa, lib
from madng_tpsa.descriptor import _NativeDescriptor


def test_descriptor_is_reused():
    a = Descriptor(6, 2)
    b = Descriptor(6, 2)
    assert a is b  # GTPSA dedups equivalent descriptors
    assert a != Descriptor(6, 3)
    assert a != Descriptor(4, 2)
    assert a._owner in Descriptor._owners_by_ptr.values()


def test_descriptor_comparison_with_other_types_returns_not_implemented():
    descriptor = Descriptor(2, 3)
    assert descriptor.__eq__(object()) is NotImplemented
    assert descriptor != object()
    assert (descriptor == object()) is False


@pytest.mark.parametrize('nv, order', [(1, 1), (2, 3), (6, 4)])
def test_descriptor_queries_come_from_c(nv, order):
    d = Descriptor(nv, order)
    assert (d.num_vars, d.order) == (nv, order)
    assert (d.num_params, d.monomial_length) == (0, nv)


def test_descriptor_order_must_be_positive():
    with pytest.raises(ValueError, match='Descriptor order must be positive'):
        Descriptor(6, 0)


def test_descriptor_with_parameters():
    d = Descriptor(6, 2, num_params=2, param_order=1)
    assert (d.num_vars, d.num_params, d.param_order) == (6, 2, 1)
    assert d.monomial_length == 8
    assert 'num_params=2' in repr(d)


def test_factories_dispatch_to_complex_tpsas():
    d = Descriptor(2, 2, num_params=2)

    assert isinstance(d.constant(1 + 2j), ComplexTpsa)
    assert isinstance(d.var(1, 1j), ComplexTpsa)
    assert isinstance(d.param(1, 1j), ComplexTpsa)
    variables = d.vars([0.0, 1j])
    parameters = d.params([0.0, 1j])
    assert isinstance(variables[0], Tpsa)
    assert isinstance(variables[1], ComplexTpsa)
    assert isinstance(parameters[0], Tpsa)
    assert isinstance(parameters[1], ComplexTpsa)
    assert [t.const_part for t in variables] == [0j, 1j]
    assert [t.const_part for t in parameters] == [0j, 1j]

    assert isinstance(d.constant(1.0), Tpsa)
    assert isinstance(d.var(1), Tpsa)
    assert isinstance(d.param(1), Tpsa)
    assert all(isinstance(t, Tpsa) for t in d.vars())
    assert all(isinstance(t, Tpsa) for t in d.params())


def test_descriptor_param_order_must_be_positive():
    with pytest.raises(ValueError, match='Descriptor parameter order must be positive'):
        Descriptor(6, 2, num_params=2, param_order=0)


def test_is_valid_monomial():
    # Never call get() beyond the order: GTPSA exit(1)s the interpreter, so this
    # predicate is the only safe way to ask.
    d = Descriptor(2, 2)
    assert d.is_valid_monomial((1, 1))
    assert not d.is_valid_monomial((2, 1))  # total order 3 > 2
    assert not d.is_valid_monomial((1, 1, 1))  # wrong length


def test_is_valid_monomial_checks_integer_type_and_byte_range():
    descriptor = Descriptor(2, 3)
    assert descriptor.is_valid_monomial(np.array([1, 0], dtype=np.int64))
    assert descriptor.is_valid_monomial(range(2))
    for invalid in (True, np.bool_(1), -1, 256, 1.5, '1'):
        assert not descriptor.is_valid_monomial((invalid, 0))


def test_monomial_index_matches_the_stored_coefficient():
    # The index is what a C loop uses with mad_tpsa_geti instead of decoding a
    # monomial per read, so it has to select the same coefficient as get().
    d = Descriptor(6, 3)
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
        assert lib.mad_tpsa_geti(t.ptr, index) == float(value)
        assert t.get(monomial) == float(value)


def test_monomial_index_of_the_constant_term_is_zero():
    d = Descriptor(6, 2)
    assert d.monomial_index((0, 0, 0, 0, 0, 0)) == 0


def test_monomial_index_with_parameters():
    d = Descriptor(6, 2, num_params=2, param_order=1)
    t = d.zero()
    monomial = (1, 0, 0, 0, 0, 0, 1, 0)  # one variable times one parameter
    t.set(monomial, 0.25)
    index = d.monomial_index(monomial)
    assert lib.mad_tpsa_geti(t.ptr, index) == 0.25
    # a parameter beyond param_order is not representable
    with pytest.raises(ValueError, match='Invalid monomial'):
        d.monomial_index((0, 0, 0, 0, 0, 0, 2, 0))


def test_monomial_index_rejects_invalid_monomials():
    d = Descriptor(2, 2)
    with pytest.raises(ValueError, match='Invalid monomial'):
        d.monomial_index((2, 1))  # total order 3 > 2
    with pytest.raises(ValueError, match='Invalid monomial'):
        d.monomial_index((1, 1, 1))  # wrong length


def test_monomial_index_of_a_low_order_series_is_still_readable():
    # A series can be shorter than the descriptor. geti has to return 0 there
    # rather than read past the coefficients.
    d = Descriptor(6, 4)
    t = d.zero(order=1)
    index = d.monomial_index((0, 0, 0, 0, 0, 4))
    assert lib.mad_tpsa_geti(t.ptr, index) == 0.0


def test_descriptor_instances_and_from_ptr():
    d = Descriptor(6, 4)
    live = list(Descriptor._owners_by_ptr.values())
    assert d._owner in live
    assert all(isinstance(x, _NativeDescriptor) for x in live)
    # the same C pointer always wraps to the same Python object
    assert Descriptor.from_ptr(d.ptr) is d


def test_descriptor_accepts_variable_and_parameter_labels():
    d = Descriptor(variables=['x', 'px'], order=3, parameters=['delta'])

    assert d.num_vars == 2
    assert d.num_params == 1
    assert d.variables == ('x', 'px')
    assert d.parameters == ('delta',)
    assert d.var('px').grad() == [0.0, 1.0]
    assert d.param('delta').param_grad() == [1.0]


def test_descriptor_generates_default_labels():
    d = Descriptor(2, 3, num_params=1)

    assert d.variables == ('v_1', 'v_2')
    assert d.parameters == ('p_1',)


def test_descriptor_accepts_per_variable_max_orders():
    d = Descriptor(variables=['x', 'y'], order=3, max_orders=[1, 3])

    assert d.max_orders == (1, 3)
    assert d.is_valid_monomial((1, 0))
    assert not d.is_valid_monomial((2, 0))
    assert d.is_valid_monomial((0, 3))


def test_descriptor_accepts_parameter_max_orders():
    d = Descriptor(variables=['x'], order=3, parameters=['k'], max_orders=[3, 1])

    assert d.max_orders == (3, 1)
    assert d.is_valid_monomial((1, 1))
    assert not d.is_valid_monomial((0, 2))


def test_descriptor_warns_when_max_orders_raise_order():
    with pytest.warns(UserWarning, match='Order 2 was given, but it was overridden to 4'):
        d = Descriptor(variables=['x', 'y'], order=2, max_orders=[2, 4])

    assert d.order == 4


def test_descriptor_rejects_mismatched_label_counts():
    with pytest.raises(ValueError, match='Variable labels must contain 3 entries'):
        Descriptor(3, 2, variables=['x', 'y'])

    with pytest.raises(ValueError, match='Param labels must contain 2 entries'):
        Descriptor(2, 2, num_params=2, parameters=['k'])

    with pytest.raises(ValueError, match='Length of max_orders must have the length'):
        Descriptor(2, 2, num_params=1, max_orders=[1, 1])

    with pytest.raises(ValueError, match='All max_orders must be positive'):
        Descriptor(2, 2, max_orders=[1, 0])


def test_descriptor_rejects_duplicate_labels():
    with pytest.raises(ValueError, match='Variable labels must be unique'):
        Descriptor(variables=['x', 'x'], order=2)

    with pytest.raises(ValueError, match='Param labels must be unique'):
        Descriptor(1, 2, parameters=['k', 'k'])


def test_descriptor_unknown_labels_raise_key_error():
    d = Descriptor(variables=['x'], order=2, parameters=['k'])

    with pytest.raises(KeyError):
        d.var('y')
    with pytest.raises(KeyError):
        d.param('q')


def test_tpsa_keeps_descriptor_alive():
    d = Descriptor(variables=['x'], order=2)
    descriptor_ref = weakref.ref(d)
    t = d.var('x')

    del d
    gc.collect()

    assert descriptor_ref() is t.descriptor
    assert t.descriptor.variables == ('x',)


def test_is_valid_monomial_rejects_wrong_length():
    descriptor = Descriptor(
        variables=['x', 'px'],
        order=3,
        parameters=['k'],
    )

    assert not descriptor.is_valid_monomial((1, 0))
    assert not descriptor.is_valid_monomial((1, 0, 0, 0))
    assert descriptor.is_valid_monomial((1, 0, 0))


def test_descriptor_to_dict():
    desc1 = Descriptor(order=5, variables=('x', 'y'))
    dct1 = desc1.to_dict()
    assert set(dct1.keys()) == {
        '__class__',
        'order',
        'param_order',
        'variables',
        'parameters',
        'max_orders',
    }
    assert dct1['__class__'] == 'Descriptor'
    assert dct1['order'] == 5
    assert dct1['param_order'] == 0
    assert dct1['variables'] == ('x', 'y')
    assert dct1['parameters'] == ()
    assert dct1['max_orders'] is None
    desc1_copy1 = Descriptor.from_dict(dct1)
    assert desc1_copy1 is desc1
    # Test json serialisation
    desc1_copy2 = Descriptor.from_dict(json.loads(json.dumps(dct1)))
    assert desc1_copy2 is desc1

    desc2 = Descriptor(order=5, variables=('x', 'y'), max_orders=(3, 4))
    dct2 = desc2.to_dict()
    assert set(dct2.keys()) == {
        '__class__',
        'order',
        'param_order',
        'variables',
        'parameters',
        'max_orders',
    }
    assert dct2['__class__'] == 'Descriptor'
    assert dct2['order'] == 5
    assert dct2['param_order'] == 0
    assert dct2['variables'] == ('x', 'y')
    assert dct2['parameters'] == ()
    assert dct2['max_orders'] == (3, 4)
    desc2_copy1 = Descriptor.from_dict(dct2)
    assert desc2_copy1 is desc2
    # Test json serialisation
    desc2_copy2 = Descriptor.from_dict(json.loads(json.dumps(dct2)))
    assert desc2_copy2 is desc2

    desc3 = Descriptor(order=5, variables=('x', 'y'), param_order=2, parameters=('k1', 'k2'))
    dct3 = desc3.to_dict()
    assert set(dct3.keys()) == {
        '__class__',
        'order',
        'param_order',
        'variables',
        'parameters',
        'max_orders',
    }
    assert dct3['__class__'] == 'Descriptor'
    assert dct3['order'] == 5
    assert dct3['param_order'] == 2
    assert dct3['variables'] == ('x', 'y')
    assert dct3['parameters'] == ('k1', 'k2')
    assert dct3['max_orders'] is None
    desc3_copy1 = Descriptor.from_dict(dct3)
    assert desc3_copy1 is desc3
    # Test json serialisation
    desc3_copy2 = Descriptor.from_dict(json.loads(json.dumps(dct3)))
    assert desc3_copy2 is desc3

    desc4 = Descriptor(
        order=5,
        variables=('x', 'y'),
        param_order=2,
        parameters=('k1', 'k2'),
        max_orders=(5, 4, 1, 1),
    )
    dct4 = desc4.to_dict()
    assert set(dct4.keys()) == {
        '__class__',
        'order',
        'param_order',
        'variables',
        'parameters',
        'max_orders',
    }
    assert dct4['__class__'] == 'Descriptor'
    assert dct4['order'] == 5
    assert dct4['param_order'] == 2
    assert dct4['variables'] == ('x', 'y')
    assert dct4['parameters'] == ('k1', 'k2')
    assert dct4['max_orders'] == (5, 4, 1, 1)
    desc4_copy1 = Descriptor.from_dict(dct4)
    assert desc4_copy1 is desc4
    # Test json serialisation
    desc4_copy2 = Descriptor.from_dict(json.loads(json.dumps(dct4)))
    assert desc4_copy2 is desc4
