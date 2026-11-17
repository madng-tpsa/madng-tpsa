"""Integration coverage for generated MAD-NG map calls."""

import pytest

from madng_tpsa import ComplexTpsa, Descriptor, Tpsa, TpsaMap, TpsaVectorField


@pytest.mark.parametrize('complex_needed', [False, True])
def test_parameter_pointer_cache(complex_needed):
    d = Descriptor(variables=('x', 'px'), parameters=('k',), order=3, param_order=2)
    x, px = d.vars()
    if complex_needed:
        x, px = 1j * x, 1j * px
    m = TpsaMap([x, px])
    first = m.pointer_array_with_params
    assert first is m.pointer_array_with_params
    assert len(first) == 3
    assert first[0] == m[0].ptr
    assert first[2] == m._params[0].ptr
    replacement = 2 * m[0]
    m[0] = replacement
    second = m.pointer_array_with_params
    assert second is m.pointer_array_with_params
    assert second is not first
    assert second[0] == replacement.ptr
    assert second[2] == m._params[0].ptr
    if isinstance(replacement, ComplexTpsa):
        replacement.const_part = 1j
    else:
        assert isinstance(replacement, Tpsa)
        replacement.const_part = 1.0
    assert m.pointer_array_with_params is second


def test_parameter_free_pointer_cache():
    d = Descriptor(variables=('x', 'px'), order=3)
    m = TpsaMap(d)
    assert m.pointer_array_with_params is m.pointer_array


def test_actual_descriptor_union_and_complex_promotion():
    da = Descriptor(variables=('x', 'px'), parameters=('k',), order=3, param_order=2)
    db = Descriptor(variables=('x', 'px'), parameters=('q',), order=3, param_order=2)
    x, _ = da.vars()
    (k,) = da.params()
    (q,) = db.params()
    left = TpsaVectorField([k * x, da.zero()])
    right = TpsaVectorField([1j * q, db.complex_zero()])
    out = left.lie_bracket(right)
    assert out.is_complex
    assert out.descriptor.parameters == ('k', 'q')
    assert out[0].monomial_coeffs(tol=0)[(0, 0, 1, 1)] == pytest.approx(-1j)
    assert not left.is_complex
