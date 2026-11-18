"""End-to-end coverage for declarative MAD-NG map/vector wrappers."""

import numpy as np
import pytest

from madng_tpsa import ComplexTpsa, Descriptor, Tpsa, TpsaMap, TpsaTensor, TpsaVectorField
from madng_tpsa._cffi import ffi
from madng_tpsa.madng_definitions import (
    EXPPB,
    FGRAD,
    FLD2VEC,
    FUNCTIONS,
    LOGPB,
    MNRM,
    MORD,
    PMINV,
    POISBRA,
    TRANSLATE,
)
from madng_tpsa.madng_function import generate_cffi_header, unique_signatures


def test_map_wrapper_registry_and_signatures():
    names = {function.name for function in FUNCTIONS}
    assert {
        'mad_tpsa_pminv',
        'mad_tpsa_poisbra',
        'mad_tpsa_exppb',
        'mad_tpsa_logpb',
        'mad_tpsa_translate',
        'mad_tpsa_mnrm',
        'mad_tpsa_mord',
        'mad_tpsa_fld2vec',
        'mad_tpsa_fgrad',
    } <= names
    assert 'mad_ctpsa_translate' in generate_cffi_header(FUNCTIONS)
    tags = [sig.tag for sig in unique_signatures(FUNCTIONS)]
    assert len(tags) == len(set(tags))
    assert PMINV.return_like == 'ma'
    assert EXPPB.return_like == 'mb'


@pytest.mark.parametrize('complex_needed', [False, True])
def test_partial_inverse_with_parameters(complex_needed):
    d = Descriptor(variables=('q', 'p'), parameters=('k',), order=3, param_order=2)
    q, p = d.vars()
    (k,) = d.params()
    base = TpsaMap([q + k, 2 * p])
    source = base.to_complex() if complex_needed else base
    result = source.partial_inverse([True, False])
    direct = PMINV(source, ffi.new('int[]', [1, 0, 0]))
    assert isinstance(direct, TpsaMap)
    np.testing.assert_allclose(direct.evaluate((3, 2), (0.5,)), (2.5, 4))
    assert isinstance(result, TpsaMap)
    assert result.is_complex == complex_needed
    np.testing.assert_allclose(result.evaluate((3, 2), (0.5,)), (2.5, 4))
    np.testing.assert_allclose(source.evaluate((3, 2), (0.5,)), (3.5, 4))


def test_exponential_and_logarithm_wrappers_keep_container_types():
    d = Descriptor(variables=('q', 'p'), order=4)
    q, p = d.vars()
    mapping = TpsaMap([q + p * p, p])
    field = TpsaVectorField(d)
    result = EXPPB(field, mapping)
    assert isinstance(result, TpsaMap)
    np.testing.assert_allclose(result.evaluate((0.2, 0.5)), mapping.evaluate((0.2, 0.5)))
    # The absent initial guess is represented by a native null pointer.
    log_components = LOGPB(TpsaMap(d), ffi.NULL)
    assert len(log_components) == d.num_vars
    assert all(not t.monomial_coeffs(tol=0) for t in log_components)


def test_translate_norm_order_and_scalar_field_wrappers():
    d = Descriptor(variables=('q', 'p'), order=4)
    q, p = d.vars()
    vector = TpsaMap([q, p])
    translated = TRANSLATE(vector, [0.5, -2.0])
    assert isinstance(translated, TpsaMap)
    np.testing.assert_allclose(translated.evaluate((1.0, 3.0)), (1.5, 1.0))
    assert MNRM(vector) == pytest.approx(vector.norm())
    assert MORD(vector, True) == 1  # noqa: FBT003
    assert MORD(vector, False) == max(series.truncation_order for series in vector.flat)  # noqa: FBT003
    translated_complex = TRANSLATE(vector, [1j, 0], complex_needed=True)
    assert translated_complex.is_complex
    np.testing.assert_allclose(translated_complex.evaluate((1.0, 3.0)), (1.0 + 1j, 3.0))

    field = TpsaVectorField([p, -q])
    assert (FGRAD(field, q * q) - 2 * q * p).monomial_coeffs(tol=0) == {}
    generator = FLD2VEC(TpsaVectorField.from_generator(q * q / 2))
    np.testing.assert_allclose(TpsaTensor([generator]).evaluate((1.2, 0.7)), [1.2**2 / 2])


@pytest.mark.parametrize('complex_needed', [False, True])
def test_generated_poisson_bracket(complex_needed):
    d = Descriptor(variables=('q', 'p', 'x', 'px'), order=3)
    q, p, x, px = d.vars()

    if complex_needed:
        a = ComplexTpsa.from_tpsa(q + x) * 1j
        b = ComplexTpsa.from_tpsa(p + px) * 1j
    else:
        a = q + x
        b = p + px

    result = POISBRA(a, b, nv=0)
    assert result.const_part == pytest.approx(-2 if complex_needed else 2)

    first_pair = POISBRA(a, b, nv=2)
    assert first_pair.const_part == pytest.approx(-1 if complex_needed else 1)

    if isinstance(a, ComplexTpsa):
        assert isinstance(b, ComplexTpsa)
        assert a.poisson_bracket(b, num_pairs=1) == first_pair
        assert a.poisson_bracket(b) == result
    else:
        assert isinstance(b, Tpsa)
        assert a.poisson_bracket(b, num_pairs=1) == first_pair
        assert a.poisson_bracket(b) == result


def test_generated_poisson_bracket_native_error_recovery():
    # A native descriptor mismatch must be caught by the generated protected call.
    d1 = Descriptor(variables=('q', 'p'), order=3)
    d2 = Descriptor(variables=('x', 'px'), order=4)
    q, p = d1.var(1), d2.var(2)
    output = d1.zero()

    from madng_tpsa._cffi import lib

    native = getattr(lib, POISBRA.real.name)
    wrapped = getattr(lib, POISBRA.real.wrapper_name)
    assert wrapped(native, q.ptr, p.ptr, output.ptr, 0) != 0
    # The installed error handler must be restored for subsequent valid calls.
    good_p = d1.var(2)
    assert wrapped(native, q.ptr, good_p.ptr, output.ptr, 0) == 0
    assert output.const_part == pytest.approx(1)
