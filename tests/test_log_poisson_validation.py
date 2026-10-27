"""Native logarithm round trips, including non-Hamiltonian vector fields."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from madng_tpsa import ComplexTpsa, Descriptor, TpsaMap, TpsaVectorField


def _assert_maps_close(actual: TpsaMap, expected: TpsaMap, tol: float = 1e-9) -> None:
    assert actual.shape == expected.shape
    for actual_component, expected_component in zip(actual.flat, expected.flat, strict=True):
        left = actual_component.monomial_coeffs(tol=0)
        right = expected_component.monomial_coeffs(tol=0)
        for monomial in left.keys() | right.keys():
            assert np.isclose(left.get(monomial, 0), right.get(monomial, 0), atol=tol, rtol=tol), (
                monomial,
                left.get(monomial, 0),
                right.get(monomial, 0),
            )


def _shear() -> TpsaMap:
    d = Descriptor(variables=('log_q', 'log_p'), order=5)
    q, p = d.vars()
    return TpsaMap([q, p - 0.015 * q**2])


def test_logarithm_of_identity_is_zero_field() -> None:
    d = Descriptor(variables=('log_ident_q', 'log_ident_p'), order=5)
    field = TpsaMap(d).log_fgrad()
    assert all(tpsa.is_zero() for tpsa in field.flat)


def test_logarithm_of_polynomial_kick_reconstructs_both_components() -> None:
    mapping = _shear()
    field = mapping.log_fgrad()
    recovered = TpsaMap(mapping.descriptor).exp_fgrad(field)
    _assert_maps_close(recovered, mapping)
    # A scalar generator is an *additional* condition, not a requirement of log_fgrad.
    generator = mapping.log_poisson()
    assert generator.descriptor is mapping.descriptor


def test_logarithm_of_non_hamiltonian_map_is_allowed() -> None:
    d = Descriptor(variables=('log_nonsym_q', 'log_nonsym_p'), order=5)
    q, _ = d.vars()
    # div(F) != 0, so this field cannot be Hamiltonian for canonical (q,p).
    field = TpsaVectorField([0.012 * q**2, d.zero()])
    mapping = TpsaMap(d).exp_fgrad(field)
    logarithm = mapping.log_fgrad()
    _assert_maps_close(TpsaMap(d).exp_fgrad(logarithm), mapping, tol=1e-8)
    with pytest.raises(ValueError, match='Hamiltonian|representable|integrab'):
        logarithm.to_generator()


def test_logarithm_of_complex_kick_and_complex_result() -> None:
    d = Descriptor(variables=('log_cq', 'log_cp'), order=5)
    q, p = d.vars()
    mapping = TpsaMap([q, p + (0.01 + 0.005j) * q**2])
    logarithm = mapping.log_fgrad()
    assert logarithm.is_complex
    assert all(isinstance(t, ComplexTpsa) for t in logarithm.flat)
    _assert_maps_close(TpsaMap(d).exp_fgrad(logarithm), mapping, tol=1e-8)


def test_parameter_dependent_lie_logarithm() -> None:
    d = Descriptor(variables=('log_pq', 'log_pp'), parameters=('log_k',), order=5, param_order=2)
    q, p = d.vars()
    k = d.param('log_k')
    mapping = TpsaMap([q, p - 0.02 * k * q**2])
    logarithm = mapping.log_fgrad()
    _assert_maps_close(TpsaMap(d).exp_fgrad(logarithm), mapping, tol=1e-8)
    np.testing.assert_allclose(
        TpsaMap(d).exp_fgrad(logarithm).evaluate((0.1, 0.2), parameters=(0.5,)),
        mapping.evaluate((0.1, 0.2), parameters=(0.5,)),
        atol=1e-9,
    )


def test_explicit_initial_guess_and_wrong_type() -> None:
    mapping = _shear()
    zero = TpsaVectorField([mapping.descriptor.zero(), mapping.descriptor.zero()])
    _assert_maps_close(TpsaMap(mapping.descriptor).exp_fgrad(mapping.log_fgrad(zero)), mapping)
    with pytest.raises(TypeError, match='initial_guess'):
        mapping.log_fgrad(cast('Any', 42))


def test_validation_rejects_deliberately_corrupted_exponential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _shear()

    # Controlled difficult case: simulate an inaccurate native logarithm by
    # corrupting the re-exponentiation. This deterministically tests detection,
    # without depending on undocumented native iteration/convergence heuristics.
    def inaccurate_exp(self: TpsaMap, _generator: Any) -> TpsaMap:
        return TpsaMap(self.descriptor)

    monkeypatch.setattr(TpsaMap, 'exp_fgrad', inaccurate_exp)
    with pytest.raises(ValueError, match='logarithm|reconstruct'):
        mapping.log_fgrad(validate=True)
    unchecked = mapping.log_fgrad(validate=False)
    assert isinstance(unchecked, TpsaVectorField)


@pytest.mark.parametrize('bad', [-1.0, float('nan'), float('inf')])
def test_invalid_validation_tolerance_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match='tolerance|finite|non-negative'):
        _shear().log_fgrad(atol=bad)
    with pytest.raises(ValueError, match='tolerance|finite|non-negative'):
        _shear().log_fgrad(rtol=bad)


def test_challenging_nonlinear_input_either_reconstructs_or_fails_clearly() -> None:
    d = Descriptor(variables=('log_hard_q', 'log_hard_p'), order=5)
    q, p = d.vars()
    # A larger nonlinear perturbation can be difficult for the iterative
    # native logarithm. A successful return must still reconstruct the map.
    mapping = TpsaMap([1.3 * q + 0.08 * q**2, p])
    try:
        field = mapping.log_fgrad(validate=True, atol=1e-8, rtol=1e-8)
    except ValueError as exc:
        assert 'did not reconstruct' in str(exc)
    else:
        _assert_maps_close(TpsaMap(d).exp_fgrad(field), mapping, tol=1e-8)
