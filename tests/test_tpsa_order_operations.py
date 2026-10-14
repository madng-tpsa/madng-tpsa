import json

import numpy as np
import pytest

import madng_tpsa
from madng_tpsa._tpsa_base import _TpsaBase


@pytest.fixture(params=['real', 'complex'])
def polynomial(request):
    descriptor = madng_tpsa.Descriptor(2, 4)

    if request.param == 'complex':
        coefficients = {
            (0, 0): 1 + 2j,
            (1, 0): 2 - 1j,
            (1, 1): 3 + 4j,
            (2, 1): -5 + 2j,
            (0, 4): 7 - 3j,
        }
        tpsa = madng_tpsa.ComplexTpsa.from_dict(
            {
                '__class__': 'ComplexTpsa',
                'descriptor': descriptor.to_dict(),
                'truncation_order': 4,
                **{json.dumps(k): [v.real, v.imag] for k, v in coefficients.items()},
            }
        )
        return tpsa, coefficients

    coefficients_real = {
        (0, 0): 1.0,
        (1, 0): 2.0,
        (1, 1): 3.0,
        (2, 1): -5.0,
        (0, 4): 7.0,
    }
    tpsa = madng_tpsa.Tpsa.from_dict(
        {
            '__class__': 'Tpsa',
            'descriptor': descriptor.to_dict(),
            'truncation_order': 4,
            **{json.dumps(k): v for k, v in coefficients_real.items()},
        }
    )
    return tpsa, coefficients_real


def test_homogeneous_extracts_requested_order(polynomial):
    tpsa, coefficients = polynomial
    result = tpsa.homogeneous(3)
    assert result.truncation_order == 3
    assert result.degree == 3
    assert result.monomial_coeffs() == pytest.approx({(2, 1): coefficients[(2, 1)]})


def test_homogeneous_extracts_constant_part(polynomial):
    tpsa, coefficients = polynomial
    result = tpsa.homogeneous(0)
    assert result.truncation_order == 0
    assert result.degree == 0
    assert result.monomial_coeffs() == pytest.approx({(0, 0): coefficients[(0, 0)]})


def test_homogeneous_empty_order_is_zero(polynomial):
    tpsa, _ = polynomial
    # A separate descriptor is easier for testing an empty valid order.
    descriptor = madng_tpsa.Descriptor(2, 4)
    empty = (
        descriptor.complex_zero(order=4)
        if isinstance(tpsa, madng_tpsa.ComplexTpsa)
        else descriptor.zero(order=4)
    )
    empty[(1, 0)] = 1
    result = empty.homogeneous(3)
    assert result.truncation_order == 3
    assert result.is_zero()


def test_truncate_keeps_orders_through_requested_order(polynomial):
    tpsa, coefficients = polynomial
    result = tpsa.truncate(2)
    assert result.truncation_order == 2
    assert result.degree == 2
    assert result.monomial_coeffs() == pytest.approx(
        {
            (0, 0): coefficients[(0, 0)],
            (1, 0): coefficients[(1, 0)],
            (1, 1): coefficients[(1, 1)],
        }
    )


def test_truncate_to_zero_keeps_only_constant(polynomial):
    tpsa, coefficients = polynomial
    result = tpsa.truncate(0)
    assert result.truncation_order == 0
    assert result.monomial_coeffs() == pytest.approx(
        {
            (0, 0): coefficients[(0, 0)],
        }
    )


def test_truncate_does_not_modify_original(polynomial):
    tpsa, coefficients = polynomial
    result = tpsa.truncate(2)
    assert result is not tpsa
    assert tpsa.truncation_order == 4
    assert tpsa.monomial_coeffs() == coefficients


def test_clear_order_removes_only_requested_order(polynomial):
    tpsa, coefficients = polynomial
    result = tpsa.clear_order(2)
    expected = dict(coefficients)
    del expected[(1, 1)]
    assert result.truncation_order == 4
    assert result.degree == 4
    assert result.monomial_coeffs() == expected


def test_clear_order_zero_removes_constant(polynomial):
    tpsa, coefficients = polynomial
    result = tpsa.clear_order(0)
    expected = dict(coefficients)
    del expected[(0, 0)]
    assert result.truncation_order == 4
    assert result.monomial_coeffs() == expected


def test_clear_order_does_not_modify_original(polynomial):
    tpsa, coefficients = polynomial
    result = tpsa.clear_order(3)
    assert result is not tpsa
    assert tpsa.monomial_coeffs() == coefficients


def test_copy_preserves_lower_allocated_order():
    descriptor = madng_tpsa.Descriptor(2, 6)
    real = descriptor.zero(order=3)
    real[(2, 0)] = 1.0
    complex_ = descriptor.complex_zero(order=2)
    complex_[(1, 1)] = 1 + 2j
    real_copy = real.copy()
    complex_copy = complex_.copy()
    assert real_copy.truncation_order == 3
    assert complex_copy.truncation_order == 2
    assert real_copy == real
    assert complex_copy == complex_


@pytest.mark.parametrize(
    'method_name',
    ['homogeneous', 'truncate', 'clear_order'],
)
def test_order_operations_use_common_validation(method_name):
    descriptor = madng_tpsa.Descriptor(2, 4)
    tpsa = descriptor.zero()
    with pytest.raises(TypeError, match='degree must be an integer'):
        getattr(tpsa, method_name)(1.5)
    with pytest.raises(ValueError, match='degree must satisfy'):
        getattr(tpsa, method_name)(5)


@pytest.mark.parametrize(
    ('degree', 'expected'),
    [
        (0, 0),
        (1, 1),
        (3, 3),
    ],
)
def test_validate_order_accepts_valid_integer(degree, expected):
    assert _TpsaBase._validate_degree(degree, 3) == expected


@pytest.mark.parametrize('order', [-1, 4])
def test_validate_degree_rejects_out_of_range(order):
    with pytest.raises(ValueError, match='degree must satisfy'):
        _TpsaBase._validate_degree(order, 3)


@pytest.mark.parametrize('order', [1.5, '2', True, None])
def test_validate_degree_rejects_non_integer(order):
    with pytest.raises(TypeError, match='degree must be an integer'):
        _TpsaBase._validate_degree(order, 3)


def test_validate_degree_accepts_numpy_integer():
    assert _TpsaBase._validate_degree(np.int64(2), 3) == 2
