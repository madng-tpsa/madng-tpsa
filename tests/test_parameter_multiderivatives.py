"""Parameter derivatives must preserve multiplicities in mixed partials."""

import numpy as np

from madng_tpsa import Descriptor, TpsaMap


def test_real_parameter_multiindex_matches_successive_derivatives() -> None:
    d = Descriptor(
        variables=('multi_q', 'multi_p'),
        parameters=('multi_k',),
        order=5,
        param_order=3,
    )
    q, p = d.vars()
    k = d.param('multi_k')
    f = 2 * k**2 * p + 3 * k**2 * q + 5 * k**3

    assert f.derivative((0, 1, 1)) == 4 * k
    assert f.derivative((0, 0, 2)) == 4 * p + 6 * q + 30 * k
    assert f.derivative((0, 0, 3)) == d.constant(30)
    assert f.derivative((0, 0, 2)) == f.derivative('multi_k').derivative('multi_k')
    assert f.derivative((0, 1, 1)) == f.derivative('multi_p').derivative('multi_k')


def test_complex_parameter_multiindex_matches_successive_derivatives() -> None:
    d = Descriptor(
        variables=('cmulti_q', 'cmulti_p'),
        parameters=('cmulti_k',),
        order=5,
        param_order=3,
    )
    q, p = d.vars()
    k = d.param('cmulti_k')
    f = (1 + 2j) * k**2 * q + (2 - 1j) * k * q * p + (3 - 4j) * k**3

    assert f.derivative((1, 0, 1)) == (2 + 4j) * k + (2 - 1j) * p
    assert f.derivative((0, 0, 2)) == (2 + 4j) * q + (18 - 24j) * k
    assert f.derivative((0, 0, 2)) == f.derivative('cmulti_k').derivative('cmulti_k')


def test_mixed_multiple_parameters() -> None:
    d = Descriptor(
        variables=('two_q', 'two_p'),
        parameters=('two_k', 'two_l'),
        order=5,
        param_order=3,
    )
    q, _ = d.vars()
    k = d.param('two_k')
    ell = d.param('two_l')
    f = (2 + 1j) * k**2 * ell * q

    assert f.derivative((0, 0, 2, 1)) == (4 + 2j) * q
    assert f.derivative((0, 0, 2, 1)) == (
        f.derivative('two_k').derivative('two_k').derivative('two_l')
    )


def test_tensor_parameter_hessian_inherits_scalar_fix() -> None:
    d = Descriptor(
        variables=('hpar_q', 'hpar_p'),
        parameters=('hpar_k',),
        order=5,
        param_order=3,
    )
    _, p = d.vars()
    k = d.param('hpar_k')
    m = TpsaMap([5 * k**2, 2 * k**2 * p])
    h = m.derivative_tensor(2, wrt='parameters')
    assert h.shape == (2, 1, 1)
    np.testing.assert_allclose(
        h.evaluate((2, 3), parameters=(4,)),
        [[[10]], [[12]]],
    )
    np.testing.assert_allclose(
        m.partial_derivative((0, 0, 2)).evaluate((2, 3), parameters=(4,)),
        [10, 12],
    )
