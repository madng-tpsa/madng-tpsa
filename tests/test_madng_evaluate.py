"""Integration tests for generated real and complex MAD-NG evaluation."""

import numpy as np
import pytest

from madng_tpsa import Descriptor, TpsaTensor


def test_generated_evaluate_real_complex_and_batching():
    """Evaluate tensors directly into correctly shaped NumPy arrays."""
    desc = Descriptor(variables=('x', 'px'), parameters=('k',), order=3, param_order=2)
    x, px = desc.vars()
    (k,) = desc.params()
    # Six components but only two descriptor variables: multiple native calls.
    tensor = TpsaTensor([x, px, k, x + 2 * px + k, x * px, x * x], shape=(2, 3))
    result = tensor.evaluate([1.5, -0.25], [0.7])
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, np.array([1.5, -0.25, 0.7, 1.7, -0.375, 2.25]).reshape(2, 3))

    ctensor = TpsaTensor([x + 2j * px, px + 1j * k, 1j * x])
    coords = [0.5 + 0.3j, -0.25 + 0.2j]
    params = [0.2 + 0.4j]
    cresult = ctensor.evaluate(coords, params)
    assert cresult.dtype == np.complex128
    np.testing.assert_allclose(
        cresult,
        [coords[0] + 2j * coords[1], coords[1] + 1j * params[0], 1j * coords[0]],
    )

    with pytest.warns(UserWarning, match='complex coordinates'):
        promoted = tensor.evaluate([1.5 + 0.25j, -0.25], [0.7])
    assert promoted.dtype == np.complex128
    np.testing.assert_allclose(promoted[0, :], [1.5 + 0.25j, -0.25, 0.7])
