"""Additional regression coverage for map ordering and derivative tensors."""

import numpy as np

from madng_tpsa import Descriptor, TpsaMap, TpsaTensor


def test_named_map_components_follow_descriptor_order() -> None:
    d = Descriptor(variables=('review_q', 'review_p'), order=4)
    q, p = d.vars()
    mapping = TpsaMap(review_p=p + q, review_q=q + p**2)
    assert mapping.coord_names == ('review_q', 'review_p')
    assert mapping.coords[0] == q + p**2
    assert mapping.coords[1] == p + q


def test_mixed_parameter_derivative_tensor_is_symmetric_but_not_aliased() -> None:
    d = Descriptor(
        variables=('review_q2', 'review_p2'),
        parameters=('review_k2',),
        order=5,
        param_order=2,
    )
    q, p = d.vars()
    k = d.param('review_k2')
    matrix = TpsaTensor([[q * k**2, p], [q**2 * k, p * k]])
    hessian = matrix.derivative_tensor(2, to_parameters=True)
    assert hessian.shape == (2, 2, 3, 3)
    values = hessian.evaluate((2, 3), parameters=(4,))
    assert values[0, 0, 0, 2] == 8
    assert values[0, 0, 2, 0] == 8
    assert values[0, 0, 2, 2] == 4
    assert values[0, 0, 0, 2] == values[0, 0, 2, 0]

    # Symmetry is mathematical, but mutable TPSA components must be independent.
    first = hessian[0, 0, 0, 2]
    second = hessian[0, 0, 2, 0]
    assert first is not second
    before = second.const_part
    first.const_part = 123.0
    assert second.const_part == before
    np.testing.assert_allclose(matrix.evaluate((2, 3), parameters=(4,)), [[32, 3], [16, 12]])


def test_tensor_partial_and_full_derivative_consistency() -> None:
    d = Descriptor(variables=('review_q3', 'review_p3'), order=5)
    q, p = d.vars()
    m = TpsaMap([q**3 + q * p**2, q * p**3])
    hessian = m.hessian_operator()
    selected = m.partial_derivative((1, 1))
    for i, component in enumerate(selected.coords):
        assert component == hessian[i, 0, 1]
