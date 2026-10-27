"""Examples of TPSA-valued Jacobians, Hessians, and generic tensor derivatives."""

import numpy as np

from madng_tpsa import Descriptor, TpsaMap, TpsaTensor

descriptor = Descriptor(variables=('q', 'p'), parameters=('k',), order=5, param_order=3)
q, p = descriptor.vars()
k = descriptor.param('k')

mapping = TpsaMap([q**3 + k * q**2 + 2 * q * p, q * p**2 + k * p])
point = (2.0, 3.0)
parameters = (4.0,)

# General partials apply the same derivative to each component and keep
# the existing tensor shape (a map therefore remains a map).
by_name = mapping.partial_derivative((('q', 2), 'k'))
by_exponents = mapping.partial_derivative((2, 0, 1))
assert isinstance(by_name, TpsaMap)
assert by_name.shape == mapping.shape
np.testing.assert_allclose(
    by_name.evaluate(point, parameters=parameters),
    by_exponents.evaluate(point, parameters=parameters),
)
print('d^3 map / dq^2 dk:', by_name.evaluate(point, parameters=parameters))  # [2. 0.]
print()

# A 1-based integer index or coordinate/parameter name picks one direction.
print(
    'd map / dk:', mapping.partial_derivative('k').evaluate(point, parameters=parameters)
)  # [4. 3.]
print(
    'd map / dq:', mapping.partial_derivative(1).evaluate(point, parameters=parameters)
)  # [34.  9.]
print()

# The full derivative tensors append one axis per differentiation order.
jacobian_tpsa = mapping.jacobian()
hessian_tpsa = mapping.hessian()
assert jacobian_tpsa.shape == (2, 2)
assert hessian_tpsa.shape == (2, 2, 2)

print('Jacobian shape:', jacobian_tpsa.shape)  # (2, 2)
print('Full Jacobian:', jacobian_tpsa)
print()
# TpsaTensor(shape=(2, 2), size=4, values=[2·p + 3·q² + 2·q·k, 2·q, p², ...])

print('Hessian shape:', hessian_tpsa.shape)  # (2, 2, 2)
print('Full Hessian:', hessian_tpsa)
print()
# TpsaTensor(shape=(2, 2, 2), size=8, values=[6·q + 2·k, 2, 2, ...])

print('Jacobian at point:')
print(mapping.evaluate_jacobian(point, parameters=parameters))
print()
# [[34.  4.]
#  [ 9. 16.]]

print('Hessian at point:')
print(mapping.evaluate_hessian(point, parameters=parameters))
print()
# [[[20.  2.]
#   [ 2.  0.]]
#
#  [[ 0.  6.]
#   [ 6.  4.]]]

# Include parameter derivatives as additional directions.
full_hessian = mapping.hessian(wrt='all')
param_hessian = mapping.derivative_tensor(2, wrt='parameters')
print('Hessian incl. parameter axes shape:', full_hessian.shape)  # (2, 3, 3)
print('Parameter Hessian shape:', param_hessian.shape)  # (2, 3, 3)
print('Mixed q-k entry:', full_hessian.evaluate(point, parameters=parameters)[0, 0, 2])  # 4.0
print()

third = mapping.derivative_tensor(3, wrt='all')
assert third.shape == (2, 3, 3, 3)
print('Full third-derivative tensor shape:', third.shape)  # (2, 3, 3, 3)
print()

# Rank independence: the same operations work on arbitrary tensors.
matrix = TpsaTensor([[q**2 * k, q * p], [k * p**2, p**3]])
differentiated_matrix = matrix.partial_derivative((('q', 2), 'k'))
assert differentiated_matrix.shape == (2, 2)
higher_matrix = matrix.derivative_tensor(2, wrt='all')
assert higher_matrix.shape == (2, 2, 3, 3)
print('Mixed derivative of a rank-two tensor:')
print(differentiated_matrix.evaluate(point, parameters=parameters))
print()
# [[2. 0.]
#  [0. 0.]]

print('Rank-two tensor Hessian shape:', higher_matrix.shape)  # (2, 2, 3, 3)
print()
