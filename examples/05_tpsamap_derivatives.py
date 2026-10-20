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
print('d^3 map / dq^2 dk:', by_name.evaluate(point, parameters=parameters))
print()
# d^3 map / dq^2 dk: [2. 0.]

# A 1-based integer index or coordinate/parameter name picks one direction.
print('d map / dk:', mapping.partial_derivative('k').evaluate(point, parameters=parameters))
print('d map / dq:', mapping.partial_derivative(1).evaluate(point, parameters=parameters))
print()
# d map / dk: [4. 3.]
# d map / dq: [34.  9.]

# The full derivative tensors append one axis per differentiation order.
jacobian_tpsa = mapping.jacobian_operator()
hessian_tpsa = mapping.hessian_operator()
assert jacobian_tpsa.shape == (2, 2)
assert hessian_tpsa.shape == (2, 2, 2)
print('Jacobian operator shape:', jacobian_tpsa.shape)
print('Hessian operator shape:', hessian_tpsa.shape)
print()
# Jacobian operator shape: (2, 2)
# Hessian operator shape: (2, 2, 2)

print('Jacobian at point:')
print(mapping.jacobian(point, parameters=parameters))
print()
# [[34.  4.]
#  [ 9. 16.]]

print('Hessian at point:')
print(mapping.hessian(point, parameters=parameters))
print()
# [[[20.  2.]
#   [ 2.  0.]]
#
#  [[ 0.  6.]
#   [ 6.  4.]]]

# Include parameter derivatives as additional directions.
full_hessian = mapping.hessian_operator(to_parameters=True)
param_hessian = mapping.derivative_tensor(2, to_variables=False, to_parameters=True)
print('Hessian incl. parameter axes shape:', full_hessian.shape)
print('Parameter Hessian shape:', param_hessian.shape)
print('Mixed q-k entry:', full_hessian.evaluate(point, parameters=parameters)[0, 0, 2])
print()
# Hessian incl. parameter axes shape: (2, 3, 3)
# Parameter Hessian shape: (2, 1, 1)
# Mixed q-k entry: 4.0

third = mapping.derivative_tensor(3, to_parameters=True)
assert third.shape == (2, 3, 3, 3)
print('Full third-derivative tensor shape:', third.shape)
print()
# Full third-derivative tensor shape: (2, 3, 3, 3)

# Rank independence: the same operations work on arbitrary tensors.
matrix = TpsaTensor([[q**2 * k, q * p], [k * p**2, p**3]])
differentiated_matrix = matrix.partial_derivative((('q', 2), 'k'))
assert differentiated_matrix.shape == (2, 2)
higher_matrix = matrix.derivative_tensor(2, to_parameters=True)
assert higher_matrix.shape == (2, 2, 3, 3)
print('Mixed derivative of a rank-two tensor:')
print(differentiated_matrix.evaluate(point, parameters=parameters))
print()
# [[2. 0.]
#  [0. 0.]]

print('Rank-two tensor Hessian shape:', higher_matrix.shape)
print()
# Rank-two tensor Hessian shape: (2, 2, 3, 3)
