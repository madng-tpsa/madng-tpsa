"""Build, inspect, evaluate, and manipulate a TpsaMap."""

import numpy as np

from madng_tpsa import Descriptor, TpsaMap

descriptor = Descriptor(
    variables=['x', 'px'],
    order=4,
)

x, px = descriptor.vars()

turn_map = TpsaMap(
    {
        'x': x + px + 0.2 * x**2,
        'px': px - 0.3 * x + 0.1 * x * px,
    }
)

print('Map:')
print(turn_map)
print()

print('Coordinate names:')
print(turn_map.coord_names)
print()

print('Access by name:')
print('x  ->', turn_map['x'].format('code'))
print('px ->', turn_map.px.format('code'))
print()

print('Constant part:')
print(turn_map.const_part)
print()

print('Jacobian at the expansion point:')
print(turn_map.jacobian())
print()

point = [1e-3, 2e-3]
print('Evaluate at', point)
print(turn_map.evaluate(point))
print()

print('Jacobian at', point)
print(turn_map.jacobian(point))
print()

print('Quadratic part only:')
quadratic = turn_map.homogeneous(2)
print('x  ->', quadratic.x.format('code'))
print('px ->', quadratic.px.format('code'))
print()

print('Truncated through first order:')
linear = turn_map.truncate(1)
print(linear)
print()

print('Map norm:')
print(turn_map.norm())
print()

print('Coefficient of x*px in px output:')
print(turn_map.coefficient('px', (1, 1)))

print()
print('Update that coefficient:')
modified = turn_map.copy()
modified.set_coefficient('px', (1, 1), 0.25)
print(modified.px.format('code'))

print()
print('Identity map:')
identity = TpsaMap.identity(descriptor)
print(identity)

print()
print('Numerical check of the identity:')
np.testing.assert_allclose(identity.evaluate(point), point)
print('OK')
