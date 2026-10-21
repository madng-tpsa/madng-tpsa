"""Build, inspect, evaluate, and manipulate a TpsaMap."""

import numpy as np

from madng_tpsa import Descriptor, TpsaMap

descriptor = Descriptor(
    variables=['x', 'px'],
    order=4,
)

x, px = descriptor.vars()

turn_map = TpsaMap(
    x=x + px + 0.2 * x**2,
    px=px - 0.3 * x + 0.1 * x * px,
)

print('Map:')
print(turn_map)
print()
# TpsaMap(
#     x=Tpsa({(1, 0): 1.0, (0, 1): 1.0, (2, 0): 0.2}),
#     px=Tpsa({(1, 0): -0.3, (0, 1): 1.0, (1, 1): 0.1})
# )

print('Coordinate names:')
print(turn_map.component_names)
print()
# ('x', 'px')

print('Access by name:')
print('x  ->', turn_map.x.format('code'))
print('px ->', turn_map.px.format('code'))
print()
# x  -> x + px + 0.2 * x**2
# px -> -0.3 * x + px + 0.1 * x * px


print('Constant part:')
print(turn_map.const_part)
print()
# [0. 0.]

print('Jacobian at the expansion point:')
print(turn_map.evaluate_jacobian())
print()
# [[ 1.   1. ]
#  [-0.3  1. ]]

point = [1e-3, 2e-3]
print('Evaluate at', point)
print(turn_map.evaluate(point))
print()
# [0.0030002 0.0017002]

print('Jacobian at', point)
print(turn_map.evaluate_jacobian(point))
print()
# [[ 1.0004  1.    ]
#  [-0.2998  1.0001]]

print('Quadratic part only:')
quadratic = turn_map.homogeneous(2)
print('x  ->', quadratic.x.format('code'))
print('px ->', quadratic.px.format('code'))
print()
# x  -> 0.2 * x**2
# px -> 0.1 * x * px

print('Truncated through first order:')
linear = turn_map.truncate(1)
print(linear)
print()
# TpsaMap(x=Tpsa({(1, 0): 1.0, (0, 1): 1.0}), px=Tpsa({(1, 0): -0.3, (0, 1): 1.0}))

print('Map norm:')
print(turn_map.norm())
print()
# 3.6000000000000005

print('Coefficient of x*px in px output:')
print(turn_map.coefficient('px', (1, 1)))
# 0.1

print()
print('Update that coefficient:')
modified = turn_map.copy()
modified.set_coefficient('px', (1, 1), 0.25)
print(modified.px.format('code'))
# -0.3 * x + px + 0.25 * x * px

print()
print('Identity map:')
identity = TpsaMap(descriptor)
print(identity)
# TpsaMap(x=Tpsa({(1, 0): 1.0}), px=Tpsa({(0, 1): 1.0}))

print()
print('Numerical check of the identity:')
np.testing.assert_allclose(identity.evaluate(point), point)
print('OK')
# OK
