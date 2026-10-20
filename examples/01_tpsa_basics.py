"""Basic TPSA construction, arithmetic, coefficients, and formatting."""

import numpy as np
from rich.console import Console

from madng_tpsa import Descriptor

console = Console()

# Two variables, expanded through total order 4.
descriptor = Descriptor(
    variables=['x', 'y'],
    order=4,
)

x, y = descriptor.vars()

print('Variables:')
print('x =', x)
print('y =', y)
print()
# x = Tpsa({(1, 0): 1.0})
# y = Tpsa({(0, 1): 1.0})

f = 2 + 3 * x - y + 4 * x * y + 2 * y**2 + x**3

print('Polynomial f:')
print('repr:  ', f)
print('code:  ', f.format('code'))
print('math:  ', f.format('math'))
print()
# repr:   Tpsa({(0, 0): 2.0, (1, 0): 3.0, (0, 1): -1.0, (1, 1): 4.0, (0, 2): 2.0, (3, 0): 1.0})
# code:   2 + 3 * x - y + 4 * x * y + 2 * y**2 + x**3
# math:   <IPython.core.display.Math object>

print('Coefficient dictionary:')
print(f.to_dict())
print()
# {
#  '__class__': 'Tpsa', 'descriptor': {'__class__': 'Descriptor', 'variables': ('x', 'y'),
#                                      'parameters': (), 'order': 4, 'param_order': 0,
#                                      'max_orders': None},
#  'truncation_order': 4, '[0, 0]': 2.0, '[1, 0]': 3.0, '[0, 1]': -1.0, '[1, 1]': 4.0,
#  '[0, 2]': 2.0, '[3, 0]': 1.0
# }

print('Selected coefficients:')
print('constant    =', f.const_part)
print('coeff[x]    =', f[(1, 0)])
print('coeff[x*y]  =', f[(1, 1)])
print('coeff[y**2] =', f[(0, 2)])
print()
# constant    = 2.0
# coeff[x]    = 3.0
# coeff[x*y]  = 4.0
# coeff[y**2] = 2.0

print('First-order coefficients:')
print('grad(f) =', f.grad())
print()
# grad(f) = [3.0, -1.0]

print('Order information:')
print('allocated order   =', f.truncation_order)
print('highest used order =', f.degree)
print()
# allocated order   = 4
# highest used order = 3

print('Homogeneous order 2:')
print(f.homogeneous(2).format('code'))
print()
# 4 * x * y + 2 * y**2

print('Truncated through order 2:')
print(f.truncate(2).format('code'))
print()
# 2 + 3 * x - y + 4 * x * y + 2 * y**2

print('Some algebra and elementary functions:')
g = (1 + x) ** 2
print('g =', g.format('code'))
print('sqrt(g) =', np.sqrt(g).format('code'))
print('exp(x) =', np.exp(x).format('code'))
print()
# g = 1 + 2 * x + x**2
# sqrt(g) = 1 + x
# exp(x) = 1 + x + 0.5 * x**2 + 0.166667 * x**3 + 0.0416667 * x**4

print('Rich coefficient table:')
console.print(f.format('table'))
# ┏━━━┳━━━┳━━━━━━━━━━━━━┓
# ┃ x ┃ y ┃ coefficient ┃
# ┡━━━╇━━━╇━━━━━━━━━━━━━┩
# │ 0 │ 0 │           2 │
# │ 1 │ 0 │           3 │
# │ 0 │ 1 │          -1 │
# │ 1 │ 1 │           4 │
# │ 0 │ 2 │           2 │
# │ 3 │ 0 │           1 │
# └───┴───┴─────────────┘
