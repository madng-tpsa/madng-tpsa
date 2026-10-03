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

f = 2 + 3 * x - y + 4 * x * y + 2 * y**2 + x**3

print('Polynomial f:')
print('repr:  ', f)
print('code:  ', f.format('code'))
print('math:  ', f.format('math'))
print()

print('Coefficient dictionary:')
print(f.to_dict())
print()

print('Selected coefficients:')
print('constant    =', f.const_part)
print('coeff[x]    =', f[(1, 0)])
print('coeff[x*y]  =', f[(1, 1)])
print('coeff[y**2] =', f[(0, 2)])
print()

print('First-order coefficients:')
print('grad(f) =', f.grad())
print()

print('Order information:')
print('allocated order   =', f.order)
print('highest used order =', f.max_nonzero_order)
print()

print('Homogeneous order 2:')
print(f.homogeneous(2).format('code'))
print()

print('Truncated through order 2:')
print(f.truncate(2).format('code'))
print()

print('Some algebra and elementary functions:')
g = (1 + x) ** 2
print('g =', g.format('code'))
print('sqrt(g) =', np.sqrt(g).format('code'))
print('exp(x) =', np.exp(x).format('code'))
print()

print('Rich coefficient table:')
console.print(f.format('table'))
