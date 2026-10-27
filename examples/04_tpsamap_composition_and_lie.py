"""Composition, inversion, parameters, and Lie-map operations."""

from madng_tpsa import Descriptor, TpsaMap, TpsaVectorField

descriptor = Descriptor(
    variables=['q', 'p'],
    order=5,
    parameters=['k'],
    param_order=2,
)

q, p = descriptor.vars()
k = descriptor.param('k')

kick = TpsaMap(
    q=q,
    p=p - k * q**2,
)

drift = TpsaMap(
    q=q + p,
    p=p,
)

print('Kick:')
print(kick)
print()
# TpsaMap(q = q, p = p - q²·k)

print('Drift:')
print(drift)
print()
# TpsaMap(q = q + p, p = p)

# Composition convention:
# left << right == left(right(z))
one_step = drift << kick

print('Drift after kick:')
print(one_step)
print()
# TpsaMap(q = q + p - q²·k, p = p - q²·k)

print('Evaluate for q=0.1, p=0.02, k=0.5:')
print(one_step.evaluate([0.1, 0.02], parameters=[0.5]))
print()
# [0.115 0.015]

print('Formal inverse:')
inverse = one_step.inverse()
print(inverse)
print()
# TpsaMap(q = q - p, p = p + q²·k - 2·q·p·k + ...)

print('Compose map with its inverse:')
identity_check = one_step << inverse
print(identity_check)
print()
# TpsaMap(q = q, p = p)

generator = k * q**3 / 3

print('Scalar Lie generator:')
print('f =', generator.format('code'))
print()
# f = 0.333333 * q**3 * k

field = TpsaVectorField.from_generator(generator)
print('MAD-NG Hamiltonian vector field G = -S grad(f):')
print('G_q =', field[0].format('code'))
print('G_p =', field[1].format('code'))
print()
# G_q = 0
# G_p = q**2 * k

print('Convert field back to scalar generator:')
print(field.to_generator().format('code'))
print()
# 0.333333 * q**3 * k

identity = TpsaMap(descriptor)
# MAD-NG convention:
# identity.exp_poisson(generator) == exp(:generator:) identity
lie_map = identity.exp_poisson(generator)

print('Lie map exp(:f:) applied to the identity:')
print(lie_map)
print()
# TpsaMap(q = q, p = p - q²·k)

assert lie_map.q == q
assert lie_map.p == p - k * q**2

print('Recover the logarithmic generator:')
print(lie_map.log_generator().format('code'))
# 0.333333 * q**3 * k
