"""Composition, inversion, parameters, and Lie-map operations."""

from madng_tpsa import (
    Descriptor,
    TpsaMap,
    field_to_vector,
    vector_to_field,
)

descriptor = Descriptor(
    variables=['q', 'p'],
    order=5,
    params=['k'],
    param_order=2,
)

q, p = descriptor.vars()
k = descriptor.param('k')

kick = TpsaMap(
    {
        'q': q,
        'p': p - k * q**2,
    }
)

drift = TpsaMap(
    {
        'q': q + p,
        'p': p,
    }
)

print('Kick:')
print(kick)
print()

print('Drift:')
print(drift)
print()

# Composition convention:
# left @ right == left(right(z))
one_step = drift @ kick

print('Drift after kick:')
print(one_step)
print()

print('Evaluate for q=0.1, p=0.02, k=0.5:')
print(one_step.evaluate([0.1, 0.02], parameters=[0.5]))
print()

print('Formal inverse:')
inverse = one_step.inverse()
print(inverse)
print()

print('Compose map with its inverse:')
identity_check = one_step @ inverse
print(identity_check)
print()

generator = k * q**3 / 3

print('Scalar Lie generator:')
print('f =', generator.format('code'))
print()

field = vector_to_field(generator)
print('MAD-NG Hamiltonian vector field G = -J grad(f):')
print('G_q =', field[0].format('code'))
print('G_p =', field[1].format('code'))
print()

print('Convert field back to scalar generator:')
print(field_to_vector(field).format('code'))
print()

identity = TpsaMap.identity(descriptor)
# MAD-NG convention:
# identity.exp_poisson(generator) == exp(:generator:) identity
lie_map = identity.exp_poisson(generator)

print('Lie map exp(:f:) applied to the identity:')
print(lie_map)
print()

print('Recover the logarithmic generator:')
print(lie_map.log_generator().format('code'))
