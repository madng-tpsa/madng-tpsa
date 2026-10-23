"""TPSA differentiation, integration, parameters, and Poisson brackets."""

from madng_tpsa import Descriptor

# Canonical coordinates are taken in pairs: (q, p).
descriptor = Descriptor(
    variables=['q', 'p'],
    order=5,
    parameters=['k'],
    param_order=2,
)

q, p = descriptor.vars()
k = descriptor.param('k')

hamiltonian = (p**2 + q**2) / 2 + k * q**3 / 3

print('Hamiltonian:')
print('H =', hamiltonian.format('code'))
print()
# H = 0.5 * q**2 + 0.5 * p**2 + 0.333333 * q**3 * k

print('Parameter derivative:')
print('dH/dk =', hamiltonian.derivative('k').format('code'))
print()
# dH/dk = 0.333333 * q**3

print('First-order parameter coefficient at the expansion point:')
print('param_grad(H) =', hamiltonian.param_grad())
print()
# param_grad(H) = [0.0]

dq = hamiltonian.derivative('q')
dp = hamiltonian.derivative('p')

print('Partial derivatives:')
print('dH/dq =', dq.format('code'))
print('dH/dp =', dp.format('code'))
print()
# dH/dq = q + q**2 * k
# dH/dp = p

print('Formal integration:')
print('integral(dH/dq, dq) =', dq.integrate('q').format('code'))
print()
# integral(dH/dq, dq) = 0.5 * q**2 + 0.333333 * q**3 * k

# Poisson bracket in canonical coordinates:
# {f, g} = df/dq dg/dp - df/dp dg/dq
print('Poisson brackets:')
print('[q, p] =', q.poisson_bracket(p).format('code'))
print('[p, q] =', p.poisson_bracket(q).format('code'))
print('[H, H] =', hamiltonian.poisson_bracket(hamiltonian).format('code'))
print()
# [q, p] = 1
# [p, q] = -1
# [H, H] = 0

observable = q**2 + p**2
print('Observable:')
print('A =', observable.format('code'))
print('[A, H] =', observable.poisson_bracket(hamiltonian).format('code'))
# A = q**2 + p**2
# [A, H] = -2 * q**2 * p * k
