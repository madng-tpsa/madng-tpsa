"""Validated Lie logarithms and their non-Hamiltonian generalisation."""

import numpy as np

from madng_tpsa import Descriptor, TpsaMap, TpsaVectorField


def close_as_tpsa(left: TpsaMap, right: TpsaMap, tol: float = 1e-9) -> bool:
    """Check the represented coefficient maps, not a single phase-space point."""
    for a, b in zip(left.flat, right.flat, strict=True):
        x = a.monomial_coeffs(tol=0)
        y = b.monomial_coeffs(tol=0)
        if any(
            not np.isclose(x.get(m, 0), y.get(m, 0), atol=tol, rtol=tol)
            for m in x.keys() | y.keys()
        ):
            return False
    return True


d = Descriptor(variables=('lie_q', 'lie_p'), order=5)
q, p = d.vars()

# An exactly solvable symplectic shear.
kick = TpsaMap([q, p - 0.01 * q**2])
log_field = kick.log_poisson()  # Validates the native iteration by default.
recovered = TpsaMap(d).exp_grad(log_field)
assert close_as_tpsa(kick, recovered)
print('Kick reconstructs:', close_as_tpsa(kick, recovered))

# log_poisson is more general than log_generator: a vector field can have
# nonzero divergence and generate a non-symplectic coordinate transformation.
field = TpsaVectorField([0.01 * q**2, d.zero()])
non_symplectic = TpsaMap(d).exp_grad(field)
recovered_field = non_symplectic.log_poisson()
assert close_as_tpsa(non_symplectic, TpsaMap(d).exp_grad(recovered_field), tol=1e-8)
print('Non-Hamiltonian map reconstructs: Success.')

# For a higher-level algorithm that does its own re-exponentiation comparison,
# the extra check can be omitted deliberately, but not silently.
unchecked_field = kick.log_poisson(validate=False)
assert close_as_tpsa(kick, TpsaMap(d).exp_grad(unchecked_field))
print('Unchecked logarithm explicitly verified: Success.')
