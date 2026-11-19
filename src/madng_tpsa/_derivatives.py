"""Normalise derivative directions into descriptor-ordered exponent tuples."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._typing import DerivativeDirection, DerivativeSpec
    from .descriptor import Descriptor


def _direction_index(descriptor: Descriptor, direction: DerivativeDirection) -> int:
    """Return a 1-based index for a named, numbered, or unit TPSA direction."""
    # Local imports avoid cycles with the public scalar TPSA modules.
    from .complex_tpsa import ComplexTpsa
    from .tpsa import Tpsa

    if isinstance(direction, (Tpsa, ComplexTpsa)):
        if direction.descriptor is not descriptor:
            err_mess = 'Derivative direction has an incompatible descriptor'
            raise ValueError(err_mess)
        # Ignore the expansion centre, but reject nonlinear or non-unit series.
        terms = [
            (exponents, coeff)
            for exponents, coeff in direction.monomial_coeffs(tol=0).items()
            if any(exponents)
        ]
        if len(terms) != 1 or terms[0][1] != 1 or sum(terms[0][0]) != 1:
            err_mess = 'Derivative direction TPSA must have exactly one unit first-order monomial'
            raise ValueError(err_mess)
        return terms[0][0].index(1) + 1
    if isinstance(direction, str):
        return descriptor.var_or_param_index(direction)
    if isinstance(direction, Integral) and not isinstance(direction, (bool, np.bool_)):
        return descriptor.var_or_param_index(int(direction))
    err_mess = 'Derivative direction must be a name, 1-based index, or unit TPSA'
    raise TypeError(err_mess)


def resolve_derivative_orders(
    descriptor: Descriptor,
    *wrt: DerivativeSpec,
    orders: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Return a nonzero exponent tuple for a collection of derivative directions.

    ``*wrt`` comprises directions and ``(direction, multiplicity)`` pairs.
    An explicit full exponent vector must be given as ``orders=``.
    Exceeding descriptor truncation limits is valid: the derivative is zero.
    """
    size = descriptor.monomial_length
    if orders is not None:
        if wrt:
            err_mess = 'Cannot combine derivative directions with orders='
            raise TypeError(err_mess)
        if len(orders) != size:
            err_mess = f'Derivative orders must have length {size}'
            raise ValueError(err_mess)
        if any(not isinstance(v, Integral) or isinstance(v, (bool, np.bool_)) for v in orders):
            err_mess = 'Derivative orders must be integers, not booleans'
            raise TypeError(err_mess)
        parsed = tuple(int(v) for v in orders)
        if any(v < 0 for v in parsed):
            err_mess = 'Derivative orders must be nonnegative'
            raise ValueError(err_mess)
    else:
        if not wrt:
            err_mess = 'Specify at least one derivative direction'
            raise ValueError(err_mess)
        powers = [0] * size
        for spec in wrt:
            if isinstance(spec, tuple):
                if len(spec) != 2:
                    err_mess = 'Derivative pair must be (direction, multiplicity)'
                    raise ValueError(err_mess)
                direction, count = spec
            else:
                direction, count = spec, 1
            if not isinstance(count, Integral) or isinstance(count, (bool, np.bool_)):
                err_mess = 'Derivative multiplicity must be an integer'
                raise TypeError(err_mess)
            if count <= 0:
                err_mess = 'Derivative multiplicity must be positive'
                raise ValueError(err_mess)
            powers[_direction_index(descriptor, direction) - 1] += int(count)
        parsed = tuple(powers)
    if not any(parsed):
        err_mess = 'Derivative specification must have positive degree'
        raise ValueError(err_mess)
    return parsed
