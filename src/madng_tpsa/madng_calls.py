"""Bind the generated C wrappers to Python function calls."""

from ._cffi import ffi, lib
from .madng_definitions import DEFINITIONS
from .madng_function import MadngFunction


def _bind(name):
    declaration, options = DEFINITIONS[name]
    return MadngFunction(declaration, ffi=ffi, lib=lib, **options)  # type: ignore


def new_components(reference, size):
    """Allocate zero TPSAs matching a tensor's descriptor and flavour."""
    descriptor = reference.descriptor
    zero = descriptor.complex_zero if reference.is_complex else descriptor.zero
    order = descriptor.order
    return [zero(order=order) for _ in range(size)]


MINV = _bind('minv')
COMPOSE = _bind('compose')
LIEBRA = _bind('liebra')
VEC2FLD = _bind('vec2fld')
