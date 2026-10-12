"""madng_tpsa: Python bindings to the GTPSA engine of MAD-NG.

Truncated Power Series Algebra: build a descriptor defining the variables,
truncation order and optional parameters, seed identity series on it, and
perform algebraic and map operations. Stored coefficients are Taylor
coefficients; first-order coefficients coincide with derivatives at the
expansion point.

    import madng_tpsa
    d = madng_tpsa.Descriptor(2, 3)          # 2 variables, order 3
    x = d.var(1, 0.5)           # x, expanded around 0.5
    y = d.var(2)
    f = x * x + 2.0 * y
    f.const_part, f.grad(), f.monomial_coeffs()
"""

from ._cffi import CDEF, ffi, lib
from ._version import __version__
from .complex_tpsa import ComplexTpsa
from .descriptor import Descriptor
from .errors import TpsaError
from .functions import (
    compose,
    evaluate,
    exp_poisson,
    field_to_vector,
    inverse,
    lie_bracket,
    log_generator,
    log_poisson,
    partial_inverse,
    pullback,
    translate,
    vector_to_field,
)
from .maps import TpsaMap
from .paths import core_library, include_dir
from .tpsa import Tpsa

__all__ = [
    'CDEF',
    'ComplexTpsa',
    'Descriptor',
    'Tpsa',
    'TpsaError',
    'core_library',
    'ffi',
    'include_dir',
    'lib',
    '__version__',
    'TpsaMap',
    'compose',
    'evaluate',
    'exp_poisson',
    'field_to_vector',
    'inverse',
    'lie_bracket',
    'log_generator',
    'log_poisson',
    'partial_inverse',
    'translate',
    'vector_to_field',
    'pullback',
]
