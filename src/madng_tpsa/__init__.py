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
from .formatting import format_polynomial
from .map import TpsaMap, symplectic_unit_matrix
from .paths import core_library, include_dir
from .tensor import TpsaTensor
from .tpsa import Tpsa
from .vector import TpsaVector, TpsaVectorField

__all__ = [
    'CDEF',
    'Descriptor',
    'TpsaError',
    'Tpsa',
    'ComplexTpsa',
    'TpsaTensor',
    'TpsaVector',
    'TpsaVectorField',
    'TpsaMap',
    'core_library',
    'ffi',
    'format_polynomial',
    'include_dir',
    'lib',
    'symplectic_unit_matrix',
    '__version__',
]
