"""Complex truncated power series objects.

``ComplexTpsa`` stores complex coefficients in a
:class:`~madng_tpsa.Descriptor`-defined algebraic space. It mirrors the real
TPSA API where operations are defined for complex series.
"""

from __future__ import annotations

import operator
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, SupportsComplex, SupportsFloat

import numpy as np
import scipy.special

from . import _cffi
from ._cffi import ffi, lib
from ._tpsa_base import _TpsaBase
from .tpsa import Tpsa

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .descriptor import Descriptor


class ComplexTpsa(_TpsaBase[complex, SupportsFloat | SupportsComplex]):
    """A complex truncated power series on a :class:`Descriptor`."""

    __slots__ = ('_descriptor', '_ptr', '__weakref__')

    def __init__(self, descriptor: Descriptor, order: int | None = None) -> None:
        """Create a zero complex series on ``descriptor``."""
        if order is None:
            order = lib.mad_tpsa_dflt

        self._descriptor = descriptor
        self._ptr = lib.mad_ctpsa_newd(descriptor.ptr, order)
        descriptor._complex_tpsas[int(ffi.cast('uintptr_t', self._ptr))] = self

    @classmethod
    def from_ptr(
        cls, ptr: Any, descriptor: Descriptor | None = None, *, owns: bool = False
    ) -> ComplexTpsa:
        """Return the interned ``ComplexTpsa`` for a raw C pointer.

        If a live ``ComplexTpsa`` already wraps ``ptr`` it is returned directly,
        so that at most one Python object exists per C allocation, preventing a
        use-after-free if two owners existed for the same pointer.

        Parameters
        ----------
        ptr:
            Raw CFFI complex TPSA pointer.
        descriptor:
            Descriptor that owns this TPSA. When omitted it is inferred from
            the pointer via ``mad_ctpsa_desc``. Required if ``owns`` is False.
        owns:
            Set it to ``True`` only when ``ptr`` points to a freshly allocated
            C object that has no Python wrapper yet, i.e. the call site created
            the allocation and is handing ownership. The default ``False``
            treats an unknown pointer as a bug and will raise.
        """
        key = int(ffi.cast('uintptr_t', ptr))

        if descriptor is not None:
            existing = descriptor._complex_tpsas.get(key)
            if existing is not None:
                return existing
            if not owns:
                message = (
                    'No live ComplexTpsa found for the given pointer. '
                    'Pass owns=True to wrap a freshly allocated pointer.'
                )
                raise ValueError(message)
        elif not owns:
            message = 'A descriptor must be provided if owns is False.'
            raise ValueError(message)
        else:
            from .descriptor import Descriptor

            descriptor = Descriptor.from_ptr(lib.mad_ctpsa_desc(ptr))
            existing = descriptor._complex_tpsas.get(key)
            if existing is not None:
                return existing

        t = cls.__new__(cls)
        t._descriptor = descriptor
        t._ptr = ptr
        descriptor._complex_tpsas[key] = t
        return t

    @classmethod
    def from_tpsa(cls, real: Tpsa, imag: Tpsa | None = None) -> ComplexTpsa:
        """Promote real and optional imaginary TPSAs to a complex TPSA."""
        if imag is None:
            imag = real.descriptor.zero(order=real.order)
        else:
            real._check_compatible(imag)

        result = cls(real.descriptor, order=real.order)
        lib.mad_ctpsa_cplx(real.ptr, imag.ptr, result.ptr)
        return result

    def __del__(self) -> None:
        if getattr(self, '_ptr', None) is not None and getattr(_cffi, 'lib', None) is not None:
            _cffi.lib.mad_ctpsa_del(self._ptr)
            self._ptr = None

    @property
    def ptr(self) -> Any:
        """Low-level CTPSA pointer for consumers marshalling across an ABI."""
        return self._ptr

    @property
    def descriptor(self) -> Descriptor:
        """Descriptor that defines this series' algebraic space."""
        return self._descriptor

    @property
    def order(self) -> int:
        """Maximum order stored by this series."""
        highest_non_zero_order = False
        return lib.mad_ctpsa_ord(self._ptr, highest_non_zero_order)

    @property
    def max_nonzero_order(self) -> int:
        """Highest order currently containing a non-zero coefficient."""
        highest_non_zero_order = True
        return lib.mad_ctpsa_ord(self._ptr, highest_non_zero_order)

    @property
    def const_part(self) -> complex:
        """Constant coefficient of the series."""
        return self._geti(0)

    def _geti(self, index: int) -> complex:
        value = ffi.new('double _Complex*')
        lib.mad_ctpsa_geti_r(self._ptr, index, value)
        return complex(value[0])

    def get(self, monomial: Iterable[int]) -> complex:
        """Return the coefficient for ``monomial``."""
        monomial_orders = list(monomial)
        self._check_monomial(monomial_orders)
        monomial_arr = ffi.new('unsigned char[]', monomial_orders)
        value = ffi.new('double _Complex*')
        lib.mad_ctpsa_getm_r(self._ptr, len(monomial_orders), monomial_arr, value)
        return complex(value[0])

    def set_const_part(self, value: SupportsFloat | SupportsComplex) -> None:
        """Set the constant coefficient."""
        self._seti(0, complex(value))

    def _seti(self, index: int, value: complex) -> None:
        lib.mad_ctpsa_seti_r(self._ptr, index, 0.0, 0.0, value.real, value.imag)

    def set(self, monomial: Iterable[int], value: SupportsFloat | SupportsComplex) -> None:
        """Set the coefficient for ``monomial``."""
        monomial_orders = list(monomial)
        self._check_monomial(monomial_orders)
        monomial_arr = ffi.new('unsigned char[]', monomial_orders)
        coefficient = complex(value)
        lib.mad_ctpsa_setm_r(
            self._ptr,
            len(monomial_orders),
            monomial_arr,
            0.0,
            0.0,
            coefficient.real,
            coefficient.imag,
        )

    def monomial_coeffs(self, tol: float = 1e-14) -> dict[tuple[int, ...], complex]:
        """Return stored coefficients with magnitude greater than ``tol``."""
        monomial_len = self.descriptor.monomial_length
        monomial_arr = ffi.new('unsigned char[]', monomial_len)
        coefficient = ffi.new('double _Complex*')
        coefficients: dict[tuple[int, ...], complex] = {}

        index = -1
        while (
            index := lib.mad_ctpsa_cycle(self._ptr, index, monomial_len, monomial_arr, coefficient)
        ) >= 0:
            value = complex(coefficient[0])
            if abs(value) > tol:
                coefficients[tuple(monomial_arr)] = value
        return coefficients

    def is_zero(self) -> bool:
        """Return whether this series has no non-zero coefficients."""
        return bool(lib.mad_ctpsa_isnul(self._ptr))

    def is_constant(self) -> bool:
        """Return whether this series has no non-constant coefficients."""
        return bool(lib.mad_ctpsa_isval(self._ptr))

    def clear(self) -> None:
        """Set all coefficients to zero in place."""
        lib.mad_ctpsa_clear(self._ptr)

    def grad(self) -> list[complex]:
        """First-order coefficients for the descriptor variables."""
        return [
            self.get([int(index == variable) for index in range(self.descriptor.num_vars)])
            for variable in range(self.descriptor.num_vars)
        ]

    def param_grad(self) -> list[complex]:
        """First-order coefficients for the descriptor parameters."""
        length = self.descriptor.monomial_length
        return [
            self.get(
                [int(index == self.descriptor.num_vars + parameter) for index in range(length)]
            )
            for parameter in range(self.descriptor.num_params)
        ]

    def copy(self) -> ComplexTpsa:
        """Return an independent copy of this series."""
        result = self.descriptor.complex_zero()
        lib.mad_ctpsa_copy(self._ptr, result._ptr)
        return result

    def real(self) -> Tpsa:
        """Return the real part as a real TPSA."""
        result = self.descriptor.zero()
        lib.mad_ctpsa_real(self._ptr, result._ptr)
        return result

    def imag(self) -> Tpsa:
        """Return the imaginary part as a real TPSA."""
        result = self.descriptor.zero()
        lib.mad_ctpsa_imag(self._ptr, result._ptr)
        return result

    def integrate(self, variable: int | str | Tpsa) -> ComplexTpsa:
        """Return the formal integral with respect to ``variable``.

        Parameters
        ----------
        variable
            May be a label, a 1-based variable/parameter index, or a TPSA identity
            variable with exactly one first-order monomial of coefficient 1. See
            :meth:`_resolve_single_monomial` for accepted inputs.
        """
        result = self.descriptor.complex_zero()
        lib.mad_ctpsa_integ(self._ptr, result._ptr, self._integration_index(variable))
        return result

    def derivative(self, variable: int | str | tuple[int, ...] | Tpsa) -> ComplexTpsa:
        """Return a partial derivative with respect to ``variable``.

        Parameters
        ----------
        variable
            May be a label, a 1-based variable/parameter index, a monomial tuple,
            or a TPSA with exactly one non-constant coefficient equal to one. See
            :meth:`_resolve_single_monomial` for accepted inputs.
        """
        result = self.descriptor.complex_zero()
        monomial = self._resolve_single_monomial(variable)
        monomial_arr = ffi.new('unsigned char[]', monomial)
        lib.mad_ctpsa_derivm(self._ptr, result._ptr, len(monomial), monomial_arr)
        return result

    def poisson_bracket(
        self,
        other: ComplexTpsa,
        num_pairs: Literal['all'] | int = 'all',
    ) -> ComplexTpsa:
        """Return the Poisson bracket with ``other`` over canonical variable pairs."""
        self._check_compatible(other)
        if num_pairs == 'all':
            c_num_vars = 0
        elif isinstance(num_pairs, int) and not isinstance(num_pairs, bool):
            c_num_vars = 2 * num_pairs
            if not 0 < c_num_vars <= self.descriptor.num_vars:
                raise ValueError(
                    f'Parameter num_pairs must satisfy 0 < 2 * num_pairs <= '
                    f'{self.descriptor.num_vars}'
                )
        else:
            message = "Parameter num_pairs must be 'all' or an integer"
            raise ValueError(message)
        result = self.descriptor.complex_zero()
        lib.mad_ctpsa_poisbra(self._ptr, other._ptr, result._ptr, c_num_vars)
        return result

    def __repr__(self) -> str:
        return f'ComplexTpsa({self.to_dict()!r})'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ComplexTpsa) and self.equals(other)

    def equals(self, other: ComplexTpsa, tol: float = 0.0) -> bool:
        """Return whether this and ``other`` have matching coefficients."""
        return bool(self._compatible(other) and lib.mad_ctpsa_equ(self._ptr, other._ptr, tol))

    def _binary_op(self, other: ComplexTpsa | Tpsa, function_name: str) -> ComplexTpsa:
        self._check_compatible(other)
        result = self.descriptor.complex_zero()
        function = ffi.addressof(lib, function_name)
        status = lib.madng_tpsa_protected_binary_call(function, self._ptr, other._ptr, result._ptr)
        if status:
            self._raise_mad_error()
        return result

    def _unary_op(self, function_name: str) -> ComplexTpsa:
        result = self.descriptor.complex_zero()
        function = ffi.addressof(lib, function_name)
        status = lib.madng_tpsa_protected_unary_call(function, self._ptr, result._ptr)
        if status:
            self._raise_mad_error()
        return result

    def __add__(self, other: ComplexTpsa | Tpsa | SupportsFloat | SupportsComplex) -> ComplexTpsa:
        if isinstance(other, ComplexTpsa):
            return self._binary_op(other, 'mad_ctpsa_add')

        if isinstance(other, Tpsa):
            return self._binary_op(other, 'mad_ctpsa_addt')

        if isinstance(other, (SupportsFloat, SupportsComplex)):
            return self._axpb(1.0, complex(other))

        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other: ComplexTpsa | Tpsa | SupportsFloat | SupportsComplex) -> ComplexTpsa:
        if isinstance(other, ComplexTpsa):
            return self._binary_op(other, 'mad_ctpsa_sub')

        if isinstance(other, Tpsa):
            return self._binary_op(other, 'mad_ctpsa_subt')

        if isinstance(other, (SupportsFloat, SupportsComplex)):
            return self._axpb(1.0, -complex(other))

        return NotImplemented

    def __rsub__(self, other: Tpsa | SupportsFloat | SupportsComplex) -> ComplexTpsa:
        if isinstance(other, Tpsa):
            self._check_compatible(other)
            result = self.descriptor.complex_zero()
            self._protected_binary(other, self, result, 'mad_ctpsa_tsub')
            return result

        if isinstance(other, (SupportsFloat, SupportsComplex)):
            return self._axpb(-1.0, complex(other))

        return NotImplemented

    def __mul__(self, other: ComplexTpsa | Tpsa | SupportsFloat | SupportsComplex) -> ComplexTpsa:
        if isinstance(other, ComplexTpsa):
            return self._binary_op(other, 'mad_ctpsa_mul')

        if isinstance(other, Tpsa):
            return self._binary_op(other, 'mad_ctpsa_mult')

        if isinstance(other, (SupportsFloat, SupportsComplex)):
            return self._scale(complex(other))

        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(
        self, other: ComplexTpsa | Tpsa | SupportsFloat | SupportsComplex
    ) -> ComplexTpsa:
        if isinstance(other, ComplexTpsa):
            return self._binary_op(other, 'mad_ctpsa_div')

        if isinstance(other, Tpsa):
            return self._binary_op(other, 'mad_ctpsa_divt')

        if isinstance(other, (SupportsFloat, SupportsComplex)):
            value = complex(other)
            if value == 0:
                message = 'Division by zero scalar'
                raise ZeroDivisionError(message)
            result = self.descriptor.complex_zero()
            lib.mad_ctpsa_divn_r(self._ptr, value.real, value.imag, result._ptr)
            return result

        return NotImplemented

    def __rtruediv__(self, other: Tpsa | SupportsFloat | SupportsComplex) -> ComplexTpsa:
        if isinstance(other, Tpsa):
            self._check_compatible(other)
            result = self.descriptor.complex_zero()
            self._protected_binary(other, self, result, 'mad_ctpsa_tdiv')
            return result

        if isinstance(other, (SupportsFloat, SupportsComplex)):
            if self.const_part == 0:
                message = 'Cannot divide by a CTPSA with zero constant part'
                raise ZeroDivisionError(message)
            value = complex(other)
            result = self.descriptor.complex_zero()
            lib.mad_ctpsa_inv_r(self._ptr, value.real, value.imag, result._ptr)
            return result

        return NotImplemented

    def __pow__(self, other: ComplexTpsa | Tpsa | SupportsFloat | SupportsComplex) -> ComplexTpsa:
        if isinstance(other, ComplexTpsa):
            return self._binary_op(other, 'mad_ctpsa_pow')

        if isinstance(other, Tpsa):
            return self._binary_op(other, 'mad_ctpsa_powt')

        if isinstance(other, Integral):
            result = self.descriptor.complex_zero()
            lib.mad_ctpsa_powi(self._ptr, int(other), result._ptr)
            return result

        if isinstance(other, (SupportsFloat, SupportsComplex)):
            value = complex(other)
            result = self.descriptor.complex_zero()
            lib.mad_ctpsa_pown_r(self._ptr, value.real, value.imag, result._ptr)
            return result

        return NotImplemented

    def __rpow__(self, other: Tpsa | SupportsFloat | SupportsComplex) -> ComplexTpsa:
        if isinstance(other, Tpsa):
            self._check_compatible(other)
            result = self.descriptor.complex_zero()
            self._protected_binary(other, self, result, 'mad_ctpsa_tpow')
            return result

        if isinstance(other, (SupportsFloat, SupportsComplex)):
            return (self * np.log(complex(other))).exp()

        return NotImplemented

    def __neg__(self) -> ComplexTpsa:
        return self._scale(-1.0)

    def __pos__(self) -> ComplexTpsa:
        return self.copy()

    def _axpb(self, scale: complex, offset: complex) -> ComplexTpsa:
        result = self.descriptor.complex_zero()
        lib.mad_ctpsa_axpb_r(
            scale.real, scale.imag, self._ptr, offset.real, offset.imag, result._ptr
        )
        return result

    def _scale(self, value: complex) -> ComplexTpsa:
        result = self.descriptor.complex_zero()
        lib.mad_ctpsa_scl_r(self._ptr, value.real, value.imag, result._ptr)
        return result

    def conjugate(self) -> ComplexTpsa:
        """Return the complex conjugate of this series."""
        return self._unary_op('mad_ctpsa_conj')

    def norm(self) -> float:
        """Return the sum of magnitudes of the stored coefficients."""
        return lib.mad_ctpsa_nrm(self._ptr)

    def unit(self) -> ComplexTpsa:
        """Return this series normalised by the magnitude of its constant part."""
        if self.const_part == 0:
            message = 'Cannot normalise a CTPSA with zero constant part'
            raise ZeroDivisionError(message)
        return self._unary_op('mad_ctpsa_unit')

    def sqrt(self) -> ComplexTpsa:
        """Return the square root of this series."""
        return self._unary_op('mad_ctpsa_sqrt')

    def exp(self) -> ComplexTpsa:
        """Return the exponential of this series."""
        return self._unary_op('mad_ctpsa_exp')

    def log(self) -> ComplexTpsa:
        """Return the natural logarithm of this series."""
        return self._unary_op('mad_ctpsa_log')

    def sin(self) -> ComplexTpsa:
        """Return the sine of this series."""
        return self._unary_op('mad_ctpsa_sin')

    def cos(self) -> ComplexTpsa:
        """Return the cosine of this series."""
        return self._unary_op('mad_ctpsa_cos')

    def tan(self) -> ComplexTpsa:
        """Return the tangent of this series."""
        return self._unary_op('mad_ctpsa_tan')

    def sinh(self) -> ComplexTpsa:
        """Return the hyperbolic sine of this series."""
        return self._unary_op('mad_ctpsa_sinh')

    def cosh(self) -> ComplexTpsa:
        """Return the hyperbolic cosine of this series."""
        return self._unary_op('mad_ctpsa_cosh')

    def tanh(self) -> ComplexTpsa:
        """Return the hyperbolic tangent of this series."""
        return self._unary_op('mad_ctpsa_tanh')

    def asin(self) -> ComplexTpsa:
        """Return the inverse sine of this series."""
        return self._unary_op('mad_ctpsa_asin')

    def acos(self) -> ComplexTpsa:
        """Return the inverse cosine of this series."""
        return self._unary_op('mad_ctpsa_acos')

    def atan(self) -> ComplexTpsa:
        """Return the inverse tangent of this series."""
        return self._unary_op('mad_ctpsa_atan')

    def asinh(self) -> ComplexTpsa:
        """Return the inverse hyperbolic sine of this series."""
        return self._unary_op('mad_ctpsa_asinh')

    def acosh(self) -> ComplexTpsa:
        """Return the inverse hyperbolic cosine of this series."""
        return self._unary_op('mad_ctpsa_acosh')

    def atanh(self) -> ComplexTpsa:
        """Return the inverse hyperbolic tangent of this series."""
        return self._unary_op('mad_ctpsa_atanh')

    def erf(self) -> ComplexTpsa:
        """Return the error function of this series."""
        return self._unary_op('mad_ctpsa_erf')

    def erfc(self) -> ComplexTpsa:
        """Return the complementary error function of this series."""
        return self._unary_op('mad_ctpsa_erfc')

    def erfcx(self) -> ComplexTpsa:
        """Return the scaled complementary error function of this series."""
        return self._unary_op('mad_ctpsa_erfcx')

    def erfi(self) -> ComplexTpsa:
        """Return the imaginary error function of this series."""
        return self._unary_op('mad_ctpsa_erfi')

    def wofz(self) -> ComplexTpsa:
        """Return the Faddeeva function of this series."""
        return self._unary_op('mad_ctpsa_wf')

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """Map supported NumPy ufunc calls to matching CTPSA operations."""
        if method != '__call__' or kwargs:
            return NotImplemented

        # We need to "unpack" the numpy scalars to let Python normal operator fallback mechanism
        # function properly: otherwise, if lhs is np.float64, we get into an infinite recursion
        inputs = tuple(value.item() if isinstance(value, np.generic) else value for value in inputs)

        if ufunc in _UFUNC_DISPATCH:
            return _UFUNC_DISPATCH[ufunc](*inputs)
        return NotImplemented

    def _protected_binary(
        self,
        left: ComplexTpsa | Tpsa,
        right: ComplexTpsa | Tpsa,
        result: ComplexTpsa,
        function_name: str,
    ) -> None:
        function = ffi.addressof(lib, function_name)
        status = lib.madng_tpsa_protected_binary_call(function, left._ptr, right._ptr, result._ptr)
        if status:
            self._raise_mad_error()

    def _compatible(self, other: ComplexTpsa | Tpsa) -> bool:
        return bool(lib.madng_tpsa_check_tpsa_compatibility(self._ptr, other._ptr))

    def _check_compatible(self, other: ComplexTpsa | Tpsa) -> None:
        if not self._compatible(other):
            message = 'Incompatible TPSA descriptors'
            raise ValueError(message)


_UFUNC_DISPATCH = {
    np.sqrt: ComplexTpsa.sqrt,
    np.exp: ComplexTpsa.exp,
    np.log: ComplexTpsa.log,
    np.sin: ComplexTpsa.sin,
    np.cos: ComplexTpsa.cos,
    np.tan: ComplexTpsa.tan,
    np.sinh: ComplexTpsa.sinh,
    np.cosh: ComplexTpsa.cosh,
    np.tanh: ComplexTpsa.tanh,
    np.arcsin: ComplexTpsa.asin,
    np.arccos: ComplexTpsa.acos,
    np.arctan: ComplexTpsa.atan,
    np.arcsinh: ComplexTpsa.asinh,
    np.arccosh: ComplexTpsa.acosh,
    np.arctanh: ComplexTpsa.atanh,
    np.conjugate: ComplexTpsa.conjugate,
    np.negative: operator.neg,
    np.positive: operator.pos,
    np.add: operator.add,
    np.subtract: operator.sub,
    np.multiply: operator.mul,
    np.divide: operator.truediv,
    np.power: operator.pow,
    scipy.special.erf: ComplexTpsa.erf,
    scipy.special.erfc: ComplexTpsa.erfc,
    scipy.special.erfcx: ComplexTpsa.erfcx,
    scipy.special.erfi: ComplexTpsa.erfi,
    scipy.special.wofz: ComplexTpsa.wofz,
}
