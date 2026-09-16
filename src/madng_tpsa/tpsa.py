"""Real truncated power series objects.

``Tpsa`` stores real coefficients in a :class:`~madng_tpsa.Descriptor`-defined
algebraic space. It provides arithmetic, calculus, and elementary functions on
truncated multivariate power series.
"""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, SupportsComplex, SupportsFloat, overload

import numpy as np
import scipy.special

from . import _cffi
from ._cffi import ffi, lib
from ._tpsa_base import _TpsaBase
from .errors import TpsaError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .complex_tpsa import ComplexTpsa
    from .descriptor import Descriptor
    from .formatting import FormatStyle


class Tpsa(_TpsaBase[float, SupportsFloat]):
    """A truncated power series in the algebraic space defined by a descriptor."""

    __slots__ = ('_descriptor', '_ptr', '__weakref__')

    def __init__(self, descriptor: Descriptor, order: int | None = None) -> None:
        """Create a zero series on ``descriptor``."""
        if order is None:
            order = lib.mad_tpsa_dflt

        self._descriptor = descriptor
        self._ptr = lib.mad_tpsa_newd(descriptor.ptr, order)
        descriptor._tpsas[int(ffi.cast('uintptr_t', self._ptr))] = self

    @classmethod
    def from_ptr(
        cls, ptr: Any, descriptor: Descriptor | None = None, *, owns: bool = False
    ) -> Tpsa:
        """Return the interned ``Tpsa`` for a raw C pointer.

        If a live ``Tpsa`` already wraps ``ptr`` it is returned directly, so
        that at most one Python object exists per C allocation, preventing a
        use-after-free if two owners existed for the same pointer.

        Parameters
        ----------
        ptr:
            Raw CFFI TPSA pointer.
        descriptor:
            Descriptor that owns this TPSA. When omitted it is inferred from
            the pointer via ``mad_tpsa_desc``. Required if ``owns`` is False.
        owns:
            Set it to ``True`` only when ``ptr`` points to a freshly allocated
            C object that has no Python wrapper yet, i.e. the call site created
            the allocation and is handing ownership. The default ``False``
            treats an unknown pointer as a bug and will raise.
        """
        key = int(ffi.cast('uintptr_t', ptr))

        if descriptor is not None:
            existing = descriptor._tpsas.get(key)
            if existing is not None:
                return existing
            if not owns:
                message = (
                    'No live Tpsa found for the given pointer. '
                    'Pass owns=True to wrap a freshly allocated pointer.'
                )
                raise ValueError(message)
        elif not owns:
            message = 'A descriptor must be provided if owns is False.'
            raise ValueError(message)
        else:
            from .descriptor import Descriptor

            descriptor = Descriptor.from_ptr(lib.mad_tpsa_desc(ptr))
            existing = descriptor._tpsas.get(key)
            if existing is not None:
                return existing

        t = cls.__new__(cls)
        t._descriptor = descriptor
        t._ptr = ptr
        descriptor._tpsas[key] = t
        return t

    def __del__(self) -> None:
        # The underlying object, or the library might have already been released,
        # if not release the handle now.
        if getattr(self, '_ptr', None) is not None and getattr(_cffi, 'lib', None) is not None:
            _cffi.lib.mad_tpsa_del(self._ptr)
            self._ptr = None

    @property
    def ptr(self) -> Any:
        """Low-level TPSA pointer for consumers marshalling across an ABI."""
        return self._ptr

    @property
    def descriptor(self) -> Descriptor:
        """Descriptor that defines this series' variables, parameters, and order."""
        return self._descriptor

    @property
    def order(self) -> int:
        """Maximum order stored by this series."""
        highest_non_zero_order = False
        return lib.mad_tpsa_ord(self._ptr, highest_non_zero_order)

    @property
    def max_nonzero_order(self) -> int:
        """Highest order currently containing a non-zero coefficient."""
        highest_non_zero_order = True
        return lib.mad_tpsa_ord(self._ptr, highest_non_zero_order)

    @property
    def const_part(self) -> float:
        """Constant coefficient of the series."""
        return lib.mad_tpsa_geti(self._ptr, 0)

    def get(self, monomial: Iterable[int]) -> float:
        """Return the coefficient for ``monomial``."""
        monomial_orders = list(monomial)
        self._check_monomial(monomial_orders)
        monomial_arr = ffi.new('unsigned char[]', monomial_orders)
        return lib.mad_tpsa_getm(self._ptr, len(monomial_orders), monomial_arr)

    def set_const_part(self, value: SupportsFloat) -> None:
        """Set the constant coefficient."""
        lib.mad_tpsa_seti(self._ptr, 0, 0.0, float(value))

    def set(self, monomial: Iterable[int], value: SupportsFloat) -> None:
        """Set the coefficient for ``monomial``."""
        monomial_orders = list(monomial)
        self._check_monomial(monomial_orders)
        monomial_arr = ffi.new('unsigned char[]', monomial_orders)
        lib.mad_tpsa_setm(self._ptr, len(monomial_orders), monomial_arr, 0.0, float(value))

    def monomial_coeffs(self, tol: SupportsFloat = 1e-14) -> dict[tuple[int, ...], float]:
        """Return stored coefficients larger than ``tol`` in absolute value.

        Keys are full monomial tuples with one entry per variable and parameter.
        """
        monomial_len = self.descriptor.monomial_length
        monomial_arr = ffi.new('unsigned char[]', monomial_len)
        coeff_ptr = ffi.new('double*')
        coeffs: dict[tuple[int, ...], float] = {}

        i = -1
        while (i := lib.mad_tpsa_cycle(self._ptr, i, monomial_len, monomial_arr, coeff_ptr)) >= 0:
            coefficient = coeff_ptr[0]
            if abs(coefficient) <= tol:
                continue
            monomial = tuple(monomial_arr)
            coeffs[monomial] = coefficient

        return coeffs

    def is_zero(self) -> bool:
        """Return whether this series has no non-zero coefficients."""
        return bool(lib.mad_tpsa_isnul(self._ptr))

    def is_constant(self) -> bool:
        """Return whether this series has no non-constant coefficients."""
        return bool(lib.mad_tpsa_isval(self._ptr))

    def clear(self) -> None:
        """Set all coefficients to zero in place."""
        lib.mad_tpsa_clear(self._ptr)

    def grad(self) -> list[float]:
        """First-order coefficients for the descriptor variables."""
        num_vars = self.descriptor.num_vars
        grad = []
        for var_idx in range(num_vars):
            monomial = [0] * num_vars
            monomial[var_idx] = 1
            grad.append(self.get(monomial))
        return grad

    def param_grad(self) -> list[float]:
        """First-order coefficients for the descriptor parameters."""
        attrs = self.descriptor._get_descriptor_attrs()
        monomial_len = attrs.num_vars + attrs.num_params
        param_grad = []
        for param_idx in range(attrs.num_params):
            monomial = [0] * monomial_len
            monomial[attrs.num_vars + param_idx] = 1
            param_grad.append(self.get(monomial))
        return param_grad

    def copy(self) -> Tpsa:
        """Return an independent copy of this series."""
        result = self.descriptor.zero()
        lib.mad_tpsa_copy(self._ptr, result._ptr)
        return result

    def integrate(self, variable: int | str | Tpsa) -> Tpsa:
        """Return the formal integral with respect to ``variable``.

        Parameters
        ----------
        variable
            May be a label, a 1-based variable/parameter index, or a TPSA identity
            variable with exactly one first-order monomial of coefficient 1. See
            :meth:`_resolve_single_monomial` for accepted inputs.
        """
        variable_index = self._integration_index(variable)
        result = self.descriptor.zero()
        lib.mad_tpsa_integ(self._ptr, result._ptr, variable_index)
        return result

    def derivative(self, variable: int | str | tuple[int, ...] | Tpsa) -> Tpsa:
        """Return a partial derivative with respect to ``variable``.

        Parameters
        ----------
        variable
            May be a label, a 1-based variable/parameter index, a monomial tuple,
            or a TPSA with exactly one non-constant coefficient equal to one. See
            :meth:`_resolve_single_monomial` for accepted inputs.
        """
        result = self.descriptor.zero()

        monomial = self._resolve_single_monomial(variable)
        monomial_arr = ffi.new('unsigned char[]', monomial)
        lib.mad_tpsa_derivm(self._ptr, result._ptr, len(monomial), monomial_arr)
        return result

    def poisson_bracket(self, other: Tpsa, num_pairs: Literal['all'] | int = 'all') -> Tpsa:
        """Return the Poisson bracket with ``other`` over canonical variable pairs."""
        self._check_compatible(other)

        if num_pairs == 'all':
            c_num_vars = 0
        else:
            if not isinstance(num_pairs, int) or isinstance(num_pairs, bool):
                message = "Parameter num_pairs must be 'all' or an integer"
                raise ValueError(message)
            c_num_vars = 2 * int(num_pairs)
            if not 0 < c_num_vars <= self.descriptor.num_vars:
                raise ValueError(
                    f'Parameter num_pairs must satisfy '
                    f'0 < 2 * num_pairs <= {self.descriptor.num_vars}',
                )

        result = self.descriptor.zero()
        lib.mad_tpsa_poisbra(self._ptr, other._ptr, result._ptr, c_num_vars)
        return result

    def __repr__(self):
        return f'Tpsa({self.to_dict()!r})'

    def format(self, style: FormatStyle = 'code') -> object:
        """Format this series as code, LaTeX math, or a rich table.

        Parameters
        ----------
        style
            Output style. See :data:`madng_tpsa.formatting.FormatStyle`.
        """
        return self.descriptor.format_polynomial(self, style=style)

    # --- arithmetic (fresh result on the same descriptor; scalars mix freely) --- #

    def _binary_op(self, other: Tpsa, fn: str) -> Tpsa:
        self._check_compatible(other)
        result = self.descriptor.zero()
        function = ffi.addressof(lib, fn)
        status = lib.madng_tpsa_protected_binary_call(
            function,
            self._ptr,
            other._ptr,
            result._ptr,
        )
        if status:
            self._raise_mad_error()
        return result

    def _coerce_operand(self, other: Tpsa | SupportsFloat) -> Tpsa:
        if isinstance(other, Tpsa):
            self._check_compatible(other)
            return other
        return self.descriptor.constant(float(other))

    def _unary_op(self, fn: str) -> Tpsa:
        result = self.descriptor.zero()
        function = ffi.addressof(lib, fn)
        status = lib.madng_tpsa_protected_unary_call(function, self._ptr, result._ptr)
        if status:
            self._raise_mad_error()
        return result

    def _two_output_unary_op(self, fn: str) -> tuple[Tpsa, Tpsa]:
        first = self.descriptor.zero()
        second = self.descriptor.zero()
        function = ffi.addressof(lib, fn)
        status = lib.madng_tpsa_protected_two_output_call(
            function,
            self._ptr,
            first._ptr,
            second._ptr,
        )
        if status:
            self._raise_mad_error()
        return first, second

    def _check_compatible(self, other: Tpsa) -> None:
        """Raise if ``other`` cannot be combined with this series."""
        if not lib.madng_tpsa_check_tpsa_compatibility(self._ptr, other._ptr):
            message = 'Incompatible TPSA descriptors'
            raise ValueError(message)

    def equals(self, other: Tpsa, tol: SupportsFloat = 0.0) -> bool:
        """Return whether this series and ``other`` have matching coefficients."""
        if not lib.madng_tpsa_check_tpsa_compatibility(self._ptr, other._ptr):
            return False
        return bool(lib.mad_tpsa_equ(self._ptr, other._ptr, float(tol)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tpsa):
            return False
        return self.equals(other)

    @overload
    def __add__(self, other: Tpsa | SupportsFloat) -> Tpsa: ...

    @overload
    def __add__(self, other: SupportsComplex) -> ComplexTpsa: ...

    def __add__(self, other: Tpsa | SupportsFloat | SupportsComplex) -> Tpsa | ComplexTpsa:
        if isinstance(other, Tpsa):
            return self._binary_op(other, 'mad_tpsa_add')

        if isinstance(other, SupportsFloat):
            result = self.descriptor.zero()
            lib.mad_tpsa_axpb(1.0, self._ptr, float(other), result._ptr)
            return result

        if isinstance(other, SupportsComplex):
            return self.descriptor.complex_zero() + self + complex(other)

        return NotImplemented

    __radd__ = __add__

    @overload
    def __sub__(self, other: Tpsa | SupportsFloat) -> Tpsa: ...

    @overload
    def __sub__(self, other: SupportsComplex) -> ComplexTpsa: ...

    def __sub__(self, other: Tpsa | SupportsFloat | SupportsComplex) -> Tpsa | ComplexTpsa:
        if isinstance(other, Tpsa):
            return self._binary_op(other, 'mad_tpsa_sub')

        if isinstance(other, SupportsFloat):
            return self.__add__(-float(other))

        if isinstance(other, SupportsComplex):
            return self.descriptor.complex_zero() + self - complex(other)

        return NotImplemented

    @overload
    def __rsub__(self, other: SupportsFloat) -> Tpsa: ...

    @overload
    def __rsub__(self, other: SupportsComplex) -> ComplexTpsa: ...

    def __rsub__(self, other: SupportsFloat | SupportsComplex) -> Tpsa | ComplexTpsa:
        if isinstance(other, SupportsFloat):
            result = self.descriptor.zero()
            lib.mad_tpsa_axpb(-1.0, self._ptr, float(other), result._ptr)
            return result

        if isinstance(other, SupportsComplex):
            return complex(other) - (self.descriptor.complex_zero() + self)

        return NotImplemented

    @overload
    def __mul__(self, other: Tpsa | SupportsFloat) -> Tpsa: ...

    @overload
    def __mul__(self, other: SupportsComplex) -> ComplexTpsa: ...

    def __mul__(self, other: Tpsa | SupportsFloat | SupportsComplex) -> Tpsa | ComplexTpsa:
        if isinstance(other, Tpsa):
            return self._binary_op(other, 'mad_tpsa_mul')

        if isinstance(other, SupportsFloat):
            result = self.descriptor.zero()
            lib.mad_tpsa_scl(self._ptr, float(other), result._ptr)
            return result

        if isinstance(other, SupportsComplex):
            return (self.descriptor.complex_zero() + self) * complex(other)

        return NotImplemented

    __rmul__ = __mul__

    @overload
    def __truediv__(self, other: Tpsa | SupportsFloat) -> Tpsa: ...

    @overload
    def __truediv__(self, other: SupportsComplex) -> ComplexTpsa: ...

    def __truediv__(self, other: Tpsa | SupportsFloat | SupportsComplex) -> Tpsa | ComplexTpsa:
        if isinstance(other, Tpsa):
            return self._binary_op(other, 'mad_tpsa_div')

        if isinstance(other, SupportsFloat):
            if other == 0:
                message = 'Division by zero scalar'
                raise ZeroDivisionError(message)
            result = self.descriptor.zero()
            lib.mad_tpsa_divn(self._ptr, float(other), result._ptr)
            return result

        if isinstance(other, SupportsComplex):
            return (self.descriptor.complex_zero() + self) / complex(other)

        return NotImplemented

    @overload
    def __rtruediv__(self, other: SupportsFloat) -> Tpsa: ...

    @overload
    def __rtruediv__(self, other: SupportsComplex) -> ComplexTpsa: ...

    def __rtruediv__(self, other: SupportsFloat | SupportsComplex) -> Tpsa | ComplexTpsa:
        if isinstance(other, SupportsFloat):
            if self.const_part == 0:
                message = 'Cannot divide by a TPSA with zero constant coefficient'
                raise ZeroDivisionError(message)
            result = self.descriptor.zero()
            lib.mad_tpsa_inv(self._ptr, float(other), result._ptr)
            return result

        if isinstance(other, SupportsComplex):
            return complex(other) / (self.descriptor.complex_zero() + self)

        return NotImplemented

    @overload
    def __pow__(self, other: Tpsa | SupportsFloat) -> Tpsa: ...

    @overload
    def __pow__(self, other: SupportsComplex) -> ComplexTpsa: ...

    def __pow__(self, other: Tpsa | SupportsFloat | SupportsComplex) -> Tpsa | ComplexTpsa:
        if isinstance(other, Tpsa):
            if not lib.madng_tpsa_check_tpsa_compatibility(self._ptr, other._ptr):
                message = 'Incompatible TPSA descriptors'
                raise ValueError(message)
            result = self.descriptor.zero()
            lib.mad_tpsa_pow(self._ptr, other._ptr, result._ptr)
            return result

        if isinstance(other, Integral):
            result = self.descriptor.zero()
            lib.mad_tpsa_powi(self._ptr, int(other), result._ptr)
            return result

        if isinstance(other, SupportsFloat):
            result = self.descriptor.zero()
            lib.mad_tpsa_pown(self._ptr, float(other), result._ptr)
            return result

        if isinstance(other, SupportsComplex):
            return (self.descriptor.complex_zero() + self) ** complex(other)

        return NotImplemented

    @overload
    def __rpow__(self, other: SupportsFloat) -> Tpsa: ...

    @overload
    def __rpow__(self, other: SupportsComplex) -> ComplexTpsa: ...

    def __rpow__(self, other: SupportsFloat | SupportsComplex) -> Tpsa | ComplexTpsa:
        if isinstance(other, SupportsFloat):
            result = self * float(np.log(float(other)))
            lib.mad_tpsa_exp(result._ptr, result._ptr)
            return result

        if isinstance(other, SupportsComplex):
            return complex(other) ** (self.descriptor.complex_zero() + self)

        return NotImplemented

    def __neg__(self) -> Tpsa:
        return self * -1.0

    def __pos__(self) -> Tpsa:
        return self.copy()

    def __abs__(self) -> Tpsa:
        return self.abs()

    def __lt__(self, other: SupportsFloat) -> bool:
        return self.const_part < float(other)

    def __le__(self, other: SupportsFloat) -> bool:
        return self.const_part <= float(other)

    def __gt__(self, other: SupportsFloat) -> bool:
        return self.const_part > float(other)

    def __ge__(self, other: SupportsFloat) -> bool:
        return self.const_part >= float(other)

    def abs(self) -> Tpsa:
        """Return the TPSA absolute value determined by the constant coefficient."""
        return self._unary_op('mad_tpsa_abs')

    def norm(self) -> float:
        """Return the sum of absolute values of the stored coefficients."""
        return lib.mad_tpsa_nrm(self._ptr)

    def unit(self) -> Tpsa:
        """Return this series divided by the magnitude of its constant coefficient."""
        if self.const_part == 0.0:
            message = 'Cannot normalise a TPSA with zero constant part'
            raise ZeroDivisionError(message)
        return self._unary_op('mad_tpsa_unit')

    def sqrt(self) -> Tpsa:
        """Return the square root of this series."""
        return self._unary_op('mad_tpsa_sqrt')

    def exp(self) -> Tpsa:
        """Return the exponential of this series."""
        return self._unary_op('mad_tpsa_exp')

    def log(self) -> Tpsa:
        """Return the natural logarithm of this series."""
        return self._unary_op('mad_tpsa_log')

    def sin(self) -> Tpsa:
        """Return the sine of this series."""
        return self._unary_op('mad_tpsa_sin')

    def cos(self) -> Tpsa:
        """Return the cosine of this series."""
        return self._unary_op('mad_tpsa_cos')

    def tan(self) -> Tpsa:
        """Return the tangent of this series."""
        return self._unary_op('mad_tpsa_tan')

    def sinc(self) -> Tpsa:
        r"""Return :math:`\sin(x) / x` for this series (the unnormalised sinc)."""
        return self._unary_op('mad_tpsa_sinc')

    def asin(self) -> Tpsa:
        """Return the inverse sine of this series."""
        return self._unary_op('mad_tpsa_asin')

    def acos(self) -> Tpsa:
        """Return the inverse cosine of this series."""
        return self._unary_op('mad_tpsa_acos')

    def atan(self) -> Tpsa:
        """Return the inverse tangent of this series."""
        return self._unary_op('mad_tpsa_atan')

    def sincos(self) -> tuple[Tpsa, Tpsa]:
        r"""Return :math:`(\sin(x), \cos(x))` for this series."""
        return self._two_output_unary_op('mad_tpsa_sincos')

    def sincosq(self) -> tuple[Tpsa, Tpsa]:
        r"""Return :math:`(\operatorname{sinc}(\sqrt{x}), \cos(\sqrt{x}))` for this series."""
        return self._two_output_unary_op('mad_tpsa_sincosq')

    def sincosmq(self) -> tuple[Tpsa, Tpsa]:
        r"""Return two related trigonometric series.

        :math:`((\operatorname{sinc}(\sqrt{x}) - 1) / x,
        (\cos(\sqrt{x}) - 1) / x)`.
        """
        return self._two_output_unary_op('mad_tpsa_sincosmq')

    def sinh(self) -> Tpsa:
        """Return the hyperbolic sine of this series."""
        return self._unary_op('mad_tpsa_sinh')

    def cosh(self) -> Tpsa:
        """Return the hyperbolic cosine of this series."""
        return self._unary_op('mad_tpsa_cosh')

    def tanh(self) -> Tpsa:
        """Return the hyperbolic tangent of this series."""
        return self._unary_op('mad_tpsa_tanh')

    def sinhc(self) -> Tpsa:
        r"""Return :math:`\sinh(x) / x` for this series (the unnormalised sinhc)."""
        return self._unary_op('mad_tpsa_sinhc')

    def asinh(self) -> Tpsa:
        """Return the inverse hyperbolic sine of this series."""
        return self._unary_op('mad_tpsa_asinh')

    def acosh(self) -> Tpsa:
        """Return the inverse hyperbolic cosine of this series."""
        return self._unary_op('mad_tpsa_acosh')

    def atanh(self) -> Tpsa:
        """Return the inverse hyperbolic tangent of this series."""
        return self._unary_op('mad_tpsa_atanh')

    def sincosh(self) -> tuple[Tpsa, Tpsa]:
        r"""Return :math:`(\sinh(x), \cosh(x))` for this series."""
        return self._two_output_unary_op('mad_tpsa_sincosh')

    def sincoshq(self) -> tuple[Tpsa, Tpsa]:
        r"""Return :math:`(\operatorname{sinhc}(\sqrt{x}), \cosh(\sqrt{x}))` for this series."""
        return self._two_output_unary_op('mad_tpsa_sincoshq')

    def sincoshmq(self) -> tuple[Tpsa, Tpsa]:
        r"""Return two related hyperbolic series.

        :math:`((\operatorname{sinhc}(\sqrt{x}) - 1) / x,
        (\cosh(\sqrt{x}) - 1) / x)`.
        """
        return self._two_output_unary_op('mad_tpsa_sincoshmq')

    def erf(self) -> Tpsa:
        """Return the error function of this series."""
        return self._unary_op('mad_tpsa_erf')

    def erfc(self) -> Tpsa:
        """Return the complementary error function of this series."""
        return self._unary_op('mad_tpsa_erfc')

    def erfcx(self) -> Tpsa:
        """Return the scaled complementary error function of this series."""
        return self._unary_op('mad_tpsa_erfcx')

    def erfi(self) -> Tpsa:
        """Return the imaginary error function of this series."""
        return self._unary_op('mad_tpsa_erfi')

    def wofz(self) -> ComplexTpsa:
        """Return the complex-valued Faddeeva function for this series."""
        from .complex_tpsa import ComplexTpsa

        return ComplexTpsa.from_tpsa(self).wofz()

    def wofz_real(self) -> Tpsa:
        """Return the real-valued Faddeeva function for this series (MAD-NG's ``wf``).

        For complex output use ``scipy.special.wofz`` of manually promote the TPSA
        to complex with (``ComplexTpsa.from_tpsa``) and then call its ``wofz``.
        """
        return self._unary_op('mad_tpsa_wf')

    def atan2(self, other: Tpsa | SupportsFloat) -> Tpsa:
        """Return ``atan2(self, other)``."""
        other = self._coerce_operand(other)
        return self._binary_op(other, 'mad_tpsa_atan2')

    def hypot(self, other: Tpsa | SupportsFloat) -> Tpsa:
        r"""Return :math:`\sqrt{\mathrm{self}^2 + \mathrm{other}^2}`."""
        other = self._coerce_operand(other)
        return self._binary_op(other, 'mad_tpsa_hypot')

    def hypot3(self, other: Tpsa | SupportsFloat, third: Tpsa | SupportsFloat) -> Tpsa:
        r"""Return :math:`\sqrt{\mathrm{self}^2 + \mathrm{other}^2 + \mathrm{third}^2}`."""
        other = self._coerce_operand(other)
        third = self._coerce_operand(third)
        if self.const_part == other.const_part == third.const_part == 0:
            message = 'At least one nonzero constant coefficient is required in hypot3'
            raise TpsaError(message)
        result = self.descriptor.zero()
        lib.mad_tpsa_hypot3(self._ptr, other._ptr, third._ptr, result._ptr)
        return result

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """Map supported NumPy ufunc calls to the matching TPSA operations."""
        if method != '__call__' or kwargs:
            return NotImplemented

        if ufunc in _UFUNC_DISPATCH:
            return _UFUNC_DISPATCH[ufunc](*inputs)
        return NotImplemented

    def __array_function__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Map supported NumPy functions to matching TPSA operations."""
        if func is np.sinc and args == (self,) and not kwargs:
            # We use mathematics convention, but NumPy prefers the DSP one (normalised sinc)
            return (np.pi * self).sinc()
        return NotImplemented


_UFUNC_DISPATCH = {
    np.absolute: Tpsa.abs,
    np.sqrt: Tpsa.sqrt,
    np.exp: Tpsa.exp,
    np.log: Tpsa.log,
    np.sin: Tpsa.sin,
    np.cos: Tpsa.cos,
    np.tan: Tpsa.tan,
    np.sinh: Tpsa.sinh,
    np.cosh: Tpsa.cosh,
    np.tanh: Tpsa.tanh,
    np.arcsin: Tpsa.asin,
    np.arccos: Tpsa.acos,
    np.arctan: Tpsa.atan,
    np.arcsinh: Tpsa.asinh,
    np.arccosh: Tpsa.acosh,
    np.arctanh: Tpsa.atanh,
    np.negative: Tpsa.__neg__,
    np.positive: Tpsa.__pos__,
    np.add: lambda left, right: left + right,
    np.subtract: lambda left, right: left - right,
    np.multiply: lambda left, right: left * right,
    np.divide: lambda left, right: left / right,
    np.power: lambda left, right: left**right,
    np.atan2: lambda left, right: (
        left.atan2(right)
        if isinstance(left, Tpsa)
        else right.descriptor.constant(float(left)).atan2(right)
    ),
    np.hypot: lambda left, right: (
        left.hypot(right) if isinstance(left, Tpsa) else right.hypot(left)
    ),
    scipy.special.erf: Tpsa.erf,
    scipy.special.erfc: Tpsa.erfc,
    scipy.special.erfcx: Tpsa.erfcx,
    scipy.special.erfi: Tpsa.erfi,
    scipy.special.wofz: Tpsa.wofz,
}
