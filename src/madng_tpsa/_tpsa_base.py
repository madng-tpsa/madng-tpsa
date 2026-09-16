"""Shared implementation for real and complex TPSA objects."""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import scipy.special

from ._cffi import ffi, lib
from .errors import TpsaError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from .descriptor import Descriptor
    from .tpsa import Tpsa


_Coefficient = TypeVar('_Coefficient')
_InputCoefficient = TypeVar('_InputCoefficient')


class _TpsaBase(ABC, Generic[_Coefficient, _InputCoefficient]):
    """Common functionality for wrappers around MAD-NG TPSA objects."""

    __slots__ = ()

    _UFUNC_DISPATCH = {
        np.sqrt: operator.methodcaller('sqrt'),
        np.exp: operator.methodcaller('exp'),
        np.log: operator.methodcaller('log'),
        np.sin: operator.methodcaller('sin'),
        np.cos: operator.methodcaller('cos'),
        np.tan: operator.methodcaller('tan'),
        np.sinh: operator.methodcaller('sinh'),
        np.cosh: operator.methodcaller('cosh'),
        np.tanh: operator.methodcaller('tanh'),
        np.arcsin: operator.methodcaller('asin'),
        np.arccos: operator.methodcaller('acos'),
        np.arctan: operator.methodcaller('atan'),
        np.arcsinh: operator.methodcaller('asinh'),
        np.arccosh: operator.methodcaller('acosh'),
        np.arctanh: operator.methodcaller('atanh'),
        np.negative: operator.neg,
        np.positive: operator.pos,
        np.add: operator.add,
        np.subtract: operator.sub,
        np.multiply: operator.mul,
        np.divide: operator.truediv,
        np.power: operator.pow,
        scipy.special.erf: operator.methodcaller('erf'),
        scipy.special.erfc: operator.methodcaller('erfc'),
        scipy.special.erfcx: operator.methodcaller('erfcx'),
        scipy.special.erfi: operator.methodcaller('erfi'),
        scipy.special.wofz: operator.methodcaller('wofz'),
    }

    @property
    @abstractmethod
    def ptr(self) -> Any:
        """Low-level TPSA pointer."""

    @property
    @abstractmethod
    def descriptor(self) -> Descriptor:
        """Descriptor that defines the TPSA space."""

    @property
    @abstractmethod
    def order(self) -> int:
        """Maximum order stored by this series."""

    @abstractmethod
    def get(self, monomial: Iterable[int]) -> _Coefficient:
        """Return a monomial coefficient."""

    @abstractmethod
    def set(self, monomial: Iterable[int], value: _InputCoefficient) -> None:
        """Set a monomial coefficient."""

    def __getitem__(self, monomial: Iterable[int]) -> _Coefficient:
        return self.get(monomial)

    def __setitem__(self, monomial: Iterable[int], value: _InputCoefficient) -> None:
        self.set(monomial, value)

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """Map supported NumPy ufunc calls to the matching TPSA operations."""
        if method != '__call__' or kwargs:
            return NotImplemented

        # NumPy scalar operators re-enter ufunc dispatch, so unpack them for Python dispatch.
        inputs = tuple(value.item() if isinstance(value, np.generic) else value for value in inputs)

        dispatcher = self._UFUNC_DISPATCH.get(ufunc)
        if dispatcher is None:
            return NotImplemented
        return dispatcher(*inputs)

    @abstractmethod
    def clear(self) -> None:
        """Set all coefficients to zero in place."""

    @abstractmethod
    def monomial_coeffs(self, tol: float = 1e-14) -> dict[tuple[int, ...], _Coefficient]:
        """Return stored monomial coefficients."""

    @abstractmethod
    def _check_compatible(self, other: Any) -> None:
        """Raise when ``other`` cannot be combined with this series."""

    @staticmethod
    def _raise_mad_error() -> None:
        location = ffi.string(lib.madng_tpsa_last_error_location()).decode()
        message = ffi.string(lib.madng_tpsa_last_error_message()).decode()
        if location:
            raise TpsaError(f'GTPSA error in {location}: {message}')
        raise TpsaError(f'GTPSA error: {message}')

    def coefficient(
        self,
        monomials: Sequence[int] | Sequence[Sequence[int]] | np.ndarray,
    ) -> _Coefficient | np.ndarray:
        """Return one coefficient or an array of coefficients."""
        monomial_arr = np.asarray(monomials, dtype=int)

        if monomial_arr.ndim == 1:
            return self.get(tuple(monomial_arr))

        if monomial_arr.ndim == 2:
            return np.array([self.get(tuple(row)) for row in monomial_arr])

        message = 'Monomials must be one monomial or a two-dimensional batch'
        raise ValueError(message)

    def to_dict(self, tol: float = 1e-14) -> dict[tuple[int, ...], _Coefficient]:
        """Return this series as a monomial-to-coefficient dictionary."""
        return self.monomial_coeffs(tol=tol)

    def from_dict(self, coefficients: Mapping[tuple[int, ...], _InputCoefficient]) -> None:
        """Replace coefficients from a monomial-to-coefficient dictionary."""
        self.clear()
        for monomial, coefficient in coefficients.items():
            self.set(monomial, coefficient)

    def _check_monomial(self, monomial: list[int]) -> None:
        """Raise if ``monomial`` is invalid for this descriptor or series order."""
        if not self.descriptor.is_valid_monomial(monomial):
            message = 'Monomial is not valid for this descriptor'
            raise ValueError(message)

        if sum(monomial) > self.order:
            message = f'Monomial order exceeds TPSA order {self.order}'
            raise ValueError(message)

    def _resolve_single_monomial(
        self, monomial: int | str | tuple[int, ...] | Tpsa
    ) -> tuple[int, ...]:
        """Validate and resolve a single non-constant monomial specification to an exponent tuple.

        Notes
        -----
        The input ``monomial`` (variable or parameter) can be specified in different ways.

        * If it's a tuple, the monomial specification is validated and returned directly.
        * If it's a ``str``, the monomial labelled ``monomial`` in the descriptor is returned.
        * If it's an ``int``, the monomial whose combined coordinate index is ``monomial`` is
          returned: for a TPSA with 3 variables and 2 parameters, values 1 to 3 correspond to
          the variables and 4, 5 to the parameters.
        * If it's a ``Tpsa``, it's checked that there is exactly one non-zero coefficient and
          that it's equal to one. The monomial corresponding to that coefficient is returned.

        Validation checks that the monomial does not correspond to the constant part.
        """
        from .tpsa import Tpsa

        length = self.descriptor.monomial_length

        if isinstance(monomial, Tpsa):
            self._check_compatible(monomial)
            monomial_arr = ffi.new('unsigned char[]', length)
            if lib.madng_tpsa_tpsa_single_monomial(monomial.ptr, length, monomial_arr) < 0:
                message = 'TPSA must contain exactly one non-constant monomial with coefficient 1'
                raise ValueError(message)
        elif isinstance(monomial, (int, str)):
            coordinate_index = self.descriptor.var_or_param_index(monomial)
            monomial_arr = [0] * length
            monomial_arr[coordinate_index - 1] = 1
        else:
            monomial_arr = monomial

        monomial_orders = tuple(monomial_arr)

        if len(monomial_orders) != length:
            raise ValueError(f'Monomial must have length {length}')

        if sum(monomial_orders) <= 0:
            message = 'Monomial must have a positive order'
            raise ValueError(message)

        if not self.descriptor.is_valid_monomial(monomial_orders):
            message = 'Monomial is not valid for this descriptor'
            raise ValueError(message)

        return monomial_orders

    def _integration_index(self, monomial: int | str | Tpsa) -> int:
        """Resolve an identity variable for formal integration."""
        monomial_orders = self._resolve_single_monomial(monomial)
        if sum(monomial_orders) != 1:
            message = 'Integration requires an identity variable'
            raise ValueError(message)

        variable_index = monomial_orders.index(1) + 1
        if variable_index > self.descriptor.num_vars:
            message = 'MAD-NG does not support integration with respect to parameters'
            raise NotImplementedError(message)

        return variable_index
