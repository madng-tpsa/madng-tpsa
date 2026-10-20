"""Shared implementation for real and complex TPSA objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from typing import Any, Self

    from numpy.typing import DTypeLike

    from .tpsa import Tpsa

import json
import operator
import warnings
from abc import ABC, abstractmethod
from numbers import Integral

import numpy as np
import scipy.special

from ._cffi import ffi, lib
from .descriptor import Descriptor
from .functions import _raise_mad_error

_Coefficient = TypeVar('_Coefficient')
_InputCoefficient = TypeVar('_InputCoefficient')
_JsonCoefficient: TypeAlias = float | list[float]


def _match_labels(
    old_labels: Sequence[str],
    new_labels: Sequence[str],
    rename: Mapping[str, str] | None = None,
    *,
    offset: int = 0,
) -> list[int | None]:
    """Map old indices to new indices, or None if absent.

    if res = _match_labels(old, new) then old = [new[i] for i in res]
    """
    rename = dict(rename or {})
    if len(set(old_labels)) != len(old_labels):
        err_mess = 'Old labels are not unique'
        raise ValueError(err_mess)
    if len(set(new_labels)) != len(new_labels):
        err_mess = 'New labels are not unique'
        raise ValueError(err_mess)

    unknown = rename.keys() - set(old_labels)
    if unknown:
        err_mess = f'Unknown source labels: {unknown}'
        raise ValueError(err_mess)

    lookup = {name: i + offset for i, name in enumerate(new_labels)}
    for target in rename.values():
        if target not in lookup:
            err_mess = f'Unknown target label: {target}'
            raise ValueError(err_mess)

    positions = [lookup.get(rename.get(name, name)) for name in old_labels]
    present = [p for p in positions if p is not None]
    if len(set(present)) != len(present):
        err_mess = 'Multiple labels map to the same target'
        raise ValueError(err_mess)

    return positions


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

    @abstractmethod
    def __init__(self, descriptor: Descriptor, truncation_order: int | None = None) -> None:
        """Initialize the TPSA object with a descriptor and an optional truncation order."""

    def to_dict(self, tol: float = 1e-14) -> dict[str, Any]:
        """Return a dictionary representation of the TPSA series."""
        dct = {
            '__class__': self.__class__.__name__,
            'descriptor': self.descriptor.to_dict(),
            'truncation_order': self.truncation_order,
        }
        for mon, coeff in self.monomial_coeffs(tol=tol).items():
            dct[json.dumps(mon)] = self._to_json_friendly_coefficient(coeff)
        return dct

    @classmethod
    def from_dict(cls, dct: dict[str, Any], descriptor: Descriptor | None = None) -> Self:
        """Instantiate a Tpsa from a dictionary representation."""
        class_name = dct.get('__class__')
        if cls.__name__ != class_name:
            err_mess = f'Wrong dict: expected class {cls.__name__}, got {class_name}'
            raise ValueError(err_mess)
        desc = dct.get('descriptor')
        if desc is None:
            if descriptor is None:
                err_mess = 'Descriptor is missing in the dictionary. Please provide it '
                err_mess += "explicitly with the argument 'Descriptor'."
                raise ValueError(err_mess)
        else:
            desc = Descriptor.from_dict(desc)
            if descriptor is None:
                descriptor = desc
            elif desc is not descriptor:
                err_mess = 'Provided Descriptor does not agree with descriptor in dict.'
                raise ValueError(err_mess)

        truncation_order = dct.get('truncation_order')
        if truncation_order is None:
            err_mess = 'Truncation order is missing in the dictionary'
            raise ValueError(err_mess)
        instance = cls(descriptor=descriptor, truncation_order=truncation_order)
        instance.clear()
        coeff_dict: dict[tuple[int, ...], _InputCoefficient] = {}
        for monomial_json, coefficient in dct.items():
            if monomial_json in ('__class__', 'descriptor', 'truncation_order'):
                continue
            try:
                monomial = json.loads(monomial_json)
            except json.JSONDecodeError as e:
                err_mess = f'Unknown key: {monomial_json}'
                raise ValueError(err_mess) from e
            if not descriptor.is_valid_monomial(monomial):
                err_mess = f'Invalid monomial: {monomial}'
                raise ValueError(err_mess)
            monomial = tuple(monomial)
            dim = descriptor.num_vars + descriptor.num_params
            if not all(tt >= 0 for tt in monomial) or len(monomial) != dim:
                err_mess = f'Invalid monomial: {monomial}'
                raise ValueError(err_mess)
            coeff_dict[monomial] = cls._from_json_friendly_coefficient(coefficient)
        instance.update(coeff_dict)
        return instance

    def to_descriptor(
        self,
        descriptor: Descriptor,
        *,
        variable_map: Mapping[str, str] | None = None,
        parameter_map: Mapping[str, str] | None = None,
        truncate: bool = False,
    ) -> Self:
        """Convert this TPSA to a different descriptor.

        If a monomial exceeds a destination order limit, it is discarded
        if truncate=True; otherwise an error is raised. If a variable or
        parameter is absent from the destination descriptor and the
        polynomial depends on it, conversion raises a ValueError, even
        when truncate=True.
        """
        old = self.descriptor
        if descriptor is old and not variable_map and not parameter_map:
            return self.copy()

        positions = [
            *_match_labels(old.variables, descriptor.variables, variable_map),
            *_match_labels(
                old.parameters, descriptor.parameters, parameter_map, offset=descriptor.num_vars
            ),
        ]

        target_order = min(self.truncation_order, descriptor.order)
        result = type(self)(descriptor, truncation_order=target_order)

        for monomial, coefficient in self.monomial_coeffs(tol=0).items():
            exponents = [0] * descriptor.monomial_length
            for power, target in zip(monomial, positions, strict=True):
                if target is None:
                    if power:
                        err_mess = 'Cannot remove a variable or parameter on which the '
                        err_mess += 'TPSA depends'
                        raise ValueError(err_mess)
                else:
                    exponents[target] = power

            # Check allocated order and all descriptor restrictions.
            if not descriptor.is_valid_monomial(exponents):
                if truncate:
                    continue
                err_mess = f'New descriptor cannot represent {tuple(exponents)}'
                raise ValueError(err_mess)
            result.set(exponents, cast('_InputCoefficient', coefficient))

        return result

    @property
    @abstractmethod
    def ptr(self) -> Any:
        """Low-level TPSA pointer."""

    @property
    @abstractmethod
    def descriptor(self) -> Descriptor:
        """Descriptor that defines the TPSA space."""

    @property
    def order(self) -> int:
        """Backwards-compatible property for truncation_order (deprecated)."""
        warnings.warn(
            'The order property is deprecated, use truncation_order instead.',
            FutureWarning,
            stacklevel=2,
        )
        return self.truncation_order

    @property
    @abstractmethod
    def truncation_order(self) -> int:
        """Maximum order stored by this series."""

    @property
    def max_nonzero_order(self) -> int:
        """Backwards-compatible property for degree (deprecated)."""
        warnings.warn(
            'The max_nonzero_order property is deprecated, use degree instead.',
            FutureWarning,
            stacklevel=2,
        )
        return self.degree

    @property
    @abstractmethod
    def degree(self) -> int:
        """Highest order currently containing a non-zero coefficient."""

    @property
    @abstractmethod
    def const_part(self) -> _Coefficient:
        """Return the constant coefficient."""

    @const_part.setter
    @abstractmethod
    def const_part(self, value: _InputCoefficient) -> None:
        """Return the constant coefficient."""

    @abstractmethod
    def set_const_part(self, value: _InputCoefficient) -> None:
        """Set the constant coefficient."""

    @abstractmethod
    def homogeneous(self, degree: int) -> Self:
        """Return the homogeneous part of the requested degree."""

    @abstractmethod
    def truncate(self, degree: int) -> Self:
        """Return a truncated copy through the requested degree."""

    @abstractmethod
    def clear_order(self, degree: int) -> Self:
        """Return a copy without coefficients of the requested degree."""

    @abstractmethod
    def get(self, monomial: Iterable[int]) -> _Coefficient:
        """Return a monomial coefficient."""

    @abstractmethod
    def set(self, monomial: Iterable[int], value: _InputCoefficient) -> None:
        """Set a monomial coefficient."""

    @abstractmethod
    def copy(self) -> Self:
        """Return an independent copy of this TPSA."""

    @staticmethod
    @abstractmethod
    def _to_json_friendly_coefficient(value: _Coefficient) -> _JsonCoefficient:
        """Convert a coefficient to a JSON-friendly format."""

    @staticmethod
    @abstractmethod
    def _from_json_friendly_coefficient(value: _JsonCoefficient) -> _InputCoefficient:
        """Convert a JSON-friendly coefficient back to its internal format."""

    def __getitem__(self, monomial: Iterable[int]) -> _Coefficient:
        return self.get(monomial)

    def __setitem__(self, monomial: Iterable[int], value: _InputCoefficient) -> None:
        self.set(monomial, value)

    def __len__(self) -> int:
        return len(self.monomial_coeffs())

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

    @abstractmethod
    def grad(self) -> list[_Coefficient]:
        """Return first-order coefficients for the descriptor variables."""

    @abstractmethod
    def param_grad(self) -> list[_Coefficient]:
        """Return first-order coefficients for descriptor parameters."""

    @staticmethod
    def _raise_mad_error() -> None:
        # For backwards compatibility
        return _raise_mad_error()

    @staticmethod
    def _validate_degree(degree: object, max_order: int) -> int:
        if not isinstance(degree, Integral) or isinstance(degree, bool):
            message = 'degree must be an integer'
            raise TypeError(message)
        degree = int(degree)
        if not 0 <= degree <= max_order:
            message = f'degree must satisfy 0 <= degree <= {max_order}'
            raise ValueError(message)
        return degree

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

    def update(self, coeff_dict: dict[tuple[int, ...], _InputCoefficient]) -> None:
        """Update the TPSA series from a dictionary of monomial coefficients."""
        for monomial, coeff in coeff_dict.items():
            self.set(monomial, coeff)

    def _check_monomial(self, monomial: list[int]) -> None:
        """Raise if ``monomial`` is invalid for this descriptor or series order."""
        if len(monomial) != self.descriptor.monomial_length:
            raise ValueError(f'Monomial must have length {self.descriptor.monomial_length}')
        if sum(monomial) > self.truncation_order:
            message = f'Monomial degree exceeds TPSA order {self.truncation_order}'
            raise ValueError(message)
        if not self.descriptor.is_valid_monomial(monomial):
            message = 'Monomial is not valid for this descriptor'
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
            message = 'Monomial must have a positive degree'
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

    def _compatible(self, other: _TpsaBase) -> bool:
        return self.descriptor is other.descriptor and bool(
            lib.madng_tpsa_check_tpsa_compatibility(self.ptr, other.ptr)
        )

    def _check_compatible(self, other: _TpsaBase) -> None:
        if not self._compatible(other):
            err_mess = 'TPSAs have incompatible descriptors or coordinate labels'
            raise ValueError(err_mess)

    def __array__(
        self,
        dtype: DTypeLike | None = None,
        copy: bool | None = None,  # noqa: FBT001
    ) -> np.ndarray:
        """Represent this TPSA as a scalar object in a NumPy array."""
        if dtype is not None and np.dtype(dtype) != np.dtype(object):
            err_mess = 'TPSAs require an object-dtype NumPy array'
            raise TypeError(err_mess)
        if copy is False:
            err_mess = 'Cannot provide a zero-copy NumPy representation'
            raise ValueError(err_mess)
        result = np.empty((), dtype=object)
        result[()] = self
        return result
