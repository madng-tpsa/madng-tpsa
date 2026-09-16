"""GTPSA descriptor objects.

A descriptor defines the algebraic space for TPSA series: how many variables are
available, the maximum polynomial order, and optionally how many parameters are
part of the monomials. Every ``Tpsa`` object is created on a descriptor, and
series can only be combined meaningfully when they belong to compatible
descriptors.

The Python object is a small handle to the underlying MAD-NG GTPSA descriptor.
MAD-NG interns equivalent descriptors, and this module mirrors that by reusing the
same Python ``Descriptor`` object for the same C descriptor pointer.
"""

from __future__ import annotations

import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    NamedTuple,
    SupportsComplex,
    SupportsFloat,
    overload,
)
from weakref import WeakValueDictionary

from . import _cffi
from ._cffi import ffi, lib
from .complex_tpsa import ComplexTpsa
from .tpsa import Tpsa

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .formatting import FormatStyle


class _DescriptorAttrs(NamedTuple):
    num_vars: int
    order: int
    num_params: int
    param_order: int


class Descriptor:
    """Algebraic space definition for one or more ``Tpsa`` series.

    Parameters are appended after the variables in monomial tuples. For example,
    a descriptor with two variables and one parameter uses monomials of length
    three, where the last entry is the parameter order.
    """

    _instances_by_ptr: ClassVar[WeakValueDictionary[int, Descriptor]] = WeakValueDictionary()
    _ptr: Any
    var_labels: tuple[str, ...]
    param_labels: tuple[str, ...]
    _tpsas: WeakValueDictionary[int, Tpsa]
    _complex_tpsas: WeakValueDictionary[int, ComplexTpsa]

    __slots__ = ('_ptr', 'param_labels', 'var_labels', '_tpsas', '_complex_tpsas', '__weakref__')

    def __new__(
        cls,
        num_vars: int | None = None,
        order: int | None = None,
        *,
        num_params: int | None = None,
        param_order: int = 1,
        variables: Sequence[str] | None = None,
        params: Sequence[str] | None = None,
        max_orders: Sequence[int] | None = None,
    ) -> Descriptor:
        """Create or reuse a descriptor."""
        if order is None:
            message = 'Descriptor order is required'
            raise TypeError(message)

        var_labels = cls._resolve_labels('variable', num_vars, variables, 'v_')

        if num_params is None and params is None:
            num_params = 0

        param_labels = cls._resolve_labels('param', num_params, params, 'p_')

        num_vars = len(var_labels)
        num_params = len(param_labels)

        if order <= 0:
            message = 'Descriptor order must be positive'
            raise ValueError(message)

        if num_params > 0 and param_order <= 0:
            message = 'Descriptor parameter order must be positive'
            raise ValueError(message)

        if max_orders is not None:
            max_orders_arr = cls._resolve_max_orders(max_orders, num_vars + num_params)
            ptr = lib.mad_desc_newvpo(num_vars, order, num_params, param_order, max_orders_arr)
        elif num_params > 0:
            ptr = lib.mad_desc_newvp(num_vars, order, num_params, param_order)
        else:
            ptr = lib.mad_desc_newv(num_vars, order)

        descriptor = cls.from_ptr(ptr, var_labels=var_labels, param_labels=param_labels)

        if variables is not None and tuple(variables) != descriptor.var_labels:
            warnings.warn(
                'A descriptor of this shape already exists with different variable labels.'
                'The variables argument will be ignored, and the original labels will be reused.',
                stacklevel=2,
            )

        if params is not None and tuple(params) != descriptor.param_labels:
            warnings.warn(
                'A descriptor of this shape already exists with different parameter labels.'
                'The params argument will be ignored, and the original labels will be reused.',
                stacklevel=2,
            )

        if descriptor.order > order:
            warnings.warn(
                f'Order {order} was given, but it was overridden to {descriptor.order}.',
                stacklevel=2,
            )

        return descriptor

    @staticmethod
    def _resolve_labels(
        name: str,
        count: int | None,
        labels: Sequence[str] | None,
        prefix: str,
    ) -> tuple[str, ...]:
        if labels is None:
            if count is None:
                raise TypeError(f'Descriptor {name} count or labels are required')
            return tuple(f'{prefix}{index}' for index in range(1, int(count) + 1))

        resolved = tuple(labels)
        if count is not None and int(count) != len(resolved):
            raise ValueError(f'{name.capitalize()} labels must contain {count} entries')

        if len(set(resolved)) != len(resolved):
            raise ValueError(f'{name.capitalize()} labels must be unique')

        return resolved

    @staticmethod
    def _resolve_max_orders(
        max_orders: Sequence[int],
        monomial_length: int,
    ) -> list[int]:
        resolved = [int(max_order) for max_order in max_orders]
        if len(resolved) != monomial_length:
            message = (
                f'Length of max_orders must have the length of vars + params ({monomial_length}).'
            )
            raise ValueError(message)

        if any(max_order <= 0 for max_order in resolved):
            message = 'All max_orders must be positive'
            raise ValueError(message)
        return resolved

    @classmethod
    def from_ptr(
        cls,
        ptr: Any,
        var_labels: Sequence[str] | None = None,
        param_labels: Sequence[str] | None = None,
    ) -> Descriptor:
        """Return the interned ``Descriptor`` for a raw C pointer."""
        key = int(ffi.cast('uintptr_t', ptr))
        descriptor = cls._instances_by_ptr.get(key)

        if descriptor is not None:
            return descriptor

        if var_labels is None or param_labels is None:
            message = 'No existing Descriptor found: var_labels, param_labels must be provided.'
            raise ValueError(message)

        descriptor = super().__new__(cls)
        descriptor._ptr = ptr
        descriptor.var_labels = tuple(var_labels)
        descriptor.param_labels = tuple(param_labels)
        descriptor._tpsas = WeakValueDictionary()
        descriptor._complex_tpsas = WeakValueDictionary()
        cls._instances_by_ptr[key] = descriptor

        return descriptor

    def __del__(self) -> None:
        if getattr(self, '_ptr', None) is None:
            return

        if getattr(_cffi, 'lib', None) is not None:
            _cffi.lib.mad_desc_del(self._ptr)
        self._ptr = None

    def _default_var_labels(self) -> tuple[str, ...]:
        return tuple(f'v_{index}' for index in range(1, self.num_vars + 1))

    def _default_param_labels(self) -> tuple[str, ...]:
        return tuple(f'p_{index}' for index in range(1, self.num_params + 1))

    def __init__(
        self,
        num_vars: int | None = None,
        order: int | None = None,
        *,
        num_params: int = 0,
        param_order: int = 1,
        variables: Sequence[str] | None = None,
        params: Sequence[str] | None = None,
        max_orders: Sequence[int] | None = None,
    ) -> None:
        """Descriptor initialisation is handled in ``__new__`` for interning."""

    @property
    def ptr(self) -> Any:
        """Descriptor pointer."""
        return self._ptr

    def _get_descriptor_attrs(self) -> _DescriptorAttrs:
        """Query the attributes of the GTPSA descriptor."""
        order_ptr = ffi.new('unsigned char*')
        num_params_ptr = ffi.new('int*')
        param_order_ptr = ffi.new('unsigned char*')

        num_vars = lib.mad_desc_getnv(self._ptr, order_ptr, num_params_ptr, param_order_ptr)

        return _DescriptorAttrs(
            num_vars=num_vars,
            order=order_ptr[0],
            num_params=num_params_ptr[0],
            param_order=param_order_ptr[0],
        )

    @property
    def num_vars(self) -> int:
        """Number of variables (excluding parameters) supported by the descriptor."""
        return self._get_descriptor_attrs().num_vars

    @property
    def order(self) -> int:
        """Maximum order supported by the descriptor."""
        return self._get_descriptor_attrs().order

    @property
    def num_params(self) -> int:
        """Number of parameters supported by the descriptor."""
        return self._get_descriptor_attrs().num_params

    @property
    def param_order(self) -> int:
        """Combined parameter order cap of the descriptor."""
        return self._get_descriptor_attrs().param_order

    @property
    def monomial_length(self) -> int:
        """Length of a full monomial: ``num_vars + num_params``."""
        attrs = self._get_descriptor_attrs()
        return attrs.num_vars + attrs.num_params

    @property
    def max_orders(self) -> tuple[int, ...]:
        """Per-variable and per-parameter maximum orders."""
        max_order_arr = ffi.new('unsigned char[]', self.monomial_length)
        lib.mad_desc_maxord(self._ptr, self.monomial_length, max_order_arr)
        return tuple(max_order_arr)

    def is_valid_monomial(self, monomial: Sequence[int]) -> bool:
        """Whether ``monomial`` is representable (querying beyond-order aborts C)."""
        arr = ffi.new('unsigned char[]', monomial)
        return bool(lib.mad_desc_isvalidm(self._ptr, len(monomial), arr))

    def monomial_index(self, monomial: Iterable[int]) -> int:
        """Index of ``monomial`` in the descriptor's coefficient array."""
        m = tuple(int(x) for x in monomial)
        if not self.is_valid_monomial(m):
            raise ValueError(f'Invalid monomial {tuple(m)} for this descriptor')
        arr = ffi.new('unsigned char[]', m)
        return lib.mad_desc_idxm(self._ptr, len(m), arr)

    @overload
    def constant(self, value: SupportsComplex, /) -> ComplexTpsa: ...

    @overload
    def constant(self, value: SupportsFloat, /) -> Tpsa: ...

    def constant(self, value: SupportsFloat | SupportsComplex, /) -> Tpsa | ComplexTpsa:
        """Create a constant TPSA series on this descriptor."""
        if isinstance(value, SupportsComplex):
            t = self.complex_zero()
            t.set_const_part(value)
            return t

        t = self.zero()
        t.set_const_part(value)
        return t

    def zero(self, order: int | None = None) -> Tpsa:
        """Create a zero TPSA series on this descriptor."""
        return Tpsa(self, order=order)

    def complex_zero(self, order: int | None = None) -> ComplexTpsa:
        """Create a zero complex TPSA series on this descriptor."""
        return ComplexTpsa(self, order=order)

    @overload
    def var(
        self,
        index: int | str,
        value: SupportsFloat = 0.0,
        order: int | None = None,
    ) -> Tpsa: ...

    @overload
    def var(
        self,
        index: int | str,
        value: SupportsComplex,
        order: int | None = None,
    ) -> ComplexTpsa: ...

    def var(
        self,
        index: int | str,
        value: SupportsFloat | SupportsComplex = 0.0,
        order: int | None = None,
    ) -> Tpsa | ComplexTpsa:
        """Create identity variable ``index`` on this descriptor.

        The variable index starts from 1 and is expanded around ``value``.
        A variable label may be passed instead of an index. A complex value
        creates a ``ComplexTpsa``.
        """
        if order is not None and order <= 0:
            message = 'Variable order must be positive'
            raise ValueError(message)

        variable_index = self._var_index(index)
        if isinstance(value, SupportsComplex):
            t = self.complex_zero(order=order)
            value = complex(value)
            lib.mad_ctpsa_setvar_r(t.ptr, value.real, value.imag, variable_index, 0.0, 0.0)
            return t

        t = self.zero(order=order)
        lib.mad_tpsa_setvar(t.ptr, float(value), variable_index, 0.0)
        return t

    @overload
    def vars(self, values: Sequence[SupportsComplex]) -> tuple[ComplexTpsa, ...]: ...

    @overload
    def vars(self, values: Sequence[SupportsFloat] | None = None) -> tuple[Tpsa, ...]: ...

    @overload
    def vars(
        self, values: Sequence[SupportsFloat | SupportsComplex]
    ) -> tuple[Tpsa | ComplexTpsa, ...]: ...

    def vars(
        self, values: Sequence[SupportsFloat | SupportsComplex] | None = None
    ) -> tuple[Tpsa | ComplexTpsa, ...]:
        """Create identity series for all variables on this descriptor.

        If ``values`` is provided, each variable is expanded around the
        corresponding value. Complex values create ``ComplexTpsa`` series.
        """
        if values is None:
            values = [0.0] * self.num_vars
        if len(values) != self.num_vars:
            message = 'Values must contain one entry per variable'
            raise ValueError(message)
        return tuple(self.var(index, value) for index, value in enumerate(values, start=1))

    @overload
    def param(
        self,
        index: int | str,
        value: SupportsComplex,
    ) -> ComplexTpsa: ...

    @overload
    def param(
        self,
        index: int | str,
        value: SupportsFloat = 0.0,
    ) -> Tpsa: ...

    def param(
        self,
        index: int | str,
        value: SupportsFloat | SupportsComplex = 0.0,
    ) -> Tpsa | ComplexTpsa:
        """Create identity parameter ``index`` on this descriptor.

        The parameter index starts from 1. Parameters are appended after
        variables in monomial tuples. A parameter label may be passed instead of
        an index. A complex value creates a ``ComplexTpsa``.

        The created TPSA will automatically be of order 1; for higher orders
        instantiate a new constant TPSA with the desired order and manually
        call ``Tpsa.set`` to set the desired parameter coefficient.
        """
        order = 1
        parameter_index = self._param_index(index)
        if isinstance(value, SupportsComplex):
            t = self.complex_zero(order=order)
            value = complex(value)
            lib.mad_ctpsa_setprm_r(t.ptr, value.real, value.imag, parameter_index)
            return t

        t = self.zero(order=order)
        lib.mad_tpsa_setprm(t.ptr, float(value), parameter_index)
        return t

    def _var_index(self, variable: int | str) -> int:
        if isinstance(variable, str):
            try:
                return self.var_labels.index(variable) + 1
            except ValueError as exc:
                raise KeyError(variable) from exc
        return int(variable)

    def _param_index(self, parameter: int | str) -> int:
        if isinstance(parameter, str):
            try:
                return self.param_labels.index(parameter) + 1
            except ValueError as exc:
                raise KeyError(parameter) from exc
        return int(parameter)

    def var_or_param_index(self, variable: int | str) -> int:
        """Return the 1-based combined variable/parameter index for ``variable``."""
        if isinstance(variable, str):
            if variable in self.var_labels:
                return self.var_labels.index(variable) + 1
            if variable in self.param_labels:
                return self.num_vars + self.param_labels.index(variable) + 1
            raise KeyError(variable)

        index = int(variable)
        if index <= 0 or index > self.num_vars + self.num_params:
            message = 'Variable index out of range'
            raise ValueError(message)

        return index

    @overload
    def params(self, values: Sequence[SupportsComplex]) -> tuple[ComplexTpsa, ...]: ...

    @overload
    def params(self, values: Sequence[SupportsFloat] | None = None) -> tuple[Tpsa, ...]: ...

    @overload
    def params(
        self, values: Sequence[SupportsFloat | SupportsComplex]
    ) -> tuple[Tpsa | ComplexTpsa, ...]: ...

    def params(
        self, values: Sequence[SupportsFloat | SupportsComplex] | None = None
    ) -> tuple[Tpsa | ComplexTpsa, ...]:
        """Create identity series for all parameters on this descriptor.

        If ``values`` is provided, each parameter is expanded around the
        corresponding value. Complex values create ``ComplexTpsa`` series.
        """
        if values is None:
            values = [0.0] * self.num_params
        if len(values) != self.num_params:
            message = 'Values must contain one entry per parameter'
            raise ValueError(message)
        return tuple(self.param(index, value) for index, value in enumerate(values, start=1))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Descriptor) and self._ptr == other._ptr

    def __hash__(self) -> int:
        return int(ffi.cast('uintptr_t', self._ptr))

    def __repr__(self) -> str:
        attrs = self._get_descriptor_attrs()

        if attrs.num_params:
            return (
                f'Descriptor(num_vars={attrs.num_vars}, order={attrs.order}, '
                f'num_params={attrs.num_params}, param_order={attrs.param_order}, '
                f'var_labels={self.var_labels!r}, param_labels={self.param_labels!r})'
            )

        return (
            f'Descriptor(num_vars={attrs.num_vars}, order={attrs.order}, '
            f'var_labels={self.var_labels!r})'
        )

    def format_polynomial(self, tpsa: Tpsa, style: FormatStyle = 'code') -> object:
        """Format ``tpsa`` using this descriptor's variable and parameter labels."""
        if tpsa.descriptor is not self:
            message = 'Cannot format a TPSA from a different descriptor'
            raise ValueError(message)

        from .formatting import format_polynomial

        return format_polynomial(tpsa, self.var_labels + self.param_labels, style=style)
