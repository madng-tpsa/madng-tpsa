"""GTPSA descriptor objects.

A descriptor defines the algebraic space for TPSA series: its variables,
parameters, and their polynomial-order limits. Every ``Tpsa`` object is created
on a descriptor, and series can only be combined meaningfully when they belong
to compatible descriptors.

The Python object is a small handle to the underlying MAD-NG GTPSA descriptor.
MAD-NG interns equivalent native algebras. This module gives each distinct
Python label set its own Descriptor view of the shared native pointer, and
keeps the native allocation alive until the last labelled view is released.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, SupportsComplex, overload

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Any, ClassVar, SupportsFloat

    from .complex_tpsa import ComplexTpsa
    from .formatting import FormatStyle
    from .tpsa import Tpsa

import warnings
from operator import index
from weakref import WeakValueDictionary

import numpy as np

from . import _cffi
from ._cffi import ffi, lib


class _DescriptorAttrs(NamedTuple):
    num_vars: int
    order: int
    num_params: int
    param_order: int


class _NativeDescriptor:
    """Single Python owner of a shared MAD-NG descriptor allocation."""

    _tpsas: WeakValueDictionary[int, Tpsa]
    _complex_tpsas: WeakValueDictionary[int, ComplexTpsa]
    __slots__ = (
        'ptr',
        'address',
        'instances',
        'tpsas',
        'complex_tpsas',
        '__weakref__',
    )

    def __init__(self, ptr: Any) -> None:
        self.ptr = ptr
        self.address = int(ffi.cast('uintptr_t', ptr))
        self.instances: WeakValueDictionary[
            tuple[type, tuple[str, ...], tuple[str, ...]], Descriptor
        ] = WeakValueDictionary()
        self.tpsas: WeakValueDictionary[int, Tpsa] = WeakValueDictionary()
        self.complex_tpsas: WeakValueDictionary[int, ComplexTpsa] = WeakValueDictionary()

    def __del__(self) -> None:
        ptr = getattr(self, 'ptr', None)
        if ptr is None:
            return
        self.ptr = None
        native_lib = getattr(_cffi, 'lib', None)
        if native_lib is not None:
            native_lib.mad_desc_del(ptr)


class Descriptor:
    """Algebraic space definition for one or more ``Tpsa`` series.

    Parameters are appended after the variables in monomial tuples. For example,
    a descriptor with two variables and one parameter uses monomials of length
    three, where the last entry is the parameter order.
    """

    _owners_by_ptr: ClassVar[WeakValueDictionary[int, _NativeDescriptor]] = WeakValueDictionary()
    _owner: _NativeDescriptor
    _ptr: Any
    variables: tuple[str, ...]
    parameters: tuple[str, ...]
    _tpsas: WeakValueDictionary[int, Tpsa]
    _complex_tpsas: WeakValueDictionary[int, ComplexTpsa]
    __slots__ = (
        '_owner',
        '_ptr',
        'variables',
        'parameters',
        '_tpsas',
        '_complex_tpsas',
        '__weakref__',
    )

    def __new__(
        cls,
        num_vars: int | None = None,
        order: int | None = None,
        *,
        num_params: int | None = None,
        param_order: int = 1,
        variables: Sequence[str] | None = None,
        parameters: Sequence[str] | None = None,
        max_orders: Sequence[int] | None = None,
        params: Sequence[str] | None = None,  # Only for backwards compatibility
    ) -> Descriptor:
        """Create or reuse a descriptor: equivalent descriptors are interned.

        Parameters
        ----------
        num_vars
            Number of variables. Omit it when ``variables`` provides their labels.
        order
            Maximum combined order of variable and parameter monomials.
        num_params
            Number of parameters. Omit it when ``parameters`` provides their labels;
            defaults to zero when ``parameters`` is omitted.
        param_order
            Maximum combined order of parameter monomials.
        variables
            Variable labels. Their number must match ``num_vars`` when both are given.
        parameters
            Parameter labels. Their number must match ``num_params`` when both are given.
        max_orders
            Per-variable and per-parameter order caps. It must contain one positive
            entry per variable and parameter.
        """
        if order is None:
            message = 'Descriptor order is required'
            raise TypeError(message)

        if parameters is None and params is not None:
            warnings.warn(
                'The params argument is deprecated and will be removed in a future version. '
                'Use parameters instead.',
                FutureWarning,
                stacklevel=2,
            )
            parameters = params

        var_labels = cls._resolve_labels('variable', num_vars, variables, 'v_')

        if num_params is None and parameters is None:
            num_params = 0

        param_labels = cls._resolve_labels('param', num_params, parameters, 'p_')

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

        descriptor = cls.from_ptr(ptr, variables=var_labels, parameters=param_labels)
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
            message = 'Length of max_orders must have the length of vars + parameters '
            message += f'({monomial_length}).'
            raise ValueError(message)

        if any(max_order <= 0 for max_order in resolved):
            message = 'All max_orders must be positive'
            raise ValueError(message)
        return resolved

    @classmethod
    def from_ptr(
        cls,
        ptr: Any,
        variables: Sequence[str] | None = None,
        parameters: Sequence[str] | None = None,
    ) -> Descriptor:
        """Intern by both native pointer AND Python labels.

        An unlabelled lookup is allowed only with a unique live labelled view.
        A newly encountered native pointer is assumed Python-owned, matching
        the previous from_ptr() contract. Borrowed pointers need an explicit
        ownership policy before they may be passed here.
        """
        if ptr == ffi.NULL:
            err_mess = 'Cannot wrap a null descriptor pointer'
            raise ValueError(err_mess)
        address = int(ffi.cast('uintptr_t', ptr))
        owner = Descriptor._owners_by_ptr.get(address)
        if variables is None and parameters is None:
            live = list(owner.instances.values()) if owner is not None else []
            if len(live) == 1:
                return live[0]
            if len(live) > 1:
                err_mess = 'Ambiguous descriptor pointer: multiple labelled views exist. '
                err_mess += 'Supply variables= and parameters= explicitly.'
                raise ValueError(err_mess)
            err_mess = 'Unknown descriptor pointer: supply variables= and parameters= explicitly.'
            raise ValueError(err_mess)
        if variables is None or parameters is None:
            err_mess = 'Supply both variables and parameters, or neither'
            raise ValueError(err_mess)
        var_labels = tuple(variables)
        param_labels = tuple(parameters)
        if len(set(var_labels)) != len(var_labels):
            err_mess = 'Variable labels must be unique'
            raise ValueError(err_mess)
        if len(set(param_labels)) != len(param_labels):
            err_mess = 'Parameter labels must be unique'
            raise ValueError(err_mess)
        if set(var_labels) & set(param_labels):
            err_mess = 'A label cannot be both variable and parameter'
            raise ValueError(err_mess)
        # Reject metadata that disagrees with the actual native descriptor.
        native_order = ffi.new('unsigned char*')
        native_np = ffi.new('int*')
        native_po = ffi.new('unsigned char*')
        native_nv = lib.mad_desc_getnv(ptr, native_order, native_np, native_po)
        if len(var_labels) != native_nv or len(param_labels) != native_np[0]:
            err_mess = f'Native descriptor requires {native_nv} variables and '
            err_mess += f'{native_np[0]} parameters, got '
            err_mess += f'{len(var_labels)} and {len(param_labels)} labels'
            raise ValueError(err_mess)
        if owner is None:
            owner = _NativeDescriptor(ptr)
            Descriptor._owners_by_ptr[address] = owner
        key = (cls, var_labels, param_labels)
        existing = owner.instances.get(key)
        if existing is not None:
            return existing
        descriptor = object.__new__(cls)
        descriptor._owner = owner
        descriptor._ptr = owner.ptr
        descriptor.variables = var_labels
        descriptor.parameters = param_labels
        descriptor._tpsas = owner.tpsas
        descriptor._complex_tpsas = owner.complex_tpsas
        owner.instances[key] = descriptor
        return descriptor

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the descriptor."""
        attr = self._get_descriptor_attrs()
        if len(self.variables) != attr.num_vars:
            raise ValueError(f'Expected {attr.num_vars} variables, got {len(self.variables)}')
        if len(self.parameters) != attr.num_params:
            raise ValueError(f'Expected {attr.num_params} parameters, got {len(self.parameters)}')
        # Only specify max_order if it is non-trivial
        max_orders = tuple(int(v) for v in self.max_orders)
        defaults = (self.order,) * self.num_vars + (self.param_order,) * self.num_params
        return {
            '__class__': self.__class__.__name__,
            'variables': self.variables,
            'parameters': self.parameters,
            'order': attr.order,
            'param_order': attr.param_order,
            'max_orders': None if max_orders == defaults else max_orders,
        }

    @classmethod
    def from_dict(cls, dct) -> Descriptor:
        """Instantiate a Descriptor from a dictionary representation."""
        class_name = dct.get('__class__')
        if cls.__name__ != class_name:
            err_mess = f'Wrong dict: expected class {cls.__name__}, got {class_name}'
            raise ValueError(err_mess)
        for kk in ('variables', 'parameters', 'order', 'param_order', 'max_orders'):
            if kk not in dct:
                raise ValueError(f'Missing required key: {kk}')
        return cls(
            variables=dct['variables'],
            parameters=dct['parameters'],
            order=dct['order'],
            param_order=dct['param_order'],
            max_orders=dct['max_orders'],
        )

    def union(self, other: Descriptor) -> Descriptor:
        """Create a common descriptor containing both variable sets."""
        if not isinstance(other, Descriptor):
            err_mess = 'Expected another Descriptor'
            raise TypeError(err_mess)
        if self is other:
            return self

        variables = tuple(dict.fromkeys((*self.variables, *other.variables)))
        parameters = tuple(dict.fromkeys((*self.parameters, *other.parameters)))
        overlap = set(variables) & set(parameters)
        if overlap:
            err_mess = f'Labels cannot be both variables and parameters: {overlap}'
            raise ValueError(err_mess)
        order = max(self.order, other.order)
        param_order = max(self.param_order, other.param_order)
        # Merge the limits for labels actually present in each descriptor.
        limits: dict[str, int] = {}
        for desc in (self, other):
            labels = (*desc.variables, *desc.parameters)
            for name, limit in zip(labels, desc.max_orders, strict=True):
                limits[name] = max(limits.get(name, 0), int(limit))

        max_orders = tuple(limits[name] for name in (*variables, *parameters))
        defaults = (order,) * len(variables) + (param_order,) * len(parameters)

        return type(self)(
            variables=variables,
            parameters=parameters,
            order=order,
            param_order=param_order,
            max_orders=None if max_orders == defaults else max_orders,
        )

    def _default_var_labels(self) -> tuple[str, ...]:
        return tuple(f'v_{index}' for index in range(1, self.num_vars + 1))

    def _default_param_labels(self) -> tuple[str, ...]:
        return tuple(f'p_{index}' for index in range(1, self.num_params + 1))

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
    def var_labels(self) -> tuple[str, ...]:
        """Backwards-compatible property for variable labels (deprecated)."""
        warnings.warn(
            'The var_labels property is deprecated and will be removed in a future version. '
            'Use the variables property instead.',
            FutureWarning,
            stacklevel=2,
        )
        return self.variables

    @property
    def param_labels(self) -> tuple[str, ...]:
        """Backwards-compatible property for parameter labels (deprecated)."""
        warnings.warn(
            'The param_labels property is deprecated and will be removed in a future version. '
            'Use the parameters property instead.',
            FutureWarning,
            stacklevel=2,
        )
        return self.parameters

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
        """Length of a full monomial, equal to ``num_vars`` + ``num_params``."""
        attrs = self._get_descriptor_attrs()
        return attrs.num_vars + attrs.num_params

    @property
    def max_orders(self) -> tuple[int, ...]:
        """Per-variable and per-parameter maximum orders."""
        max_order_arr = ffi.new('unsigned char[]', self.monomial_length)
        lib.mad_desc_maxord(self._ptr, self.monomial_length, max_order_arr)
        return tuple(max_order_arr)

    def is_valid_monomial(self, monomial: Sequence[int]) -> bool:
        """Whether ``monomial`` is representable (querying beyond order is illegal in the C API)."""
        try:
            powers = tuple(index(v) for v in monomial)
        except TypeError:
            return False
        # Some manual checks first, to avoid false positives due to C casting
        if len(powers) != self.monomial_length:
            return False
        if any(isinstance(v, (bool, np.bool_)) for v in monomial):
            return False
        if any(not 0 <= p <= 255 for p in powers):
            return False
        arr = ffi.new('unsigned char[]', powers)
        return bool(lib.mad_desc_isvalidm(self._ptr, len(powers), arr))

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
            t.const_part = value
            return t

        t = self.zero()
        t.const_part = value
        return t

    def zero(self, order: int | None = None) -> Tpsa:
        """Create a zero TPSA series on this descriptor."""
        from .tpsa import Tpsa

        return Tpsa(self, truncation_order=order)

    def complex_zero(self, order: int | None = None) -> ComplexTpsa:
        """Create a zero complex TPSA series on this descriptor."""
        from .complex_tpsa import ComplexTpsa

        return ComplexTpsa(self, truncation_order=order)

    def from_monomial_coeffs(self, coeffs, *, order=None):
        """Construct a TPSA from {monomial: coefficient}."""
        is_complex = any(np.iscomplexobj(c) for c in coeffs.values())
        result = self.complex_zero(order=order) if is_complex else self.zero(order=order)
        for monomial, coefficient in coeffs.items():
            result.set(monomial, coefficient)
        return result

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
        call :meth:`madng_tpsa.Tpsa.set` to set the desired parameter coefficient.
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
                return self.variables.index(variable) + 1
            except ValueError as exc:
                raise KeyError(variable) from exc
        return int(variable)

    def _param_index(self, parameter: int | str) -> int:
        if isinstance(parameter, str):
            try:
                return self.parameters.index(parameter) + 1
            except ValueError as exc:
                raise KeyError(parameter) from exc
        return int(parameter)

    def var_or_param_index(self, variable: int | str) -> int:
        """Return the 1-based combined variable/parameter index for ``variable``."""
        if isinstance(variable, str):
            if variable in self.variables:
                return self.variables.index(variable) + 1
            if variable in self.parameters:
                return self.num_vars + self.parameters.index(variable) + 1
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
        if not isinstance(other, Descriptor):
            err_mess = 'Expected another Descriptor'
            raise TypeError(err_mess)
        return (
            type(self) is type(other)
            and self._owner is other._owner
            and self.variables == other.variables
            and self.parameters == other.parameters
        )

    def __hash__(self) -> int:
        return hash(
            (
                type(self),
                self._owner.address,
                self.variables,
                self.parameters,
            )
        )

    def __repr__(self) -> str:
        attrs = self._get_descriptor_attrs()

        if attrs.num_params:
            return (
                f'Descriptor(num_vars={attrs.num_vars}, order={attrs.order}, '
                f'num_params={attrs.num_params}, param_order={attrs.param_order}, '
                f'variables={self.variables!r}, parameters={self.parameters!r})'
            )

        return (
            f'Descriptor(num_vars={attrs.num_vars}, order={attrs.order}, '
            f'variables={self.variables!r})'
        )

    def format_polynomial(self, tpsa: Tpsa, style: FormatStyle = 'code') -> object:
        """Format ``tpsa`` using this descriptor's variable and parameter labels.

        Parameters
        ----------
        style
            Output style. See :data:`madng_tpsa.formatting.FormatStyle`.
        """
        if tpsa.descriptor is not self:
            message = 'Cannot format a TPSA from a different descriptor'
            raise ValueError(message)

        from .formatting import format_polynomial

        return format_polynomial(tpsa, self.variables + self.parameters, style=style)
