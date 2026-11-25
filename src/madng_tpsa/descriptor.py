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

from typing import TYPE_CHECKING, NamedTuple, SupportsComplex, cast, overload

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from typing import Any, ClassVar, SupportsFloat

    from .complex_tpsa import ComplexTpsa
    from .tpsa import Tpsa

import warnings
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
    # Resolved lazily to avoid importing TPSA classes during Descriptor module import.
    _real_tpsa_type: ClassVar[type[Tpsa] | None] = None
    _complex_tpsa_type: ClassVar[type[ComplexTpsa] | None] = None
    _owner: _NativeDescriptor
    _ptr: Any
    var_names: tuple[str, ...]
    param_names: tuple[str, ...]
    _tpsas: WeakValueDictionary[int, Tpsa]
    _complex_tpsas: WeakValueDictionary[int, ComplexTpsa]
    __slots__ = (
        '_owner',
        '_ptr',
        'var_names',
        'param_names',
        '_tpsas',
        '_complex_tpsas',
        '__weakref__',
    )

    def __new__(
        cls,
        var_names: int | Sequence[str],
        order: int,
        *,
        param_names: int | Sequence[str] | None = None,
        param_order: int = 1,
        max_orders: Sequence[int] | None = None,
    ) -> Descriptor:
        """Create or reuse a descriptor: equivalent descriptors are interned.

        Parameters
        ----------
        var_names
            Variable names. Alternatively, an integer specifying the number of
            variables can be provided (in which case the variables are named
            automatically as v_0, v_1, v_2, ...).
        order
            Maximum combined order of variable and parameter monomials.
        param_names
            Parameter names. Alternatively, an integer specifying the number of
            parameters can be provided (in which case the parameters are named
            automatically as p_0, p_1, p_2, ...).
        param_order
            Maximum combined order of parameter monomials.
        max_orders
            Per-variable and per-parameter order caps. It must contain one positive
            entry per variable and parameter.
        """
        if isinstance(var_names, int):
            var_names = tuple(f'v_{index}' for index in range(1, var_names + 1))
        _num_vars = len(var_names)
        if param_names is None:
            param_names = ()
        elif isinstance(param_names, int):
            param_names = tuple(f'p_{index}' for index in range(1, param_names + 1))
        _num_params = len(param_names)

        if order <= 0:
            message = 'Descriptor order must be positive'
            raise ValueError(message)
        if _num_params > 0 and param_order <= 0:
            message = 'Descriptor parameter order must be positive'
            raise ValueError(message)

        if max_orders is not None:
            _num_tot = _num_vars + _num_params
            max_orders_arr = [int(max_order) for max_order in max_orders]
            if len(max_orders_arr) != _num_tot:
                message = (
                    'Length of max_orders must have the length of var_names + param_names '
                    f'({_num_tot}).'
                )
                raise ValueError(message)
            if any(max_order <= 0 for max_order in max_orders_arr):
                message = 'All max_orders must be positive'
                raise ValueError(message)
            ptr = lib.mad_desc_newvpo(_num_vars, order, _num_params, param_order, max_orders_arr)

        elif _num_params > 0:
            ptr = lib.mad_desc_newvp(_num_vars, order, _num_params, param_order)

        else:
            ptr = lib.mad_desc_newv(_num_vars, order)

        descriptor = cls.from_ptr(ptr, var_names=var_names, param_names=param_names)
        if descriptor.order > order:
            warnings.warn(
                f'Order {order} was given, but it was overridden to {descriptor.order}.',
                stacklevel=2,
            )

        return descriptor

    @classmethod
    def from_ptr(
        cls,
        ptr: Any,
        var_names: Sequence[str] | None = None,
        param_names: Sequence[str] | None = None,
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
        if var_names is None and param_names is None:
            live = list(owner.instances.values()) if owner is not None else []
            if len(live) == 1:
                return live[0]
            if len(live) > 1:
                err_mess = (
                    'Ambiguous descriptor pointer: multiple labelled views exist. '
                    'Supply var_names= and param_names= explicitly.'
                )
                raise ValueError(err_mess)
            err_mess = 'Unknown descriptor pointer: supply var_names= and param_names= explicitly.'
            raise ValueError(err_mess)
        if var_names is None or param_names is None:
            err_mess = 'Supply both var_names and param_names, or neither'
            raise ValueError(err_mess)
        var_labels = tuple(var_names)
        param_labels = tuple(param_names)
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
            err_mess = (
                f'Native descriptor requires {native_nv} var_names and '
                f'{native_np[0]} param_names, got '
                f'{len(var_labels)} and {len(param_labels)} labels'
            )
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
        descriptor.var_names = var_labels
        descriptor.param_names = param_labels
        descriptor._tpsas = owner.tpsas
        descriptor._complex_tpsas = owner.complex_tpsas
        owner.instances[key] = descriptor
        return descriptor

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the descriptor."""
        attr = self._get_descriptor_attrs()
        if len(self.var_names) != attr.num_vars:
            raise ValueError(f'Expected {attr.num_vars} variables, got {len(self.var_names)}')
        if len(self.param_names) != attr.num_params:
            raise ValueError(f'Expected {attr.num_params} parameters, got {len(self.param_names)}')
        # Only specify max_order if it is non-trivial
        max_orders = tuple(int(v) for v in self.max_orders)
        defaults = (self.order,) * self.num_vars + (self.param_order,) * self.num_params
        return {
            '__class__': self.__class__.__name__,
            'var_names': self.var_names,
            'param_names': self.param_names,
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
        for kk in ('var_names', 'param_names', 'order', 'param_order', 'max_orders'):
            if kk not in dct:
                raise ValueError(f'Missing required key: {kk}')
        return cls(
            var_names=dct['var_names'],
            param_names=dct['param_names'],
            order=dct['order'],
            param_order=dct['param_order'],
            max_orders=dct['max_orders'],
        )

    def union(self, other: Descriptor, *, restrict_order: bool = False) -> Descriptor:
        """Create a common descriptor containing both variable sets."""
        if not isinstance(other, Descriptor):
            err_mess = 'Expected another Descriptor'
            raise TypeError(err_mess)
        if self is other:
            return self

        var_names = tuple(dict.fromkeys((*self.var_names, *other.var_names)))
        param_names = tuple(dict.fromkeys((*self.param_names, *other.param_names)))
        overlap = set(var_names) & set(param_names)
        if overlap:
            err_mess = f'Labels cannot be both variables and parameters: {overlap}'
            raise ValueError(err_mess)
        param_order = max(self.param_order, other.param_order)
        if restrict_order:
            order = min(self.order, other.order)
            if self.param_order > 0 and other.param_order > 0:
                param_order = min(self.param_order, other.param_order)
        else:
            order = max(self.order, other.order)
        # Merge the limits for labels actually present in each descriptor.
        limits: dict[str, int] = {}
        for desc in (self, other):
            labels = (*desc.var_names, *desc.param_names)
            for name, limit in zip(labels, desc.max_orders, strict=True):
                limits[name] = max(limits.get(name, 0), int(limit))

        max_orders = tuple(limits[name] for name in (*var_names, *param_names))
        defaults = (order,) * len(var_names) + (param_order,) * len(param_names)

        return type(self)(
            var_names=var_names,
            param_names=param_names,
            order=order,
            param_order=param_order,
            max_orders=None if max_orders == defaults else max_orders,
        )

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
        # from_ptr() verifies that labels agree with the native descriptor.
        return len(self.var_names)

    @property
    def num_params(self) -> int:
        """Number of parameters supported by the descriptor."""
        return len(self.param_names)

    @property
    def order(self) -> int:
        """Maximum order supported by the descriptor."""
        return self._get_descriptor_attrs().order

    @property
    def param_order(self) -> int:
        """Combined parameter order cap of the descriptor."""
        return self._get_descriptor_attrs().param_order

    @property
    def monomial_length(self) -> int:
        """Length of a full monomial, equal to ``num_vars`` + ``num_params``."""
        return len(self.var_names) + len(self.param_names)

    @property
    def max_orders(self) -> tuple[int, ...]:
        """Per-variable and per-parameter maximum orders."""
        max_order_arr = ffi.new('unsigned char[]', self.monomial_length)
        lib.mad_desc_maxord(self._ptr, self.monomial_length, max_order_arr)
        return tuple(max_order_arr)

    def is_valid_monomial(self, monomial: Sequence[object] | np.ndarray) -> bool:
        """Whether ``monomial`` is representable (querying beyond order is illegal in the C API)."""
        try:
            if len(monomial) != self.monomial_length:
                return False
        except TypeError:
            return False
        if any(isinstance(v, (bool, np.bool_)) for v in monomial):
            return False
        try:
            # CFFI performs integer conversion and unsigned-byte range checks.
            # tuple() also accepts array/range inputs (ffi.new does not).
            arr = ffi.new('unsigned char[]', tuple(monomial))
        except (TypeError, ValueError, OverflowError):
            return False
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
            t.const_part = value
            return t

        t = self.zero()
        t.const_part = value
        return t

    def zero(self, order: int | None = None) -> Tpsa:
        """Create a zero TPSA series on this descriptor."""
        tpsa_type = Descriptor._real_tpsa_type
        if tpsa_type is None:
            from .tpsa import Tpsa

            tpsa_type = Tpsa
            Descriptor._real_tpsa_type = tpsa_type
        return tpsa_type(self, truncation_order=order)

    def complex_zero(self, order: int | None = None) -> ComplexTpsa:
        """Create a zero complex TPSA series on this descriptor."""
        tpsa_type = Descriptor._complex_tpsa_type
        if tpsa_type is None:
            from .complex_tpsa import ComplexTpsa

            tpsa_type = ComplexTpsa
            Descriptor._complex_tpsa_type = tpsa_type
        return tpsa_type(self, truncation_order=order)

    def polynomial(
        self,
        coeffs: Mapping[tuple[int, ...], SupportsFloat | SupportsComplex],
        *,
        order: int | None = None,
        needs_complex: bool = False,
    ) -> Tpsa | ComplexTpsa:
        """Create a scalar TPSA from monomial coefficients on this descriptor.

        Complex-valued coefficients select the complex TPSA representation.
        """
        if needs_complex or any(
            isinstance(value, (complex, np.complexfloating)) for value in coeffs.values()
        ):
            complex_result = self.complex_zero(order=order)
            for monomial, coefficient in coeffs.items():
                complex_result.set(monomial, coefficient)
            return complex_result
        result = self.zero(order=order)
        for monomial, coefficient in coeffs.items():
            result.set(monomial, cast('SupportsFloat', coefficient))
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
                return self.var_names.index(variable) + 1
            except ValueError as exc:
                raise KeyError(variable) from exc
        return int(variable)

    def _param_index(self, parameter: int | str) -> int:
        if isinstance(parameter, str):
            try:
                return self.param_names.index(parameter) + 1
            except ValueError as exc:
                raise KeyError(parameter) from exc
        return int(parameter)

    def var_or_param_index(self, name_or_index: int | str) -> int:
        """Return the 1-based combined variable/parameter index for ``name_or_index``."""
        if isinstance(name_or_index, str):
            if name_or_index in self.var_names:
                return self.var_names.index(name_or_index) + 1
            if name_or_index in self.param_names:
                return self.num_vars + self.param_names.index(name_or_index) + 1
            raise KeyError(name_or_index)

        index = int(name_or_index)
        if index <= 0 or index > self.num_vars + self.num_params:
            message = 'Variable or parameter index out of range'
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
            return NotImplemented
        return (
            type(self) is type(other)
            and self._owner is other._owner
            and self.var_names == other.var_names
            and self.param_names == other.param_names
        )

    def __hash__(self) -> int:
        return hash(
            (
                type(self),
                self._owner.address,
                self.var_names,
                self.param_names,
            )
        )

    def __repr__(self) -> str:
        attrs = self._get_descriptor_attrs()

        if attrs.num_params:
            return (
                f'Descriptor(var_names={self.var_names!r}, order={attrs.order}, '
                f'param_names={self.param_names!r}, param_order={attrs.param_order})'
            )

        return f'Descriptor(var_names={self.var_names!r}, order={attrs.order})'
