"""Wrapper for multidimensional collections of TPSAs.

The native MAD-NG library represents any map (of any rank) as an array of TPSA
pointers rather than as a separate C object. This module provides a small
Python container around such an array, and adds map operations as methods to it.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from numbers import Integral
from typing import TYPE_CHECKING, Generic, SupportsComplex, SupportsFloat, cast

from ._typing import TpsaSeriesT

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from typing import Any, Self

    from ._typing import DerivativeSpec, MapSeriesT, Nested, Scalar, TpsaSeries
    from .maps import TpsaMap

from math import prod
from operator import index
from warnings import warn

import numpy as np

from ._cffi import ffi, lib
from .complex_tpsa import ComplexTpsa
from .descriptor import Descriptor
from .tpsa import Tpsa


def _flatten_nested(values: Nested[TpsaSeries]) -> tuple[list[TpsaSeries], tuple[int, ...]]:
    """Flatten nested TPSAs and infer a rectangular shape."""
    if isinstance(values, (Tpsa, ComplexTpsa)):
        return [values], ()
    if isinstance(values, np.ndarray) and values.ndim == 0:
        return _flatten_nested(values.item())

    if not isinstance(values, (Sequence, np.ndarray)) or isinstance(
        values, (str, bytes, bytearray)
    ):
        err_mess = 'Expected a TPSA or a sequence of TPSAs'
        raise TypeError(err_mess)
    if len(values) == 0:
        err_mess = 'Empty dimensions are not supported'
        raise ValueError(err_mess)

    flat: list[TpsaSeries] = []
    child_shape: tuple[int, ...] | None = None

    for i, item in enumerate(values):
        components, shape = _flatten_nested(item)
        if child_shape is None:
            child_shape = shape
        elif shape != child_shape:
            err_mess = f'Ragged tensor at index {i}: expected {child_shape}, got {shape}'
            raise ValueError(err_mess)
        flat.extend(components)

    assert child_shape is not None
    return flat, (len(values), *child_shape)


class TpsaTensor(Generic[TpsaSeriesT]):
    """Multidimensional collection of TPSAs in row-major order.

    Partial indexing returns a lightweight view sharing the
    underlying TPSA objects.
    """

    __slots__ = (
        '_flat',
        '_shape',
        '_strides',
        '_offset',
        '_descriptor',
        '_pointer_array',
        '_is_complex',
    )
    _flat: list[TpsaSeriesT]
    _shape: tuple[int, ...]
    _strides: tuple[int, ...]
    _offset: int
    _descriptor: Descriptor
    _pointer_array: ffi.CData
    _is_complex: bool

    def __init__(self, values: Nested[TpsaSeriesT], *, shape: Sequence[int] | None = None) -> None:
        """Initialize a TPSA tensor as a container for series of TPSAs.

        Parameters
        ----------
        values : Nested[TpsaSeries]
            A (potentially nested) list of Tpsa's and ComplexTpsa's.
        shape : Sequence[int], default None
            Intended shape for the TpsaTensor. If omitted, it is inferred
            from 'values'.

        Examples
        --------
        The following are valid initialisations:
            TpsaTensor([a, b, c, d])
            TpsaTensor([a, b, c, d], shape=(2, 2))
            TpsaTensor([[a, b], [c, d]])
            TpsaTensor(np.array([[a, b], [c, d]], dtype=object))
            TpsaTensor([[[a, b], [c, d]]])
        """
        # Infer the shape from the shape of values, if present
        data, inferred_shape = _flatten_nested(values)
        dims = inferred_shape if shape is None else tuple(shape)

        if not dims or any(
            isinstance(n, (bool, np.bool_)) or not isinstance(n, int) or n < 1 for n in dims
        ):
            err_mess = "Attribute 'shape' must contain positive integer dimensions"
            raise ValueError(err_mess)
        if shape is not None and inferred_shape not in (dims, (prod(dims),)):
            err_mess = f'Input shape {inferred_shape} incompatible with {dims}'
            raise ValueError(err_mess)

        if len(data) != prod(dims):
            err_mess = f'Shape {dims} requires {prod(dims)} TPSAs; got {len(data)}'
            raise ValueError(err_mess)
        if any(not isinstance(v, (Tpsa, ComplexTpsa)) for v in data):
            err_mess = 'All entries must be TPSAs'
            raise TypeError(err_mess)

        descriptor = data[0].descriptor
        if any(v.descriptor is not descriptor for v in data[1:]):
            err_mess = 'All entries must share the same Descriptor'
            raise ValueError(err_mess)

        strides = [1] * len(dims)
        for axis in range(len(dims) - 2, -1, -1):
            strides[axis] = strides[axis + 1] * dims[axis + 1]

        if any(isinstance(tpsa, ComplexTpsa) for tpsa in data):
            data = [
                tpsa if isinstance(tpsa, ComplexTpsa) else ComplexTpsa.from_tpsa(tpsa)
                for tpsa in data
            ]
            self._is_complex = True
        else:
            self._is_complex = False

        # The constructor promotes mixed inputs to one homogeneous TPSA type.
        self._flat = cast('list[TpsaSeriesT]', data)
        self._shape = dims
        self._strides = tuple(strides)
        self._offset = 0
        self._descriptor = descriptor
        self._pointer_array = ffi.new('void*[]', [tpsa.ptr for tpsa in data])

    def _new_like(self, values: Sequence[TpsaSeriesT]) -> Self:
        """Create a new tensor with the same structure."""
        return type(self)(values, shape=self.shape)

    @classmethod
    def _from_dict_components(cls, values: Sequence[TpsaSeries], dct: dict[str, Any]) -> Self:
        """Create a new instance from a dict.

        This should probably be overwritten when inherited.
        """
        return cls(values, shape=tuple(dct['shape']))

    def to_dict(self, tol: float = 1e-14) -> dict[str, Any]:
        """Return a dictionary representation of a TpsaTensor."""
        dct = {
            '__class__': self.__class__.__name__,
            'descriptor': self.descriptor.to_dict(),
            'shape': self.shape,
        }
        flat = []
        # Only store the TPSAs from the partial view
        for tpsa in self.flat:
            tpsa_dct = tpsa.to_dict(tol=tol)
            tpsa_dct.pop('descriptor', None)  # Duplicate
            flat.append(tpsa_dct)
        dct['flat'] = flat
        return dct

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> Self:
        """Instantiate a TpsaTensor from a dictionary representation."""
        class_name = dct.get('__class__')
        if cls.__name__ != class_name:
            err_mess = f'Wrong dict: expected class {cls.__name__}, got {class_name}'
            raise ValueError(err_mess)
        desc_dct = dct.get('descriptor')
        if desc_dct is None:
            err_mess = 'Descriptor is missing in the dictionary'
            raise ValueError(err_mess)
        descriptor = Descriptor.from_dict(desc_dct)
        shape = dct.get('shape')
        if shape is None:
            err_mess = 'Shape is missing in the dictionary'
            raise ValueError(err_mess)
        shape = tuple(shape)
        flat = dct.get('flat')
        if flat is None:
            err_mess = 'Flat list is missing in the dictionary'
            raise ValueError(err_mess)
        tpsa_flat = []
        for tpsa_dct in flat:
            this_cls = tpsa_dct.get('__class__', None)
            if this_cls is None:
                err_mess = "Missing '__class__' key in TPSA dict."
                raise ValueError(err_mess)
            if this_cls == 'ComplexTpsa':
                tpsa_flat.append(ComplexTpsa.from_dict(tpsa_dct, descriptor=descriptor))
            elif this_cls == 'Tpsa':
                tpsa_flat.append(Tpsa.from_dict(tpsa_dct, descriptor=descriptor))
            else:
                err_mess = f'Unsupported class {this_cls} in TPSA dict.'
                raise ValueError(err_mess)
        return cls._from_dict_components(tpsa_flat, dct)

    def copy(self) -> Self:
        """Return an independent copy of the tensor."""
        return self._new_like([tpsa.copy() for tpsa in self.flat])

    def to_descriptor(
        self,
        descriptor: Descriptor,
        *,
        variable_map: Mapping[str, str] | None = None,
        parameter_map: Mapping[str, str] | None = None,
        truncate: bool = False,
    ) -> Self:
        """Convert the TpsaTensor to a new descriptor."""
        return self._new_like(
            [
                tpsa.to_descriptor(
                    descriptor,
                    variable_map=variable_map,
                    parameter_map=parameter_map,
                    truncate=truncate,
                )
                for tpsa in self.flat
            ]
        )

    def _locate(self, key: int | tuple[int, ...]) -> tuple[int, int]:
        """Return the absolute storage offset and number of indexed axes."""
        keys = key if isinstance(key, tuple) else (key,)
        if not keys or len(keys) > self.rank:
            err_mess = 'Incorrect number of indices'
            raise IndexError(err_mess)
        offset = self._offset
        for axis, raw in enumerate(keys):
            if isinstance(raw, (bool, np.bool_)):
                err_mess = 'Boolean indices are not supported'
                raise TypeError(err_mess)
            idx = index(raw)
            dim = self._shape[axis]
            if not -dim <= idx < dim:
                err_mess = f'Index {idx} out of bounds for axis {axis}'
                raise IndexError(err_mess)
            offset += (idx % dim) * self._strides[axis]
        return offset, len(keys)

    def __getitem__(self, key: int | tuple[int, ...]) -> TpsaSeriesT | TpsaTensor[TpsaSeriesT]:
        """Return the TPSA or a tensor view corresponding to the given index or indices."""
        offset, consumed = self._locate(key)

        if consumed == self.rank:
            # Full indices down to one TPSA
            return self._flat[offset]

        # Create a view without copying the TPSAs.
        view = object.__new__(type(self))
        view._flat = self._flat
        view._shape = self._shape[consumed:]
        view._strides = self._strides[consumed:]
        view._offset = offset
        view._descriptor = self._descriptor
        view._pointer_array = self._pointer_array
        view._is_complex = self._is_complex
        return view

    def __setitem__(
        self, key: int | tuple[int, ...], value: TpsaSeriesT | TpsaTensor[TpsaSeriesT]
    ) -> None:
        """Set the TPSA or tensor view corresponding to the given index or indices."""
        offset, consumed = self._locate(key)
        replacement: tuple[TpsaSeriesT, ...]
        if consumed == self.rank:
            # Individual Tpsa replacement
            if not isinstance(value, (Tpsa, ComplexTpsa)):
                err_mess = 'Expected a TPSA'
                raise TypeError(err_mess)
            replacement = (value,)

        else:
            # Partial replace by view
            if not isinstance(value, TpsaTensor):
                err_mess = 'Expected a TpsaTensor'
                raise TypeError(err_mess)
            expected_shape = self.shape[consumed:]
            if value.shape != expected_shape:
                err_mess = f'Expected shape {expected_shape}, got {value.shape}'
                raise ValueError(err_mess)
            replacement = tuple(value.flat)

        pointers = tuple(t.ptr for t in replacement)

        # Validate everything before changing shared storage.
        if any(tpsa.descriptor is not self.descriptor for tpsa in replacement):
            err_mess = 'Descriptors must be identical'
            raise ValueError(err_mess)
        if any(isinstance(tpsa, ComplexTpsa) != self.is_complex for tpsa in replacement):
            err_mess = 'Cannot mix real and complex TPSAs'
            raise TypeError(err_mess)

        # Replace the Python references.
        count = len(replacement)
        self._flat[offset : offset + count] = replacement

        # Keep the native pointer array in sync.
        for i, ptr in enumerate(pointers):
            self._pointer_array[offset + i] = ptr

    def __len__(self) -> int:
        """Return the size of the first dimension of the tensor."""
        return self._shape[0]

    def __iter__(self) -> Iterator[TpsaSeriesT | TpsaTensor[TpsaSeriesT]]:
        """Return an iterator over the first dimension of the tensor."""
        return (self[i] for i in range(len(self)))

    def __contains__(self, tpsa: TpsaSeries) -> bool:
        return any(t is tpsa for t in self.flat)

    def __repr__(self) -> str:
        """Return a string representation of the tensor."""
        return f'{self.__class__.__name__}(shape={self.shape}, size={self.size})'

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the tensor as a tuple of dimensions."""
        return self._shape

    @property
    def rank(self) -> int:
        """Return the number of dimensions of the tensor."""
        return len(self._shape)

    @property
    def size(self) -> int:
        """Return the total number of TPSAs in the tensor."""
        return prod(self._shape)

    @property
    def flat(self) -> tuple[TpsaSeriesT, ...]:
        """Tuple view on the tensor's TPSAs as a flattened sequence."""
        return tuple(self._flat[self._offset : self._offset + self.size])

    @property
    def pointer_array(self) -> ffi.CData:
        """Return a C array of pointers to the tensor's TPSAs."""
        return self._pointer_array[self._offset : self._offset + self.size]
        # return ffi.addressof(self._pointer_array, self._offset)

    @property
    def descriptor(self):
        """Return the descriptor associated with the tensor."""
        return self._descriptor

    @property
    def num_vars(self) -> int:
        """Return the number of descriptor variables."""
        return self.descriptor.num_vars

    @property
    def num_params(self) -> int:
        """Return the number of descriptor parameters."""
        return self.descriptor.num_params

    @property
    def degree(self) -> int:
        """Return the maximum degree of the tensor's polynomials."""
        return max(tpsa.degree for tpsa in self.flat)

    @property
    def truncation_order(self) -> int:
        """Return the truncation order of the tensor."""
        orders = {tpsa.truncation_order for tpsa in self.flat}
        if len(orders) != 1:
            err_mess = 'Tensor components have different truncation orders. '
            err_mess += "Use 'truncation_orders' instead."
            raise ValueError(err_mess)
        return orders.pop()

    @property
    def truncation_orders(self) -> np.ndarray:
        """Return the truncation orders per TPSA in the tensor."""
        return np.asarray([tpsa.truncation_order for tpsa in self.flat]).reshape(self.shape)

    @property
    def const_part(self) -> np.ndarray:
        """Return the constant part for each of the tensor's TPSAs."""
        dtype = complex if self.is_complex else float
        return np.asarray([tpsa.const_part for tpsa in self.flat], dtype=dtype).reshape(self.shape)

    @const_part.setter
    def const_part(self, value: Scalar | Sequence[Scalar]) -> None:
        """Set the constant part of every tensor component."""
        values = np.asarray(value)
        if values.ndim == 0:
            values = np.full(self.shape, values.item())
        elif values.shape != self.shape:
            err_mess = f'Expected shape {self.shape}, got {values.shape}'
            raise ValueError(err_mess)
        scalars = tuple(values.flat)
        if not self.is_complex:
            if any(complex(vv).imag != 0 for vv in scalars):
                err_mess = 'Assigning complex coefficients to a real TpsaTensor. Please '
                err_mess += 'promote the tensor to complex first.'
                raise ValueError(err_mess)
            scalars = tuple(float(complex(v).real) for v in scalars)
        for tpsa, scalar in zip(self.flat, scalars, strict=True):
            tpsa.const_part = scalar

    @property
    def is_complex(self) -> bool:
        """Return whether the tensor contains complex TPSAs."""
        return self._is_complex

    def to_complex(self) -> TpsaTensor[ComplexTpsa]:
        """Return a tensor where each TPSA is cast as complex."""
        values: list[ComplexTpsa] = []

        for tpsa in self.flat:
            if isinstance(tpsa, ComplexTpsa):
                values.append(tpsa.copy())
            elif isinstance(tpsa, Tpsa):
                values.append(ComplexTpsa.from_tpsa(tpsa))
            else:
                err_mess = f'Unsupported TPSA type: {type(tpsa).__name__}'
                raise TypeError(err_mess)

        # _new_like preserves the container subclass, but the
        # component type changes from TpsaSeriesT to ComplexTpsa.
        return cast(
            'TpsaTensor[ComplexTpsa]',
            self._new_like(cast('Sequence[TpsaSeriesT]', values)),
        )

    def homogeneous(self, degree: int) -> TpsaTensor[TpsaSeriesT]:
        """Return only the homogeneous part of exactly `degree` for each of the tensor's TPSAs."""
        return self._new_like([tpsa.homogeneous(degree) for tpsa in self.flat])

    def truncate(self, degree: int) -> TpsaTensor[TpsaSeriesT]:
        """Return a tensor where all TPSAs are truncated through `degree`."""
        return self._new_like([tpsa.truncate(degree) for tpsa in self.flat])

    def clear_order(self, degree: int) -> TpsaTensor[TpsaSeriesT]:
        """Return a tensor where all TPSAs have order `degree` removed."""
        return self._new_like([tpsa.clear_order(degree) for tpsa in self.flat])

    def _derivative_orders(self, spec: DerivativeSpec) -> tuple[int, ...]:
        """Resolve one mixed-partial specification to descriptor exponent orders.

        Integer indices are 1-based across variables followed by parameters.
        An all-integer sequence is a full order vector; a sequence mixing
        labels/indices and ``(coordinate, count)`` pairs names directions.
        """
        size = self.descriptor.monomial_length
        if isinstance(spec, Tpsa):
            # Preserve scalar TPSA's identity-monomial convention and checks.
            return self.flat[0]._resolve_single_monomial(spec)

        if isinstance(spec, str):
            factors: tuple[int | str | tuple[int | str, int], ...] = (spec,)
        elif isinstance(spec, Integral):
            if isinstance(spec, (bool, np.bool_)):
                err_mess = 'Derivative coordinate must be a name or 1-based index'
                raise TypeError(err_mess)
            factors = (int(spec),)
        elif isinstance(spec, Sequence) and not isinstance(spec, (bytes, bytearray)):
            factors = tuple(spec)
            if factors and all(isinstance(v, Integral) for v in factors):
                if any(isinstance(v, (bool, np.bool_)) for v in factors):
                    err_mess = 'Derivative orders must be integers, not booleans'
                    raise TypeError(err_mess)
                # `all(isinstance(...))` does not narrow tuple element types for ty.
                orders = tuple(int(v) for v in cast('tuple[int, ...]', factors))
                if len(orders) != size:
                    err_mess = f'Derivative order vector must have length {size}'
                    raise ValueError(err_mess)
                if any(v < 0 for v in orders):
                    err_mess = 'Derivative orders must be nonnegative'
                    raise ValueError(err_mess)
                if not any(orders):
                    err_mess = 'Derivative specification must have positive degree'
                    raise ValueError(err_mess)
                return orders
        else:
            err_mess = 'Invalid derivative specification'
            raise TypeError(err_mess)

        powers = [0] * size
        for factor in factors:
            if isinstance(factor, tuple):
                if len(factor) != 2:
                    err_mess = 'Derivative factor must be (coordinate, multiplicity)'
                    raise ValueError(err_mess)
                coordinate, count = factor
            else:
                coordinate, count = factor, 1
            if not isinstance(coordinate, (str, Integral)) or isinstance(
                coordinate, (bool, np.bool_)
            ):
                err_mess = 'Derivative coordinate must be a name or 1-based index'
                raise TypeError(err_mess)
            if not isinstance(count, Integral) or isinstance(count, (bool, np.bool_)):
                err_mess = 'Derivative multiplicity must be an integer'
                raise TypeError(err_mess)
            if count <= 0:
                err_mess = 'Derivative multiplicity must be positive'
                raise ValueError(err_mess)
            index = self.descriptor.var_or_param_index(coordinate)
            powers[index - 1] += int(count)
        if not any(powers):
            err_mess = 'Derivative specification must have positive degree'
            raise ValueError(err_mess)
        return tuple(powers)

    def partial_derivative(self, spec: DerivativeSpec) -> Self:
        """Differentiate each TPSA without changing the tensor's shape or type.

        ``spec`` accepts a 1-based descriptor index, a variable/parameter
        name, an identity TPSA, a full exponent vector, or a sequence of
        names/indices and ``(coordinate, multiplicity)`` pairs.

        Examples
        --------
        ``tensor.partial_derivative((('q', 2), 'k'))`` computes
        ``d^3 tensor / dq^2 dk``; ``tensor.partial_derivative((2, 0, 1))``
        expresses the same derivative for descriptor ``(q, p; k)``.
        """
        orders = self._derivative_orders(spec)
        result: list[Tpsa | ComplexTpsa] = []
        for component in self.flat:
            if not isinstance(component, (Tpsa, ComplexTpsa)):
                err_mess = 'Tensor contains an unsupported TPSA component'
                raise TypeError(err_mess)
            if self.descriptor.is_valid_monomial(orders):
                result.append(component.derivative(orders))
            elif isinstance(component, ComplexTpsa):
                result.append(self.descriptor.complex_zero(order=component.truncation_order))
            else:
                result.append(self.descriptor.zero(order=component.truncation_order))
        return self._new_like(cast('Sequence[TpsaSeriesT]', result))

    def derivative_tensor(
        self,
        order: int,
        *,
        to_variables: bool = True,
        to_parameters: bool = False,
    ) -> TpsaTensor[TpsaSeriesT]:
        """Append ``order`` derivative axes to the existing tensor shape.

        Each new axis enumerates selected descriptor directions: variables
        first (when ``to_variables=True``), then parameters (when
        ``to_parameters=True``). Every component contains the corresponding
        repeated/mixed partial derivative, still represented as a TPSA.

        A tensor of shape ``(a, b)`` differentiated twice with respect to
        ``n`` selected directions has shape ``(a, b, n, n)``.
        """
        if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
            err_mess = 'Derivative order must be a positive integer'
            raise ValueError(err_mess)
        if not isinstance(to_variables, bool) or not isinstance(to_parameters, bool):
            err_mess = 'Derivative direction flags must be booleans'
            raise TypeError(err_mess)

        indices = (tuple(range(self.num_vars)) if to_variables else ()) + (
            tuple(range(self.num_vars, self.num_vars + self.num_params)) if to_parameters else ()
        )
        if not indices:
            err_mess = 'Select at least one existing variable or parameter'
            raise ValueError(err_mess)
        if len(indices) != (self.num_vars if to_variables else 0) + (
            self.num_params if to_parameters else 0
        ):
            err_mess = 'Selected derivative directions do not match descriptor lengths'
            raise ValueError(err_mess)

        # The original tensor axes are outermost; derivative axes are appended
        # in lexicographic row-major order, so every mixed partial is present.
        result: list[Tpsa | ComplexTpsa] = []
        for component in self.flat:
            if not isinstance(component, (Tpsa, ComplexTpsa)):
                err_mess = 'Tensor contains an unsupported TPSA component'
                raise TypeError(err_mess)
            for positions in product(indices, repeat=int(order)):
                powers = [0] * self.descriptor.monomial_length
                for j in positions:
                    powers[j] += 1
                if self.descriptor.is_valid_monomial(powers):
                    result.append(component.derivative(tuple(powers)))
                elif isinstance(component, ComplexTpsa):
                    result.append(self.descriptor.complex_zero(order=component.truncation_order))
                else:
                    result.append(self.descriptor.zero(order=component.truncation_order))

        shape = self.shape + (len(indices),) * int(order)
        return cast('TpsaTensor[TpsaSeriesT]', TpsaTensor(result, shape=shape))

    def evaluate(
        self,
        coordinates: Sequence[Scalar] | np.ndarray,
        *,
        parameters: Sequence[Scalar] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the map at numerical coordinates."""
        parameter_values: tuple[Scalar, ...] = (
            (0.0,) * self.num_params if parameters is None else tuple(parameters)
        )
        if len(parameter_values) != self.num_params:
            err_mess = f'Expected {self.num_params} parameter values, got {len(parameter_values)}'
            raise ValueError(err_mess)
        if len(coordinates) != self.num_vars:
            err_mess = f'Expected {self.num_vars} coordinate values, got {len(coordinates)}'
            raise ValueError(err_mess)

        point = (*coordinates, *parameter_values)
        if not self.is_complex and any(complex(vv).imag != 0 for vv in point):
            self = self.to_complex()
            warn_mess = 'Assigning complex coordinates to a real TpsaTensor. '
            warn_mess += 'Used a temporary complex tensor for evaluation.'
            warn(warn_mess, UserWarning, stacklevel=2)

        if self.is_complex:
            in_array = ffi.new('double _Complex[]', [complex(value) for value in point])
            out_array = ffi.new('double _Complex[]', self.size)
        else:
            in_array = ffi.new('double[]', [complex(value).real for value in point])
            out_array = ffi.new('double[]', self.size)

        # MAD-NG's map_eval enforces an exact length (num_vars) so we batch the call
        pointers = self.pointer_array
        batch_size = self.num_vars
        for start in range(0, self.size, batch_size):
            count = min(batch_size, self.size - start)
            args = (
                count,
                pointers[start : start + count],
                len(point),
                in_array,
                ffi.addressof(out_array, start),
            )
            if self.is_complex:
                status = lib.madng_ctpsa_protected_map_eval_call(lib.mad_ctpsa_eval, *args)
            else:
                status = lib.madng_tpsa_protected_map_eval_call(lib.mad_tpsa_eval, *args)
            if status:
                self.flat[0]._raise_mad_error()

        if self.is_complex:
            return np.asarray([complex(vv) for vv in out_array]).reshape(self.shape)
        return np.asarray([float(vv) for vv in out_array]).reshape(self.shape)

    @staticmethod
    def _align_descriptors(left, right, *, extend_left=False, extend_right=False):
        """Reconcile descriptors and promote both operands to complex if needed.

        The right operand can be a tensor/map or a scalar TPSA. Promotion
        must be applied to the converted operands that are actually returned.
        """
        from .maps import TpsaMap

        if not isinstance(left, (TpsaTensor, Tpsa, ComplexTpsa)) or not isinstance(
            right, (TpsaTensor, Tpsa, ComplexTpsa)
        ):
            return left, right, False
        # Only allow extension on parameters,not on variables
        if set(left.descriptor.variables) != set(right.descriptor.variables):
            err_mess = 'Maps have different variable sets. '
            err_mess += 'Extend them explicitly to a common descriptor first.'
            raise ValueError(err_mess)

        common = left.descriptor.union(right.descriptor)
        if extend_left and isinstance(left, TpsaMap):
            res_left = left.extend_to(common) if common is not left.descriptor else left
        else:
            res_left = left.to_descriptor(common) if common is not left.descriptor else left
        if extend_right and isinstance(right, TpsaMap):
            res_right = right.extend_to(common) if common is not right.descriptor else right
        else:
            res_right = right.to_descriptor(common) if common is not right.descriptor else right

        complex_needed = (
            isinstance(res_left, ComplexTpsa)
            or isinstance(res_right, ComplexTpsa)
            or (isinstance(res_left, TpsaTensor) and res_left.is_complex)
            or (isinstance(res_right, TpsaTensor) and res_right.is_complex)
        )
        if complex_needed:
            if isinstance(res_left, Tpsa):
                res_left = ComplexTpsa.from_tpsa(res_left)
            elif isinstance(res_left, TpsaTensor) and not res_left.is_complex:
                res_left = res_left.to_complex()
            if isinstance(res_right, Tpsa):
                res_right = ComplexTpsa.from_tpsa(res_right)
            elif isinstance(res_right, TpsaTensor) and not res_right.is_complex:
                res_right = res_right.to_complex()

        return res_left, res_right, complex_needed

    def compose(self, other: TpsaMap[MapSeriesT]) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]:
        """Return the composition ``self ∘ other``.

        ``other`` must contain one component for every descriptor variable. Descriptor
        parameters are automatically appended as identity substitutions.
        """
        from .maps import TpsaMap

        if not isinstance(other, TpsaMap):
            err_mess = f'Can only compose with a TpsaMap! Got {type(other)}'
            raise TypeError(err_mess)
        if other.size != self.num_vars:
            err_mess = f'Expected a map with {self.num_vars} components, got {other.size}'
            raise ValueError(err_mess)

        left, right, complex_needed = self._align_descriptors(
            self, other, extend_left=False, extend_right=True
        )

        result = []
        for _ in range(left.size):
            if left.is_complex:
                result.append(left.descriptor.complex_zero(order=left.descriptor.order))
            else:
                result.append(left.descriptor.zero(order=left.descriptor.order))
        pointer_array = ffi.new('void*[]', [value.ptr for value in result])

        args = (
            -left.size,
            left.pointer_array,
            right.size + len(right._params),
            right.pointer_array_with_params,
            pointer_array,
        )
        if complex_needed:
            status = lib.madng_ctpsa_protected_map_binary_call(lib.mad_ctpsa_compose, *args)
        else:
            status = lib.madng_tpsa_protected_map_binary_call(lib.mad_tpsa_compose, *args)
        if status:
            left.flat[0]._raise_mad_error()
        # Real/complex promotion is dynamic, so the generic result needs a cast.
        return cast('TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]', left._new_like(result))

    __matmul__ = compose

    def __neg__(self) -> Self:
        """Negate every component and preserve the container type."""
        values = [-cast('TpsaSeries', tpsa) for tpsa in self.flat]
        return self._new_like(cast('Sequence[TpsaSeriesT]', values))

    def __add__(
        self, other: Scalar | TpsaSeries | TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]
    ) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]:
        """Add corresponding vector components (not composition)."""
        if isinstance(other, (SupportsFloat, SupportsComplex, Tpsa, ComplexTpsa)):
            components = cast('Sequence[Any]', self.flat)
            return cast(
                'TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]',
                self._new_like([tpsa + other for tpsa in components]),
            )

        if not isinstance(other, TpsaTensor) or other.shape != self.shape:
            err_mess = 'Can only sum scalars, TPSAs, or TpsaTensors with the same shape'
            raise ValueError(err_mess)

        left, right, _ = self._align_descriptors(self, other, extend_left=True, extend_right=True)
        return left._new_like([a + b for a, b in zip(left.flat, right.flat, strict=True)])

    __radd__ = __add__

    def __sub__(
        self, other: Scalar | TpsaSeries | TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]
    ) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]:
        """Subtract corresponding vector components (not composition)."""
        if isinstance(other, (SupportsFloat, SupportsComplex, Tpsa, ComplexTpsa)):
            components = cast('Sequence[Any]', self.flat)
            return cast(
                'TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]',
                self._new_like([tpsa - other for tpsa in components]),
            )

        if not isinstance(other, TpsaTensor) or other.shape != self.shape:
            err_mess = 'Can only subtract scalars, TPSAs, or TpsaTensors with the same shape'
            raise ValueError(err_mess)

        left, right, _ = self._align_descriptors(self, other, extend_left=True, extend_right=True)
        return left._new_like([a - b for a, b in zip(left.flat, right.flat, strict=True)])

    def __rsub__(
        self, other: Scalar | TpsaSeries | TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]
    ) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]:
        """Subtract corresponding vector components (not composition)."""
        if isinstance(other, (SupportsFloat, SupportsComplex, Tpsa, ComplexTpsa)):
            components = cast('Sequence[Any]', self.flat)
            return cast(
                'TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]',
                self._new_like([other - tpsa for tpsa in components]),
            )

        if not isinstance(other, TpsaTensor) or other.shape != self.shape:
            err_mess = 'Can only subtract scalars, TPSAs, or TpsaTensors with the same shape'
            raise ValueError(err_mess)

        left, right, _ = self._align_descriptors(self, other, extend_left=True, extend_right=True)
        return left._new_like([b - a for a, b in zip(left.flat, right.flat, strict=True)])

    def __mul__(self, other: Scalar | TpsaSeries) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]:
        """Multiply all vector components by a scalar or a TPSA."""
        if isinstance(other, (SupportsFloat, SupportsComplex, Tpsa, ComplexTpsa)):
            components = cast('Sequence[Any]', self.flat)
            return cast(
                'TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]',
                self._new_like([tpsa * other for tpsa in components]),
            )
        err_mess = 'Can only multiply scalars and TPSAs.'
        raise ValueError(err_mess)

    __rmul__ = __mul__

    def __truediv__(self, other: Scalar | TpsaSeries) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]:
        """Divide all vector components by a scalar."""
        if isinstance(other, (SupportsFloat, SupportsComplex)):
            components = cast('Sequence[Any]', self.flat)
            return cast(
                'TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]',
                self._new_like([tpsa / other for tpsa in components]),
            )
        err_mess = 'Can only divide by scalars.'
        raise ValueError(err_mess)
