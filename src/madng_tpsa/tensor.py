"""Wrapper for multidimensional collections of TPSAs.

The native MAD-NG library represents any map (of any rank) as an array of TPSA
pointers rather than as a separate C object. This module provides a small
Python container around such an array, and adds map operations as methods to it.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from numbers import Integral
from typing import TYPE_CHECKING, Generic, SupportsComplex, SupportsFloat, SupportsIndex, cast

from ._typing import TpsaSeriesT

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from typing import Any, Literal, Self

    from ._typing import DerivativeSpec, MapSeriesT, Nested, Scalar, TpsaSeries
    from .map import TpsaMap

from math import prod
from operator import index
from string import ascii_letters
from warnings import warn

import numpy as np

from ._cffi import ffi, lib
from .complex_tpsa import ComplexTpsa
from .descriptor import Descriptor
from .formatting import format_polynomial
from .tpsa import Tpsa


class _TensorStorage(Generic[TpsaSeriesT]):
    """Shared TPSA references, native pointers and replacement version."""

    __slots__ = ('values', 'pointers', 'version')

    def __init__(self, values: list[TpsaSeriesT]) -> None:
        self.values = values
        self.pointers = ffi.new('void*[]', [value.ptr for value in values])
        self.version = 0

    def replace(self, indices: Sequence[int], values: Sequence[TpsaSeriesT]) -> None:
        for i, value in zip(indices, values, strict=True):
            self.values[i] = value
            self.pointers[i] = value.ptr
        self.version += 1


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
    """Multidimensional collection of TPSA components.

    TpsaTensor supports NumPy-style indexing, slicing, and transposition.
    Basic indexing and transposition produce lightweight views sharing the
    underlying TPSA objects, without copying their polynomial coefficients.

    Tensor operations include elementwise arithmetic, matrix multiplication,
    general contractions, and polynomial-map composition.
    """

    __slots__ = (
        '_storage',
        '_shape',
        '_strides',
        '_offset',
        '_descriptor',
        '_is_complex',
        '_pointer_cache',
        '_pointer_cache_version',
    )
    __array_priority__ = 1000
    _storage: _TensorStorage[TpsaSeriesT]
    _shape: tuple[int, ...]
    _strides: tuple[int, ...]
    _offset: int
    _descriptor: Descriptor
    _is_complex: bool
    _pointer_cache: ffi.CData | None
    _pointer_cache_version: int

    @property
    def _flat(self) -> list[TpsaSeriesT]:
        """Backing component list (not the logical view)."""
        return self._storage.values

    @property
    def _pointer_array(self) -> ffi.CData:
        """Backing pointer array (the full allocation)."""
        return self._storage.pointers

    def __init__(self, values: Nested[TpsaSeriesT], *, shape: Sequence[int] | None = None) -> None:
        """Create a multidimensional tensor of TPSA components.

        A TpsaTensor stores a rectangular collection of TPSA series with an
        arbitrary number of dimensions. Components can be supplied as a flat
        sequence with an explicit shape, or as a nested sequence from which
        the shape is inferred.

        All components must share the same Descriptor. If any component is a
        ComplexTpsa, all real components are automatically promoted to complex.

        Parameters
        ----------
        values : nested sequence of Tpsa or ComplexTpsa
            Components of the tensor, provided as a sequence of TPSA series.
            Nested sequences must be rectangular, with consistent dimensions
            at every level.

            Components are stored in row-major (C-style) order. Numerical
            scalars are not accepted; each component must be a TPSA series.

        shape : sequence of int, optional
            Shape of the tensor, specified as a tuple or sequence of positive
            integers. If omitted, the shape is inferred from the nesting of
            `values`.

            An explicit shape allows a flat sequence of components to be
            interpreted as a multidimensional tensor. The product of the
            dimensions must equal the number of supplied components.

        Examples
        --------
        Create a one-dimensional tensor:

        >>> d = Descriptor(variables=('x', 'px'), order=4)
        >>> x, px = d.vars()
        >>> a, b, c, e = x, px, x + px, x * px
        >>> tensor = TpsaTensor([a, b, c, e])
        >>> tensor.shape
        (4,)

        Create a two-dimensional tensor from a flat sequence:

        >>> tensor = TpsaTensor([a, b, c, e], shape=(2, 2))
        >>> tensor.shape
        (2, 2)

        Alternatively, use a nested sequence:

        >>> tensor = TpsaTensor([[a, b], [c, e]])
        >>> tensor.shape
        (2, 2)

        NumPy object arrays are also accepted:

        >>> tensor = TpsaTensor(
        ...     np.array([[a, b], [c, e]], dtype=object)
        ... )

        The nesting determines the number of dimensions:

        >>> tensor = TpsaTensor([[[a, b], [c, e]]])
        >>> tensor.shape
        (1, 2, 2)

        Mixed real and complex components are promoted to complex:

        >>> tensor = TpsaTensor([x, 1j * px])
        >>> tensor.is_complex
        True
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
        self._storage = _TensorStorage(cast('list[TpsaSeriesT]', data))
        self._shape = dims
        self._strides = tuple(strides)
        self._offset = 0
        self._descriptor = descriptor
        self._pointer_cache = None
        self._pointer_cache_version = -1

    def _new_like(self, values: Sequence[TpsaSeriesT]) -> Self:
        """Create a new tensor with the same structure."""
        return type(self)(values, shape=self.shape)

    @classmethod
    def _parse_dict(
        cls, dct: dict[str, Any]
    ) -> tuple[Descriptor, tuple[int], Sequence[TpsaSeries]]:
        """Parse and validate a dict representing the tensor."""
        class_name = dct.pop('__class__', None)
        if cls.__name__ != class_name:
            err_mess = f'Wrong dict: expected class {cls.__name__}, got {class_name}'
            raise ValueError(err_mess)
        desc_dct = dct.pop('descriptor', None)
        if desc_dct is None:
            err_mess = 'Descriptor is missing in the dictionary'
            raise ValueError(err_mess)
        descriptor = Descriptor.from_dict(desc_dct)
        shape = dct.pop('shape')
        if shape is None:
            err_mess = 'Shape is missing in the dictionary'
            raise ValueError(err_mess)
        shape = tuple(shape)
        flat = dct.pop('flat', None)
        if flat is None:
            err_mess = 'Flat list is missing in the dictionary'
            raise ValueError(err_mess)
        if prod(shape) != len(flat):
            err_mess = f'Incompatible size in dictionary! Expected {prod(shape)}, got {len(flat)}.'
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
        if dct:
            err_mess = f'Unexpected entries in dictionary: {dct.keys()}.'
            raise ValueError(err_mess)
        return descriptor, shape, tpsa_flat

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> Self:
        """Instantiate a TpsaTensor from a dictionary representation."""
        _, shape, tpsa_flat = cls._parse_dict(dct)
        return cls(tpsa_flat, shape=shape)

    def to_dict(self, tol: float = 1e-14) -> dict[str, Any]:
        """Return a dictionary representation of a TpsaTensor."""
        dct = {
            '__class__': type(self).__name__,
            'descriptor': self.descriptor.to_dict(),
            'shape': self.shape,
        }
        flat = []
        # Only store the TPSAs from the partial view
        for tpsa in self.flat:
            tpsa_dct = tpsa.to_dict(tol=tol, store_descriptor=False)
            flat.append(tpsa_dct)
        dct['flat'] = flat
        return dct

    def copy(self) -> Self:
        """Return an independent copy of the tensor."""
        return self._new_like([tpsa.copy() for tpsa in self.flat])

    def change_descriptor(
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
                tpsa.change_descriptor(
                    descriptor,
                    variable_map=variable_map,
                    parameter_map=parameter_map,
                    truncate=truncate,
                )
                for tpsa in self.flat
            ]
        )

    def _storage_indices(self) -> list[int]:
        """Map logical C-order indices to backing storage, including negative strides."""
        return [
            self._offset + sum(i * stride for i, stride in zip(idx, self._strides, strict=True))
            for idx in product(*(range(n) for n in self._shape))
        ]

    def _basic_index(self, key: object) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        """Parse NumPy-style basic indexing, returning offset/shape/strides."""
        if isinstance(key, tuple) and not key:
            error_message = 'Empty index tuple is not supported'
            raise IndexError(error_message)
        items = list(key if isinstance(key, tuple) else (key,))
        if sum(item is Ellipsis for item in items) > 1:
            error_message = 'Only one ellipsis is allowed'
            raise IndexError(error_message)
        used = sum(item is not Ellipsis and item is not None for item in items)
        if used > self.rank:
            error_message = 'Too many indices'
            raise IndexError(error_message)
        if Ellipsis in items:
            pos = next(i for i, item in enumerate(items) if item is Ellipsis)
            items[pos : pos + 1] = [slice(None)] * (self.rank - used)
        else:
            items += [slice(None)] * (self.rank - used)

        shape: list[int] = []
        strides: list[int] = []
        offset = self._offset
        axis = 0
        for item in items:
            if item is None:
                shape.append(1)
                strides.append(0)
                continue
            dim = self.shape[axis]
            stride = self._strides[axis]
            axis += 1
            if isinstance(item, slice):
                start, stop, step = item.indices(dim)
                count = len(range(start, stop, step))
                if count == 0:
                    error_message = 'Empty TPSA tensor views are not supported'
                    raise IndexError(error_message)
                offset += start * stride
                shape.append(count)
                strides.append(step * stride)
            else:
                if isinstance(item, (bool, np.bool_)):
                    error_message = 'Boolean indices are not supported'
                    raise TypeError(error_message)
                if not isinstance(item, SupportsIndex):
                    error_message = 'Tensor indices must be integers, slices, or ellipsis'
                    raise TypeError(error_message)
                idx = index(item)
                if not -dim <= idx < dim:
                    error_message = f'Index {idx} out of bounds for axis {axis - 1}'
                    raise IndexError(error_message)
                offset += (idx % dim) * stride
        return offset, tuple(shape), tuple(strides)

    def _view(
        self, offset: int, shape: tuple[int, ...], strides: tuple[int, ...]
    ) -> TpsaTensor[TpsaSeriesT]:
        """Return a shared-storage tensor view, never a partial TpsaMap."""
        view = object.__new__(TpsaTensor)
        view._storage = self._storage
        view._shape = shape
        view._strides = strides
        view._offset = offset
        view._descriptor = self._descriptor
        view._is_complex = self._is_complex
        view._pointer_cache = None
        view._pointer_cache_version = -1
        return view

    def __getitem__(self, key: object) -> TpsaSeriesT | TpsaTensor[TpsaSeriesT]:
        """Return a TPSA or a zero-copy basic-indexing view."""
        offset, shape, strides = self._basic_index(key)
        if not shape:
            return self._flat[offset]
        return self._view(offset, shape, strides)

    def __setitem__(self, key: object, value: TpsaSeriesT | TpsaTensor[TpsaSeriesT]) -> None:
        """Replace selected TPSAs atomically, including noncontiguous views."""
        offset, shape, strides = self._basic_index(key)
        target = self._view(offset, shape, strides) if shape else None
        indices = target._storage_indices() if target is not None else [offset]
        if target is None:
            if not isinstance(value, (Tpsa, ComplexTpsa)):
                error_message = 'Expected a TPSA'
                raise TypeError(error_message)
            replacements = (value,)
        else:
            if not isinstance(value, TpsaTensor):
                error_message = 'Expected a TpsaTensor'
                raise TypeError(error_message)
            if value.shape != shape:
                error_message = f'Expected shape {shape}, got {value.shape}'
                raise ValueError(error_message)
            # Materialise before any writes: overlapping source and target are safe.
            replacements = tuple(value.flat)
        if any(t.descriptor is not self.descriptor for t in replacements):
            error_message = 'Descriptors must be identical'
            raise ValueError(error_message)
        if any(isinstance(t, ComplexTpsa) != self.is_complex for t in replacements):
            error_message = 'Cannot mix real and complex TPSAs'
            raise TypeError(error_message)
        self._storage.replace(indices, replacements)

    def transpose(self, *axes: int) -> TpsaTensor[TpsaSeriesT]:
        """Return a zero-copy permuted view; no TPSA coefficients are copied."""
        if not axes:
            axes = tuple(reversed(range(self.rank)))
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if len(axes) != self.rank or any(
            not -self.rank <= index(axis) < self.rank for axis in axes
        ):
            error_message = 'axes must be a permutation of tensor dimensions'
            raise ValueError(error_message)
        normalized = tuple(index(axis) % self.rank for axis in axes)
        if set(normalized) != set(range(self.rank)):
            error_message = 'axes must be a permutation of tensor dimensions'
            raise ValueError(error_message)
        return self._view(
            self._offset,
            tuple(self.shape[i] for i in normalized),
            tuple(self._strides[i] for i in normalized),
        )

    def swapaxes(self, axis1: int, axis2: int) -> TpsaTensor[TpsaSeriesT]:
        """Swap two axes without copying components."""
        axes = list(range(self.rank))
        a, b = index(axis1), index(axis2)
        if not -self.rank <= a < self.rank or not -self.rank <= b < self.rank:
            error_message = 'axis out of bounds'
            raise ValueError(error_message)
        a, b = a % self.rank, b % self.rank
        axes[a], axes[b] = axes[b], axes[a]
        return self.transpose(*axes)

    @property
    def T(self) -> TpsaTensor[TpsaSeriesT]:  # noqa: N802
        """Transpose view (all axes reversed, like NumPy's ``.T``)."""
        return self.transpose()

    def __len__(self) -> int:
        """Return the size of the first dimension of the tensor."""
        return self._shape[0]

    def __iter__(self) -> Iterator[TpsaSeriesT | TpsaTensor[TpsaSeriesT]]:
        """Return an iterator over the first dimension of the tensor."""
        return (self[i] for i in range(len(self)))

    def __contains__(self, tpsa: TpsaSeries) -> bool:
        """Test object identity among components, *not* polynomial equality."""
        return any(t is tpsa for t in self.flat)

    def show(self, *, max_tpsa: int | None = None, max_terms: int | None = 3) -> str:
        """Return a string representation of the tensor, specifying how many TPSAs to show."""
        if max_tpsa is not None and max_tpsa < 0:
            message = 'max_tpsa must be non-negative or None'
            raise ValueError(message)
        if max_terms is not None and max_terms < 0:
            message = 'max_terms must be non-negative or None'
            raise ValueError(message)
        flat = self.flat[:max_tpsa]
        end = '' if len(flat) == self.size else ', ...'
        labels = self.descriptor.variables + self.descriptor.parameters
        components = []
        for tpsa in flat:
            tt = format_polynomial(tpsa=tpsa, labels=labels, style='unicode', max_terms=max_terms)
            components.append(tt)
        name = type(self).__name__
        return (
            f'{name}(shape={self.shape}, size={self.size}, values=[{", ".join(components)}{end}])'
        )

    def __repr__(self) -> str:
        """Return a compact, human-readable polynomial representation."""
        return self.show(max_tpsa=3)

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
        return tuple(self._flat[i] for i in self._storage_indices())

    @staticmethod
    def _c_strides(shape: Sequence[int]) -> tuple[int, ...]:
        strides = [1] * len(shape)
        for axis in range(len(shape) - 2, -1, -1):
            strides[axis] = strides[axis + 1] * shape[axis + 1]
        return tuple(strides)

    @property
    def pointer_array(self) -> ffi.CData:
        """Return a length-aware CFFI array in logical view order.

        Full C-contiguous owners share the backing array. For all other views,
        construct and cache a packed array of TPSA pointers. No TPSAs are copied.
        The cache is invalidated after a replacement through any sibling view.
        """
        if (
            self._offset == 0
            and self.size == len(self._storage.values)
            and self._strides == self._c_strides(self.shape)
        ):
            return self._storage.pointers
        if self._pointer_cache is None or self._pointer_cache_version != self._storage.version:
            self._pointer_cache = ffi.new('void*[]', [t.ptr for t in self.flat])
            self._pointer_cache_version = self._storage.version
        return self._pointer_cache

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
        wrt: Literal['variables', 'parameters', 'all'] = 'variables',
        max_components: int | None = 1_000_000,
    ) -> TpsaTensor[TpsaSeriesT]:
        """Append ``order`` derivative axes to the existing tensor shape.

        Each new axis enumerates selected descriptor directions: variables
        first (when ``wrt='variables'`` or ``wrt='all'``), then parameters
        (when ``wrt='parameters'`` or ``wrt='all'``). Every component
        contains the corresponding repeated/mixed partial derivative, still
        represented as a TPSA.

        A tensor of shape ``(a, b)`` differentiated twice with respect to
        ``n`` selected directions has shape ``(a, b, n, n)``.
        """
        if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
            err_mess = 'Derivative order must be a positive integer'
            raise ValueError(err_mess)
        if wrt not in ['variables', 'parameters', 'all']:
            err_mess = "Derivative direction argument 'wrt' has to be one of "
            err_mess += "['variables', 'parameters', 'all']."
            raise TypeError(err_mess)
        to_variables = wrt == 'variables' or wrt == 'all'
        to_parameters = wrt == 'parameters' or wrt == 'all'

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

        # The output contains `size * n**order` independently allocated TPSAs.
        # Caching native differentiation does not reduce output storage.
        if max_components is not None:
            if isinstance(max_components, bool) or not isinstance(max_components, Integral):
                err_mess = 'max_components must be a positive integer or None'
                raise TypeError(err_mess)
            if max_components <= 0:
                err_mess = 'max_components must be positive'
                raise ValueError(err_mess)
            component_count = self.size * len(indices) ** int(order)
            if component_count > max_components:
                err_mess = (
                    f'Derivative tensor requires {component_count} TPSA components, '
                    f'exceeding the limit {max_components}. Use partial_derivative() '
                    'or explicitly raise max_components if enough memory is available.'
                )
                raise ValueError(err_mess)

        # The original tensor axes are outermost; derivative axes are appended
        # in lexicographic row-major order, so every mixed partial is present.
        result: list[Tpsa | ComplexTpsa] = []
        for component in self.flat:
            if not isinstance(component, (Tpsa, ComplexTpsa)):
                err_mess = 'Tensor contains an unsupported TPSA component'
                raise TypeError(err_mess)
            # Symmetric derivative axes contain permutation-equivalent mixed
            # partials. Differentiate each multi-index only once per component.
            # Duplicate results must still be copied because TPSAs are mutable.
            cache: dict[tuple[int, ...], Tpsa | ComplexTpsa] = {}
            for positions in product(indices, repeat=int(order)):
                powers = [0] * self.descriptor.monomial_length
                for j in positions:
                    powers[j] += 1
                key = tuple(powers)
                cached = cache.get(key)
                if cached is not None:
                    result.append(cached.copy())
                    continue
                if self.descriptor.is_valid_monomial(key):
                    derivative = component.derivative(key)
                elif isinstance(component, ComplexTpsa):
                    derivative = self.descriptor.complex_zero(order=component.truncation_order)
                else:
                    derivative = self.descriptor.zero(order=component.truncation_order)
                cache[key] = derivative
                result.append(derivative)

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
    def _align_descriptors(
        left, right, *, extend_left=False, extend_right=False, reorder_variables=False
    ):
        """Reconcile descriptors and promote both operands to complex if needed.

        The right operand can be a tensor/map or a scalar TPSA. Promotion
        must be applied to the converted operands that are actually returned.
        """
        from .map import TpsaMap

        if not isinstance(left, (TpsaTensor, Tpsa, ComplexTpsa)) or not isinstance(
            right, (TpsaTensor, Tpsa, ComplexTpsa)
        ):
            return left, right, False
        # Elementwise arithmetic requires identical variable order. Composition
        # can match equivalent sets of input variables by descriptor label.
        left_variables = left.descriptor.variables
        right_variables = right.descriptor.variables
        compatible = (
            set(left_variables) == set(right_variables)
            if reorder_variables
            else left_variables == right_variables
        )
        if not compatible:
            err_mess = 'Incompatible descriptor variables or variable order'
            raise ValueError(err_mess)

        common = left.descriptor.union(right.descriptor)
        if extend_left and isinstance(left, TpsaMap):
            res_left = left.extend_descriptor(common) if common is not left.descriptor else left
        else:
            res_left = left.change_descriptor(common) if common is not left.descriptor else left
        if extend_right and isinstance(right, TpsaMap):
            res_right = right.extend_descriptor(common) if common is not right.descriptor else right
        else:
            res_right = right.change_descriptor(common) if common is not right.descriptor else right

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
        from .map import TpsaMap

        if not isinstance(other, TpsaMap):
            err_mess = f'Can only compose with a TpsaMap! Got {type(other)}'
            raise TypeError(err_mess)
        if other.size != self.num_vars:
            err_mess = f'Expected a map with {self.num_vars} components, got {other.size}'
            raise ValueError(err_mess)

        left, right, complex_needed = self._align_descriptors(
            self, other, extend_left=False, extend_right=True, reorder_variables=True
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

    def __lshift__(self, other: TpsaMap[MapSeriesT]) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]:
        """Substitute a map: ``T << M`` is ``T.compose(M)``."""
        from .map import TpsaMap

        if not isinstance(other, TpsaMap):
            return NotImplemented
        return self.compose(other)

    def __rrshift__(self, other: TpsaMap[MapSeriesT]) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]:
        """Use Dragt's reversed order: ``M >> T`` is ``T.compose(M)``."""
        from .map import TpsaMap

        if not isinstance(other, TpsaMap):
            return NotImplemented
        return self.compose(other)

    @staticmethod
    def _contract_einsum(
        expression: str, *operands: TpsaTensor[TpsaSeriesT] | np.ndarray
    ) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa] | Tpsa | ComplexTpsa:
        """Contract TPSA and numeric operands using explicit Einstein indices.

        Supports ASCII single-letter subscripts, repeated labels and explicit
        ``->`` output; ellipsis and NumPy's advanced broadcasting are omitted.
        """
        if '...' in expression or '->' not in expression:
            error_message = 'Use explicit einsum indices without ellipsis'
            raise ValueError(error_message)
        source, target = expression.replace(' ', '').split('->', 1)
        inputs = source.split(',')
        if len(inputs) != len(operands):
            error_message = 'Number of operands does not match einsum expression'
            raise ValueError(error_message)
        if any(not isinstance(op, (TpsaTensor, np.ndarray)) for op in operands):
            error_message = 'Operands must be TpsaTensor or numeric ndarray'
            raise TypeError(error_message)
        if any(any(letter not in ascii_letters for letter in part) for part in (*inputs, target)):
            error_message = 'Only ASCII einsum labels are supported'
            raise ValueError(error_message)
        if len(set(target)) != len(target):
            error_message = 'Output indices must be unique'
            raise ValueError(error_message)

        first = next((op for op in operands if isinstance(op, TpsaTensor)), None)
        if first is None:
            error_message = 'At least one operand must be a TpsaTensor'
            raise TypeError(error_message)
        descriptor = first.descriptor
        numeric = any(isinstance(op, np.ndarray) and np.iscomplexobj(op) for op in operands)
        complex_needed = numeric or any(
            isinstance(op, TpsaTensor) and op.is_complex for op in operands
        )
        compatible: list[TpsaTensor[TpsaSeriesT] | np.ndarray] = []
        for op in operands:
            if isinstance(op, TpsaTensor):
                if op.descriptor.variables != descriptor.variables:
                    error_message = 'Incompatible variable order in einsum operands'
                    raise ValueError(error_message)
                descriptor = descriptor.union(op.descriptor)
        for op in operands:
            if isinstance(op, TpsaTensor) and op.descriptor is not descriptor:
                op = op.change_descriptor(descriptor)
            compatible.append(op)

        sizes: dict[str, int] = {}
        for labels, op in zip(inputs, compatible, strict=True):
            shape = op.shape
            if len(labels) != len(shape):
                error_message = 'Einsum indices do not match operand rank'
                raise ValueError(error_message)
            for label, length in zip(labels, shape, strict=True):
                if label in sizes and sizes[label] != length:
                    error_message = f'Incompatible dimensions for index {label}'
                    raise ValueError(error_message)
                sizes[label] = length
        if any(label not in sizes for label in target):
            error_message = 'Output contains an undefined index'
            raise ValueError(error_message)
        reduced = tuple(label for label in sizes if label not in target)
        output_shape = tuple(sizes[label] for label in target)
        output: list[Tpsa | ComplexTpsa] = []
        for out_idx in product(*(range(length) for length in output_shape)):
            current = dict(zip(target, out_idx, strict=True))
            total = descriptor.complex_zero() if complex_needed else descriptor.zero()
            for contraction_idx in product(*(range(sizes[label]) for label in reduced)):
                current.update(zip(reduced, contraction_idx, strict=True))
                # Numeric ndarray elements and TPSA components have disparate
                # multiplication overloads; the accumulator is dynamically typed.
                term: Any = 1.0
                for labels, op in zip(inputs, compatible, strict=True):
                    idx = tuple(current[label] for label in labels)
                    factor: Any = op[idx]
                    term = term * factor
                total = total + term
            output.append(total)
        if not output_shape:
            return output[0]
        return TpsaTensor(output, shape=output_shape)

    @staticmethod
    def einsum(
        expression: str, *operands: TpsaTensor[Any] | np.ndarray
    ) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa] | Tpsa | ComplexTpsa:
        """Perform an explicit Einstein contraction over TPSA-valued arrays."""
        return TpsaTensor._contract_einsum(expression, *operands)

    def tensordot(
        self,
        other: TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa] | np.ndarray,
        axes: int | tuple[Sequence[int], Sequence[int]] = 2,
    ) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa] | Tpsa | ComplexTpsa:
        """Contract the supplied axes following NumPy tensordot conventions."""
        if isinstance(axes, int):
            n = axes
            if not 0 <= n <= min(self.rank, len(other.shape)):
                error_message = 'Invalid contraction rank'
                raise ValueError(error_message)
            left_axes = tuple(range(self.rank - n, self.rank))
            right_axes = tuple(range(n))
        else:
            left_axes, right_axes = tuple(axes[0]), tuple(axes[1])
        if any(not -self.rank <= index(a) < self.rank for a in left_axes) or any(
            not -len(other.shape) <= index(a) < len(other.shape) for a in right_axes
        ):
            error_message = 'Contraction axis out of bounds'
            raise ValueError(error_message)
        left_axes = tuple(index(a) % self.rank for a in left_axes)
        right_axes = tuple(index(a) % len(other.shape) for a in right_axes)
        if (
            len(left_axes) != len(right_axes)
            or len(set(left_axes)) != len(left_axes)
            or len(set(right_axes)) != len(right_axes)
        ):
            error_message = 'Invalid contraction axes'
            raise ValueError(error_message)
        letters = iter(ascii_letters)
        a = [next(letters) for _ in range(self.rank)]
        b = [next(letters) for _ in range(len(other.shape))]
        for i, j in zip(left_axes, right_axes, strict=True):
            b[j] = a[i]
        out = [v for i, v in enumerate(a) if i not in left_axes]
        out += [v for i, v in enumerate(b) if i not in right_axes]
        return self.einsum(f'{"".join(a)},{"".join(b)}->{"".join(out)}', self, other)

    def inner(
        self, other: TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa] | np.ndarray
    ) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa] | Tpsa | ComplexTpsa:
        """Contract the last axis of each operand (NumPy ``inner``)."""
        return self.tensordot(other, axes=([-1], [-1]))

    def trace(
        self, *, axis1: int = 0, axis2: int = 1
    ) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa] | Tpsa | ComplexTpsa:
        """Sum the diagonal over two axes."""
        if self.rank < 2:
            error_message = 'trace requires a tensor of rank at least two'
            raise ValueError(error_message)
        if not -self.rank <= index(axis1) < self.rank or not -self.rank <= index(axis2) < self.rank:
            error_message = 'trace axis out of bounds'
            raise ValueError(error_message)
        a, b = index(axis1) % self.rank, index(axis2) % self.rank
        if a == b:
            error_message = 'trace axes must differ'
            raise ValueError(error_message)
        letters = list(ascii_letters[: self.rank])
        letters[b] = letters[a]
        output = ''.join(letters[i] for i in range(self.rank) if i not in (a, b))
        return self.einsum(f'{"".join(letters)}->{output}', self)

    def __matmul__(
        self,
        other: TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa] | np.ndarray,
    ) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa] | Tpsa | ComplexTpsa:
        """NumPy-style matrix multiplication, *not* map composition."""
        if self.rank < 1 or len(other.shape) < 1:
            return NotImplemented
        axes = ([-1], [-1] if len(other.shape) == 1 else [-2])
        # tensordot preserves the remaining axes. For batch matmul this differs
        # from NumPy broadcast semantics: reject stacked operands for now.
        if self.rank > 2 or len(other.shape) > 2:
            error_message = 'Batched matmul is not yet supported; use einsum'
            raise NotImplementedError(error_message)
        return self.tensordot(other, axes=axes)

    def matmul(
        self, other: TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa] | np.ndarray
    ) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa] | Tpsa | ComplexTpsa:
        """Multiply rank-one/rank-two tensor arrays, identically to ``@``."""
        return self.__matmul__(other)

    def __rmatmul__(
        self, other: np.ndarray
    ) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa] | Tpsa | ComplexTpsa:
        """Matrix-multiply a numeric ndarray with a TPSA tensor."""
        if not isinstance(other, np.ndarray):
            return NotImplemented
        if other.ndim not in (1, 2) or self.rank not in (1, 2):
            error_message = 'Batched matmul is not yet supported'
            raise NotImplementedError(error_message)
        return TpsaTensor.einsum(
            'ij,jk->ik'
            if other.ndim == self.rank == 2
            else (
                'i,i->'
                if other.ndim == self.rank == 1
                else ('i,ij->j' if other.ndim == 1 else 'ij,j->i')
            ),
            other,
            self,
        )

    def __eq__(self, other: object) -> bool:
        """Compare to another tensor; True if all TPSAs and the descriptor are the same."""
        if not isinstance(other, TpsaTensor):
            return False
        if self.descriptor != other.descriptor:
            return False
        if len(self.flat) != len(other.flat):
            return False
        return all(tt1 == tt2 for tt1, tt2 in zip(self.flat, other.flat, strict=True))

    def __neg__(self) -> Self:
        """Negate every component and preserve the container type."""
        values = [-cast('TpsaSeries', tpsa) for tpsa in self.flat]
        return self._new_like(cast('Sequence[TpsaSeriesT]', values))

    def _elementwise(
        self, other: object, operation: str, *, reverse: bool = False
    ) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]:
        """Implement consistent descriptor alignment for elementwise arithmetic."""
        left: TpsaTensor[Any] = self
        right: object = other
        if isinstance(other, np.ndarray):
            if other.shape != self.shape:
                error_message = 'Numeric array must have matching tensor shape'
                raise ValueError(error_message)
            # Numeric entries are independent of descriptor variables.
            right = TpsaTensor(
                [self.descriptor.constant(value) for value in other.flat],
                shape=self.shape,
            )
        if isinstance(right, TpsaTensor):
            if right.shape != self.shape:
                error_message = 'Tensor operands must have the same shape'
                raise ValueError(error_message)
            left, right, _ = self._align_descriptors(self, right)
            operands = right.flat
        elif isinstance(right, (Tpsa, ComplexTpsa)):
            left, right, _ = self._align_descriptors(self, right)
            operands = (right,) * self.size
        elif isinstance(right, (SupportsFloat, SupportsComplex)):
            operands = (right,) * self.size
        else:
            return NotImplemented
        # SeriesT is bounded by _TpsaBase; arithmetic specialises in Tpsa and
        # ComplexTpsa. Dynamic intermediate operands bridge those overloads.
        typed_operands = cast('tuple[Any, ...]', tuple(operands))
        left_components = cast('tuple[Any, ...]', left.flat)
        if operation == 'add':
            values = [
                b + a if reverse else a + b
                for a, b in zip(left_components, typed_operands, strict=True)
            ]
        elif operation == 'sub':
            values = [
                b - a if reverse else a - b
                for a, b in zip(left_components, typed_operands, strict=True)
            ]
        elif operation == 'mul':
            values = [a * b for a, b in zip(left_components, typed_operands, strict=True)]
        else:
            error_message = 'Unsupported arithmetic operation'
            raise ValueError(error_message)
        return cast('TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]', left._new_like(values))

    def __add__(self, other: object) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]:
        """Add TPSAs componentwise after label-compatible descriptor alignment."""
        return self._elementwise(other, 'add')

    __radd__ = __add__

    def __sub__(self, other: object) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]:
        """Subtract TPSAs componentwise."""
        return self._elementwise(other, 'sub')

    def __rsub__(self, other: object) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]:
        """Subtract tensor components from the other operand."""
        return self._elementwise(other, 'sub', reverse=True)

    def __mul__(self, other: object) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]:
        """Multiply by a numeric or TPSA scalar, not a tensor."""
        if isinstance(other, (TpsaTensor, np.ndarray)):
            return NotImplemented
        return self._elementwise(other, 'mul')

    __rmul__ = __mul__

    def __truediv__(self, other: Scalar) -> TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]:
        """Divide tensor components by a numeric scalar."""
        if not isinstance(other, (SupportsFloat, SupportsComplex)):
            return NotImplemented
        components = cast('tuple[Any, ...]', self.flat)
        return cast(
            'TpsaTensor[Tpsa] | TpsaTensor[ComplexTpsa]',
            self._new_like([tpsa / other for tpsa in components]),
        )
