"""Map operations for MAD-NG TPSA series.

The native MAD-NG library represents a map as an array of TPSA pointers rather
than as a separate C object. This module provides a small Python container
around such an array, and adds map operations as methods to it.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, SupportsComplex, SupportsFloat, cast, overload

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Self

    from ._typing import MapSeriesT, Scalar, SelectionValue, TpsaSeries
    from .tpsa import Tpsa

from collections.abc import Sequence
from numbers import Integral
from warnings import warn

import numpy as np

from ._cffi import ffi, lib
from ._typing import TpsaSeriesT
from .complex_tpsa import ComplexTpsa
from .descriptor import Descriptor
from .tensor import TpsaTensor
from .tpsa import Tpsa


class TpsaMap(TpsaTensor[TpsaSeriesT]):
    """A vector-valued TPSA map sharing one descriptor."""

    __slots__ = ('_coord_names', '_params')
    _coord_names: tuple[str, ...]
    _params: tuple[TpsaSeries, ...]

    @overload
    def __init__(self, descriptor: Descriptor, /, **coordinates: Scalar | TpsaSeriesT) -> None: ...
    @overload
    def __init__(
        self, components: Sequence[Scalar | TpsaSeriesT], descriptor: Descriptor | None = None
    ) -> None: ...
    @overload
    def __init__(
        self, *, descriptor: Descriptor | None = None, **coordinates: Scalar | TpsaSeriesT
    ) -> None: ...

    def __init__(
        self,
        components: Descriptor | Sequence[Scalar | TpsaSeriesT] | None = None,
        descriptor: Descriptor | None = None,
        **coordinates: Scalar | TpsaSeriesT,
    ) -> None:
        """Create a TPSA map.

        Parameters
        ----------
        components : Descriptor or sequence, optional
            A descriptor to create an identity map, or a sequence of
            component values in descriptor variable order.
        descriptor : Descriptor, optional
            Explicit descriptor, if not inferable from the components.
        **coordinates
            Named components. Unspecified coordinates retain their
            identity-map values.

        Scalar values are interpreted as offsets from the corresponding
        identity coordinate.

        Examples
        --------
        The following are examples of valid initialisations:
            TpsaMap(descriptor)       # Identity map
            TpsaMap([x, px, y, py])
            TpsaMap([x, 0, 0, 0])
            TpsaMap([0.1, 0, 0.2, -0.1], descriptor=descriptor)
            TpsaMap(x=x, px=px, y=y, py=py)
            TpsaMap(x=x)
            TpsaMap(x=0.1, descriptor=descriptor)
        """
        # Interpret the positional arguments
        if isinstance(components, Descriptor):
            if descriptor is not None:
                err_mess = 'Descriptor specified twice'
                raise TypeError(err_mess)
            descriptor = components
            components = None
        if components is not None:
            if coordinates:
                err_mess = 'Cannot mix positional components with named coordinates'
                raise TypeError(err_mess)
            if not isinstance(components, Sequence):
                err_mess = 'Components must be a sequence or a Descriptor'
                raise TypeError(err_mess)
            if any(isinstance(tt, Sequence) for tt in components):
                err_mess = 'Components of TpsaMap cannot be nested! Use TpsaTensor instead'
                raise ValueError(err_mess)
            components = list(components)

        # Determine the descriptor
        values = components if components is not None else coordinates.values()
        for value in values:
            if not isinstance(value, (Tpsa, ComplexTpsa)):
                continue
            if descriptor is None:
                descriptor = value.descriptor
        if descriptor is None:
            err_mess = 'Cannot infer the descriptor. Provide it explicitly.'
            raise ValueError(err_mess)
        self._descriptor = descriptor

        # Convert positional components to named coordinates
        names = tuple(descriptor.variables)
        if components is not None:
            if len(components) != len(names):
                err_mess = f'Expected {len(names)} components, got {len(components)}'
                raise ValueError(err_mess)
            coordinates.update(dict(zip(names, components, strict=True)))

        # Validate coordinate names
        unknown = coordinates.keys() - names
        if unknown:
            err_mess = f'Unknown coordinates: {sorted(unknown)}'
            raise ValueError(err_mess)

        # Construct all coordinates
        coords = []
        for i, coord in enumerate(names):
            val = coordinates.get(coord, 0.0)
            if isinstance(val, (Tpsa, ComplexTpsa)):
                coords.append(val)
            elif isinstance(val, (SupportsFloat, SupportsComplex)):
                coords.append(descriptor.var(i + 1, val))
            else:
                err_mess = 'All arguments must be Tpsa, ComplexTpsa, or Scalar'
                raise TypeError(err_mess)

        super().__init__(values=cast('Sequence[TpsaSeriesT]', coords), shape=(descriptor.num_vars,))
        self._coord_names = names

        # Add identity operators for the parameters; needed for composition
        params = tuple(descriptor.params())
        if self.is_complex:
            params = tuple(ComplexTpsa.from_tpsa(tpsa) for tpsa in params)
        self._params = params

    def _new_like(self, values: Sequence[TpsaSeriesT]) -> Self:
        return type(self)(values)

    @classmethod
    def _from_dict_components(cls, values, dct) -> Self:
        shape = tuple(dct['shape'])
        if len(shape) != 1:
            err_mess = 'A TpsaMap always has to be rank 1!'
            raise ValueError(err_mess)
        if shape[0] != len(values):
            err_mess = f"Incompatible 'shape' in dictionary! Expected {len(values)}, got {shape}."
            raise ValueError(err_mess)
        return cls(values)

    def to_complex(self) -> TpsaMap[ComplexTpsa]:
        """Promote the components, preserving the TpsaMap container."""
        return cast('TpsaMap[ComplexTpsa]', super().to_complex())

    def __getitem__(self, key: int | tuple[int, ...]) -> TpsaSeriesT:
        """Return the TPSA at the specified map coordinate."""
        result = super().__getitem__(key)
        if not isinstance(result, (Tpsa, ComplexTpsa)):
            err_mess = 'TpsaMap requires a complete coordinate index'
            raise IndexError(err_mess)
        return result

    def change_descriptor(
        self,
        descriptor: Descriptor,
        *,
        variable_map: Mapping[str, str] | None = None,
        parameter_map: Mapping[str, str] | None = None,
        truncate: bool = False,
    ) -> Self:
        """Convert a full map, including its output-coordinate ordering."""
        rename = variable_map or {}
        output_names = tuple(rename.get(name, name) for name in self.coord_names)
        if len(set(output_names)) != len(output_names) or set(output_names) != set(
            descriptor.variables
        ):
            err_mess = 'Target variables must match the map outputs. '
            err_mess += 'Use extend_descriptor() to add new coordinates.'
            raise ValueError(err_mess)
        converted = {
            rename.get(name, name): component.change_descriptor(
                descriptor,
                variable_map=variable_map,
                parameter_map=parameter_map,
                truncate=truncate,
            )
            for name, component in self.coordinates.items()
        }
        return type(self)(
            [converted[name] for name in descriptor.variables],
            descriptor=descriptor,
        )

    def extend_descriptor(self, descriptor: Descriptor, *, merge: bool = False) -> Self:
        """Extend a coordinate map, inserting identity components.

        If merge is False, target descriptor must contain every existing
        variable and parameter.

        If merge is True, first form a union, preserving the
        requested descriptor's coordinate ordering.
        """
        if merge:
            descriptor = descriptor.union(self.descriptor)
        if not set(self.descriptor.variables) <= set(descriptor.variables):
            err_mess = 'Target descriptor must contain all original variables'
            raise ValueError(err_mess)
        if not set(self.descriptor.parameters) <= set(descriptor.parameters):
            err_mess = 'Target descriptor must contain all original parameters'
            raise ValueError(err_mess)
        converted = {
            name: component.change_descriptor(descriptor)
            for name, component in self.coordinates.items()
        }
        values = [
            converted[name] if name in converted else descriptor.var(name)
            for name in descriptor.variables
        ]
        return type(self)(values, descriptor=descriptor)

    def keys(self):
        """Return the coordinate names as keys on the coordinates dict."""
        return self.coordinates.keys()

    def values(self):
        """Return the coordinate TPSAs as values on the coordinates dict."""
        return self.coordinates.values()

    def items(self):
        """Return an items view on the coordinates dict."""
        return self.coordinates.items()

    def __getattr__(self, name: str) -> TpsaSeriesT:
        try:
            names = object.__getattribute__(self, '_coord_names')
            index = names.index(name)
            flat = object.__getattribute__(self, '_flat')
            offset = object.__getattribute__(self, '_offset')
        except (AttributeError, ValueError):
            raise AttributeError(name) from None
        return flat[offset + index]

    def __repr__(self) -> str:
        components = []
        for coord, tpsa in self.coordinates.items():
            if len(tpsa) > 3:
                components.append(f'{coord}=...')
            else:
                components.append(f'{coord}={tpsa}')
        return f'TpsaMap({", ".join(components)})'

    @property
    def coord_names(self) -> tuple[str, ...]:
        """Coordinate names of the map."""
        return self._coord_names

    @property
    def coords(self) -> tuple[TpsaSeriesT, ...]:
        """Coordinate TPSAs of the map."""
        return tuple(self.flat)

    @property
    def coordinates(self) -> Mapping[str, TpsaSeriesT]:
        """Read-only dict of coordinate names -> TPSAs of the map."""
        return MappingProxyType(dict(zip(self._coord_names, self.flat, strict=True)))

    @property
    def pointer_array_with_params(self) -> ffi.CData:
        """Return a full C array of pointers to the map's TPSAs, including parameters."""
        return ffi.new('void*[]', [t.ptr for t in (*self.flat, *self._params)])

    def set_const_part(self, values: Sequence[Scalar] | np.ndarray) -> None:
        """Set the map's constant vector (deprecated; use ``const_part``)."""
        warn(
            "The method 'set_const_part' is deprecated; use the setter for 'const_part' instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.const_part = np.asarray(values).tolist()

    def _coordinate_index(self, coord: str | int) -> int:
        """Resolve a coordinate name or zero-based integer index."""
        if isinstance(coord, str):
            if coord in self.coord_names:
                return self.coord_names.index(coord)
        elif isinstance(coord, int) and not isinstance(coord, bool) and 0 <= coord < self.num_vars:
            return coord
        err_mess = f'Not a valid coordinate name or index: {coord}'
        raise ValueError(err_mess)

    def _monomial_rows(
        self, monomials: Sequence[int] | Sequence[Sequence[int]] | np.ndarray
    ) -> np.ndarray:
        """Validate one monomial or a two-dimensional batch of monomials."""
        arr = np.asarray(monomials)
        if arr.ndim not in (1, 2):
            err_mess = 'Monomials must be a vector or a two-dimensional batch'
            raise ValueError(err_mess)
        if arr.shape[-1] != self.descriptor.monomial_length:
            err_mess = f'Monomials must have length {self.descriptor.monomial_length}'
            raise ValueError(err_mess)
        if not np.issubdtype(arr.dtype, np.integer):
            err_mess = 'Monomial exponents must be integers'
            raise ValueError(err_mess)
        rows = arr.reshape(1, -1) if arr.ndim == 1 else arr
        for row in rows:
            mono = tuple(int(v) for v in row)
            if not self.descriptor.is_valid_monomial(mono):
                err_mess = f'Invalid monomial: {mono}'
                raise ValueError(err_mess)
        return arr

    def coefficient(
        self,
        coord: str | int,
        monomials: Sequence[int] | Sequence[Sequence[int]] | np.ndarray,
    ) -> float | complex | np.ndarray:
        """Return coefficient(s) from one output series by coordinate name or index."""
        idx = self._coordinate_index(coord)
        arr = self._monomial_rows(monomials)
        return self.coords[idx].coefficient(arr)

    def set_coefficient(
        self,
        coord: str | int,
        monomial: Sequence[int] | np.ndarray,
        value: Scalar,
    ) -> None:
        """Set a coefficient of an output series, preserving its real/complex type."""
        idx = self._coordinate_index(coord)
        arr = self._monomial_rows(monomial)
        if arr.ndim != 1:
            err_mess = 'Exactly one monomial must be supplied'
            raise ValueError(err_mess)
        mono = tuple(int(v) for v in arr)
        component = self.coords[idx]
        if isinstance(component, Tpsa):
            converted = complex(value)
            if converted.imag != 0:
                err_mess = 'Cannot assign a complex coefficient to a real TPSA map'
                raise TypeError(err_mess)
            component.set(mono, converted.real)
        elif isinstance(component, ComplexTpsa):
            component.set(mono, value)
        else:
            err_mess = f'Unsupported TPSA component type: {type(component).__name__}'
            raise TypeError(err_mess)

    def jacobian_operator(
        self, *, to_variables: bool = True, to_parameters: bool = False
    ) -> TpsaTensor[TpsaSeriesT]:
        """Return the rank-two TPSA-valued Jacobian over selected inputs."""
        return self.derivative_tensor(1, to_variables=to_variables, to_parameters=to_parameters)

    def jacobian(
        self,
        coordinates: Sequence[Scalar] | np.ndarray | None = None,
        *,
        parameters: Sequence[Scalar] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the Jacobian with respect to descriptor variables.

        Parameters
        ----------
        coordinates
            Numerical coordinates at which to evaluate the Jacobian. Default 0.
        parameters
            Numerical values for the descriptor parameters. Default 0.
        """
        if coordinates is None and parameters is None:
            return np.asarray([tpsa.grad() for tpsa in self.coords])
        if coordinates is None:
            coordinates = [0.0] * self.num_vars
        if parameters is None:
            parameters = [0.0] * self.num_params
        return self.jacobian_operator(to_variables=True, to_parameters=False).evaluate(
            coordinates=coordinates, parameters=parameters
        )

    def param_jacobian(
        self,
        coordinates: Sequence[Scalar] | np.ndarray | None = None,
        *,
        parameters: Sequence[Scalar] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the Jacobian with respect to descriptor parameters.

        Parameters
        ----------
        coordinates
            Numerical coordinates at which to evaluate the parameter Jacobian. Default 0.
        parameters
            Numerical values for the descriptor parameters. Default 0.
        """
        if self.num_params == 0:
            dtype = complex if self.is_complex else float
            return np.empty((len(self), 0), dtype=dtype)
        if coordinates is None and parameters is None:
            return np.asarray([tpsa.param_grad() for tpsa in self.coords])
        if coordinates is None:
            coordinates = [0.0] * self.num_vars
        if parameters is None:
            parameters = [0.0] * self.num_params
        return self.jacobian_operator(to_variables=False, to_parameters=True).evaluate(
            coordinates=coordinates, parameters=parameters
        )

    def set_jacobian(self, jacobian: Sequence[Sequence[Scalar]] | np.ndarray) -> None:
        """Replace all first-order coefficients for descriptor variables."""
        matrix = np.asarray(jacobian)
        expected_shape = (len(self), self.num_vars)
        if matrix.shape != expected_shape:
            err_mess = f'Jacobian must have shape {expected_shape}, got {matrix.shape}'
            raise ValueError(err_mess)
        if not self.is_complex and np.any(np.imag(matrix) != 0):
            err_mess = 'Cannot assign complex coefficients to a real map'
            raise TypeError(err_mess)
        for row_idx, row in enumerate(matrix):
            for variable, value in enumerate(row):
                mono = [0] * self.descriptor.monomial_length
                mono[variable] = 1
                self.set_coefficient(row_idx, mono, value)

    def set_param_jacobian(self, jacobian: Sequence[Sequence[Scalar]] | np.ndarray) -> None:
        """Replace all first-order coefficients for descriptor parameters."""
        matrix = np.asarray(jacobian)
        expected_shape = (len(self), self.num_params)
        if matrix.shape != expected_shape:
            err_mess = f'Parameter Jacobian must have shape {expected_shape}, got {matrix.shape}'
            raise ValueError(err_mess)
        if not self.is_complex and np.any(np.imag(matrix) != 0):
            err_mess = 'Cannot assign complex coefficients to a real map'
            raise TypeError(err_mess)
        for row_idx, row in enumerate(matrix):
            for parameter, value in enumerate(row):
                mono = [0] * self.descriptor.monomial_length
                mono[self.num_vars + parameter] = 1
                self.set_coefficient(row_idx, mono, value)

    def hessian_operator(
        self, *, to_variables: bool = True, to_parameters: bool = False
    ) -> TpsaTensor[TpsaSeriesT]:
        """Return the rank-three TPSA-valued Hessian over selected inputs.

        Entry ``[i, j, k]`` is the second derivative of output ``i`` with
        respect to inputs ``j`` and ``k`` (variables followed by parameters).
        """
        return self.derivative_tensor(2, to_variables=to_variables, to_parameters=to_parameters)

    def hessian(
        self,
        coordinates: Sequence[Scalar] | np.ndarray | None = None,
        *,
        parameters: Sequence[Scalar] | np.ndarray | None = None,
        to_variables: bool = True,
        to_parameters: bool = False,
    ) -> np.ndarray:
        """Evaluate the map Hessian at a numerical phase-space/parameter point.

        By default evaluate at the origin (all coordinates and parameters zero).
        The result has shape ``(num_outputs, num_inputs, num_inputs)``. Each
        input axis follows descriptor variable order and then parameter order
        when ``to_parameters=True``. No setter is provided.
        """
        if coordinates is None:
            coordinates = [0.0] * self.num_vars
        return self.hessian_operator(
            to_variables=to_variables, to_parameters=to_parameters
        ).evaluate(coordinates, parameters=parameters)

    def compose(self, other: TpsaMap[MapSeriesT]) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Compose full coordinate maps, preserving the TpsaMap result type."""
        return cast('TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]', super().compose(other))

    __matmul__ = compose

    def inverse(self) -> TpsaMap[TpsaSeriesT]:
        """Return the formal inverse map about a zero-constant expansion point.

        Descriptor parameters are retained as identity substitutions.
        The linear map must be invertible.
        """
        if np.any(self.const_part != 0):
            err_mess = 'Map inversion requires zero constant terms; recenter it first'
            raise ValueError(err_mess)
        if self.is_complex:
            result = [
                self.descriptor.complex_zero(order=self.descriptor.order)
                for _ in range(self.num_vars)
            ]
            target_params = [ComplexTpsa.from_tpsa(t) for t in self.descriptor.params()]
            protected_func = lib.madng_ctpsa_protected_map_inverse_call
            func = lib.mad_ctpsa_minv
        else:
            result = [
                self.descriptor.zero(order=self.descriptor.order) for _ in range(self.num_vars)
            ]
            target_params = list(self.descriptor.params())
            protected_func = lib.madng_tpsa_protected_map_inverse_call
            func = lib.mad_tpsa_minv
        output = (*result, *target_params)
        status = protected_func(
            func,
            self.num_vars + self.num_params,
            self.pointer_array_with_params,
            self.num_vars,
            ffi.new('void*[]', [t.ptr for t in output]),
        )
        if status:
            self.flat[0]._raise_mad_error()
        return cast('TpsaMap[TpsaSeriesT]', type(self)(result))

    def partial_inverse(self, select: Sequence[SelectionValue]) -> TpsaMap[TpsaSeriesT]:
        """Invert selected variables, leaving other variables and parameters intact.

        ``select`` contains one boolean or zero/one integer per variable.
        """
        if np.any(self.const_part != 0):
            err_mess = 'Partial inversion requires zero constant terms; recenter it first'
            raise ValueError(err_mess)
        if len(select) != self.num_vars:
            err_mess = f'Expected {self.num_vars} selection entries, got {len(select)}'
            raise ValueError(err_mess)
        choices: list[int] = []
        for value in select:
            if isinstance(value, (bool, np.bool_)):
                choices.append(int(value))
            elif isinstance(value, Integral):
                if int(value) not in (0, 1):
                    err_mess = 'Selection entries must be 0 or 1'
                    raise ValueError(err_mess)
                choices.append(int(value))
            else:
                err_mess = 'Selection entries must be booleans or integers'
                raise TypeError(err_mess)
        choices.extend([0] * self.num_params)

        if self.is_complex:
            result = [
                self.descriptor.complex_zero(order=self.descriptor.order)
                for _ in range(self.num_vars)
            ]
            target_params = [ComplexTpsa.from_tpsa(t) for t in self.descriptor.params()]
            protected_func = lib.madng_ctpsa_protected_map_partial_inverse_call
            func = lib.mad_ctpsa_pminv
        else:
            result = [
                self.descriptor.zero(order=self.descriptor.order) for _ in range(self.num_vars)
            ]
            target_params = list(self.descriptor.params())
            protected_func = lib.madng_tpsa_protected_map_partial_inverse_call
            func = lib.mad_tpsa_pminv
        output = (*result, *target_params)
        status = protected_func(
            func,
            self.num_vars + self.num_params,
            self.pointer_array_with_params,
            self.num_vars,
            ffi.new('void*[]', [t.ptr for t in output]),
            ffi.new('int[]', choices),
        )
        if status:
            self.flat[0]._raise_mad_error()
        return cast('TpsaMap[TpsaSeriesT]', type(self)(result))

    def translate(
        self, offsets: Sequence[Scalar] | np.ndarray
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Return the input-translated map ``M(z + offsets, k)``.

        ``offsets`` has one entry per phase-space variable, in descriptor
        order. Descriptor parameters ``k`` remain unchanged. Real maps are
        promoted to complex when any offset has an imaginary part; the
        original map is never modified.

        For parameter-free descriptors, invoke MAD-NG's native ``translate``.
        Its implementation seeds each substitution with ``setvar``, which
        accepts *only* phase-space variable indices. When descriptor parameters
        are present, use native map composition with their identity TPSAs
        instead; appending zero offsets to the native function is invalid.

        This shifts the map's *arguments*, not its output:
        ``M(z + offsets, k)``, rather than ``M(z, k) + offsets``.
        """
        offsets = tuple(offsets)
        if len(offsets) != self.num_vars:
            err_mess = f'Expected {self.num_vars} offsets, got {len(offsets)}'
            raise ValueError(err_mess)
        if any(not isinstance(value, (SupportsFloat, SupportsComplex)) for value in offsets):
            err_mess = 'Offsets must be real or complex scalars'
            raise TypeError(err_mess)

        complex_needed = self.is_complex or any(complex(value).imag != 0 for value in offsets)

        # Native translate constructs substitutions with mad_*tpsa_setvar.
        # Parameter indices are not legal inputs to setvar, so use the same
        # protected native composition as TpsaTensor.compose for these maps.
        if self.num_params:
            substitutions = [
                self.descriptor.var(
                    index,
                    complex(value) if complex_needed else complex(value).real,
                )
                for index, value in enumerate(offsets, start=1)
            ]
            return self.compose(TpsaMap(substitutions))

        source = self.to_complex() if complex_needed and not self.is_complex else self
        if complex_needed:
            result = [
                source.descriptor.complex_zero(order=source.descriptor.order)
                for _ in range(source.num_vars)
            ]
            translations = ffi.new('double _Complex[]', [complex(v) for v in offsets])
            protected_func = lib.madng_ctpsa_protected_map_translate_call
            func = lib.mad_ctpsa_translate
        else:
            result = [
                source.descriptor.zero(order=source.descriptor.order)
                for _ in range(source.num_vars)
            ]
            translations = ffi.new('double[]', [complex(v).real for v in offsets])
            protected_func = lib.madng_tpsa_protected_map_translate_call
            func = lib.mad_tpsa_translate
        status = protected_func(
            func,
            source.num_vars,
            source.pointer_array,
            len(offsets),
            translations,
            ffi.new('void*[]', [t.ptr for t in result]),
        )
        if status:
            source.flat[0]._raise_mad_error()
        return TpsaMap(result)

    @classmethod
    def from_generator(cls, generator: Tpsa | ComplexTpsa) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Construct the native Hamiltonian field ``-J grad(generator)``.

        A scalar ``f`` in ``exp_poisson(f)`` uses the *negative* of this field.
        """
        if not isinstance(generator, (Tpsa, ComplexTpsa)):
            err_mess = 'Generator must be a Tpsa or ComplexTpsa'
            raise TypeError(err_mess)
        descriptor = generator.descriptor
        if descriptor.num_vars % 2:
            err_mess = 'Hamiltonian fields require an even number of variables'
            raise ValueError(err_mess)
        if isinstance(generator, ComplexTpsa):
            output = [
                descriptor.complex_zero(order=descriptor.order) for _ in range(descriptor.num_vars)
            ]
            protected_func = lib.madng_ctpsa_protected_scalar_to_map_call
            func = lib.mad_ctpsa_vec2fld
        else:
            output = [descriptor.zero(order=descriptor.order) for _ in range(descriptor.num_vars)]
            protected_func = lib.madng_tpsa_protected_scalar_to_map_call
            func = lib.mad_tpsa_vec2fld
        status = protected_func(
            func,
            descriptor.num_vars,
            generator.ptr,
            ffi.new('void*[]', [t.ptr for t in output]),
        )
        if status:
            generator._raise_mad_error()
        return cast('TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]', cls(output))

    def to_generator(
        self, *, validate: bool = True, atol: float = 1e-12, rtol: float = 1e-10
    ) -> Tpsa | ComplexTpsa:
        """Recover a scalar generator from a Hamiltonian vector field.

        Use MAD-NG's native ``fld2vec`` to invert ``from_generator``:
        the map is interpreted as ``F = -J grad(f)``, with canonical
        variable ordering ``(q1, p1, q2, p2, ...)``. The recovered generator
        has its phase-space-independent integration term ``C(parameters)``
        fixed to zero.

        With ``validate=True`` (default), reconstruct the field from the
        candidate generator and compare *all* represented coefficients.
        Raise ``ValueError`` if the field is non-Hamiltonian, or if a
        generator needs an order not representable by this descriptor.
        Use ``validate=False`` only for deliberate projections/truncations.
        """
        if self.num_vars % 2:
            err_mess = 'Hamiltonian fields require an even number of variables'
            raise ValueError(err_mess)
        if atol < 0 or rtol < 0:
            err_mess = 'atol and rtol must be non-negative'
            raise ValueError(err_mess)
        if self.is_complex:
            result = self.descriptor.complex_zero(order=self.descriptor.order)
            protected_func = lib.madng_ctpsa_protected_map_to_scalar_call
            func = lib.mad_ctpsa_fld2vec
        else:
            result = self.descriptor.zero(order=self.descriptor.order)
            protected_func = lib.madng_tpsa_protected_map_to_scalar_call
            func = lib.mad_tpsa_fld2vec
        status = protected_func(
            func,
            self.num_vars,
            self.pointer_array,
            result.ptr,
        )
        if status:
            self.flat[0]._raise_mad_error()

        if self.num_params:
            # Native fld2vec divides each coefficient by its phase-space degree.
            # Parameter-only slots can therefore become NaN (0/0), even though
            # the undetermined integration term C(parameters) must be zero.
            # Rebuild a clean series: assigning zero in place is insufficient,
            # since native setm computes 0 * NaN + 0, which remains NaN.
            if isinstance(result, ComplexTpsa):
                clean = self.descriptor.complex_zero(order=result.truncation_order)
                clean.update(
                    {
                        monomial: coefficient
                        for monomial, coefficient in result.monomial_coeffs(tol=0).items()
                        if any(monomial[: self.num_vars])
                    }
                )
            else:
                clean = self.descriptor.zero(order=result.truncation_order)
                clean.update(
                    {
                        monomial: coefficient
                        for monomial, coefficient in result.monomial_coeffs(tol=0).items()
                        if any(monomial[: self.num_vars])
                    }
                )
            result = clean

        if validate:
            rebuilt = type(self).from_generator(result)
            for name, original, reconstructed in zip(
                self.coord_names, self.flat, rebuilt.flat, strict=True
            ):
                expected = original.monomial_coeffs(tol=0)
                actual = reconstructed.monomial_coeffs(tol=0)
                for monomial in expected.keys() | actual.keys():
                    lhs = expected.get(monomial, 0.0)
                    rhs = actual.get(monomial, 0.0)
                    if not np.isclose(lhs, rhs, atol=atol, rtol=rtol):
                        err_mess = (
                            f'Field is not representable as a Hamiltonian generator '
                            f'at the descriptor truncation order: coordinate {name}, '
                            f'monomial {monomial}, original {lhs}, reconstructed {rhs}. '
                            'The field may be non-integrable or require higher order.'
                        )
                        raise ValueError(err_mess)
        return result

    def lie_bracket(self, other: TpsaMap[MapSeriesT]) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Return the native Lie bracket of two vector fields."""
        if not isinstance(other, TpsaMap):
            err_mess = f'Can only take Lie bracket with another TpsaMap! Got {type(other)}'
            raise TypeError(err_mess)
        left, right, complex_needed = self._align_descriptors(
            self, other, extend_left=True, extend_right=True
        )

        if complex_needed:
            result = [
                left.descriptor.complex_zero(order=left.descriptor.order)
                for _ in range(left.num_vars)
            ]
            protected_func = lib.madng_ctpsa_protected_map_pair_call
            func = lib.mad_ctpsa_liebra
        else:
            result = [
                left.descriptor.zero(order=left.descriptor.order) for _ in range(left.num_vars)
            ]
            protected_func = lib.madng_tpsa_protected_map_pair_call
            func = lib.mad_tpsa_liebra
        status = protected_func(
            func,
            left.num_vars,
            left.pointer_array,
            right.pointer_array,
            ffi.new('void*[]', [t.ptr for t in result]),
        )
        if status:
            self.flat[0]._raise_mad_error()
        return TpsaMap(result)

    def exp_poisson(
        self, generator: Tpsa | ComplexTpsa | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Apply the native Lie exponential to this map.

        Scalar generators use MAD-NG's convention: native field ``-vec2fld(f)``.
        Vector-field generators are passed to the native exponential directly.
        """
        if isinstance(generator, (Tpsa, ComplexTpsa)):
            left, right, complex_needed = self._align_descriptors(
                self, generator, extend_left=True, extend_right=False
            )
            right = -TpsaMap.from_generator(right)
        elif isinstance(generator, TpsaMap):
            left, right, complex_needed = self._align_descriptors(
                self, generator, extend_left=True, extend_right=True
            )
        else:
            err_mess = 'Generator must be a scalar TPSA or TpsaMap field'
            raise TypeError(err_mess)

        if complex_needed:
            result = [
                left.descriptor.complex_zero(order=left.descriptor.order)
                for _ in range(left.num_vars)
            ]
            protected_call = lib.madng_ctpsa_protected_map_binary_call
            func = lib.mad_ctpsa_exppb
        else:
            result = [
                left.descriptor.zero(order=left.descriptor.order) for _ in range(left.num_vars)
            ]
            protected_call = lib.madng_tpsa_protected_map_binary_call
            func = lib.mad_tpsa_exppb
        status = protected_call(
            func,
            left.num_vars,
            right.pointer_array,
            left.num_vars,
            left.pointer_array,
            ffi.new('void*[]', [t.ptr for t in result]),
        )
        if status:
            self.flat[0]._raise_mad_error()
        return TpsaMap(result)

    def log_poisson(
        self, initial_guess: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None = None
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Return a vector field logarithm compatible with ``exp_poisson``."""
        if initial_guess is not None and not isinstance(initial_guess, TpsaMap):
            err_mess = f'Map initial_guess must be a TpsaMap! Got {type(initial_guess)}'
            raise TypeError(err_mess)
        if initial_guess is not None:
            left, right, complex_needed = self._align_descriptors(
                self, initial_guess, extend_left=True, extend_right=True
            )
        else:
            left = self
            right = initial_guess
            complex_needed = left.is_complex

        if complex_needed:
            result = [
                left.descriptor.complex_zero(order=left.descriptor.order)
                for _ in range(self.num_vars)
            ]
            protected_func = lib.madng_ctpsa_protected_map_pair_call
            func = lib.mad_ctpsa_logpb
        else:
            result = [
                left.descriptor.zero(order=left.descriptor.order) for _ in range(self.num_vars)
            ]
            protected_func = lib.madng_tpsa_protected_map_pair_call
            func = lib.mad_tpsa_logpb
        status = protected_func(
            func,
            left.num_vars,
            left.pointer_array,
            ffi.NULL if right is None else right.pointer_array,
            ffi.new('void*[]', [t.ptr for t in result]),
        )
        if status:
            self.flat[0]._raise_mad_error()
        return TpsaMap(result)

    def log_generator(
        self,
        initial_guess: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None = None,
        *,
        validate: bool = True,
        atol: float = 1e-12,
        rtol: float = 1e-10,
    ) -> Tpsa | ComplexTpsa:
        """Return the scalar logarithmic generator using MAD-NG's sign.

        When ``validate=True``, reject non-Hamiltonian logarithmic fields or
        generators not reconstructable at this descriptor's order.
        """
        return -self.log_poisson(initial_guess).to_generator(
            validate=validate, atol=atol, rtol=rtol
        )

    def pullback(self, function: Tpsa | ComplexTpsa) -> Tpsa | ComplexTpsa:
        """Return ``function ∘ self`` (ordinary, not inverse, pullback)."""
        if not isinstance(function, (Tpsa, ComplexTpsa)):
            err_mess = 'Pullback requires a scalar TPSA'
            raise TypeError(err_mess)
        left, right, _ = self._align_descriptors(
            self, function, extend_left=True, extend_right=False
        )
        return TpsaTensor([right]).compose(left).flat[0]

    def norm(self) -> float:
        """Return the native map norm (sum of component coefficient norms)."""
        if self.is_complex:
            return float(lib.mad_ctpsa_mnrm(self.num_vars, self.pointer_array))
        return float(lib.mad_tpsa_mnrm(self.num_vars, self.pointer_array))
