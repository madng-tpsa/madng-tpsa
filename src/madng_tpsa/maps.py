"""Map operations for MAD-NG TPSA series.

The native MAD-NG library represents a map as an array of TPSA pointers rather
than as a separate C object. This module provides a small Python container
around such an array, and adds map operations as methods to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, SupportsComplex, SupportsFloat, cast, overload

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Self

    from ._typing import Scalar, TpsaSeries
    from .tpsa import Tpsa

from collections.abc import Sequence

from ._cffi import ffi
from ._typing import TpsaSeriesT
from .complex_tpsa import ComplexTpsa
from .descriptor import Descriptor
from .tensors import TpsaTensor
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
                err_message = 'Cannot mix positional components with named coordinates'
                raise TypeError(err_message)
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
            message = 'TpsaMap requires a complete coordinate index'
            raise IndexError(message)
        return result

    def to_descriptor(
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
            err_mess += 'Use extend_to() to add new coordinates.'
            raise ValueError(err_mess)
        converted = {
            rename.get(name, name): component.to_descriptor(
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

    def extend_to(self, descriptor: Descriptor, *, merge: bool = False) -> Self:
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
            name: component.to_descriptor(descriptor)
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
            coord_names = object.__getattribute__(self, '_coord_names')
        except AttributeError:
            raise AttributeError(name) from None
        try:
            flat = object.__getattribute__(self, 'flat')
        except AttributeError:
            raise AttributeError(name) from None
        try:
            dct = dict(zip(coord_names, flat, strict=True))
            return dct[name]
        except KeyError:
            raise AttributeError(name) from None

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
    def coordinates(self) -> dict[str, TpsaSeriesT]:
        """Dict of oordinate names -> TPSAs of the map."""
        return dict(zip(self._coord_names, self.flat, strict=True))

    @property
    def pointer_array_with_params(self) -> ffi.CData:
        """Return a full C array of pointers to the map's TPSAs, including parameters."""
        return ffi.new('void*[]', [t.ptr for t in (*self.flat, *self._params)])

    # def jacobian_operator(self) -> TpsaTensor[TpsaSeriesT]:
    #     """Return the full TPSA-valued Jacobian."""
    #     return TpsaTensor(
    #         [
    #             coord.derivative(j)
    #             for coord in self.coords
    #             for j in range(1, self.num_vars + 1)
    #         ],
    #         shape=(len(self), self.num_vars),
    #     )

    # def jacobian(
    #     self,
    #     coordinates: Sequence[Scalar] | None = None,
    #     *,
    #     parameters: Sequence[Scalar] | None = None,
    # ) -> np.ndarray:
    #     """Return the Jacobian matrix of the map.

    #     Parameters
    #     ----------
    #     coordinates
    #         Numerical coordinates at which to evaluate the Jacobian.
    #     parameters
    #         Numerical values for the descriptor parameters.
    #     """
    #     if coordinates is None and parameters is None:
    #         return np.asarray([tpsa.grad() for tpsa in self])
    #     if coordinates is None:
    #         coordinates = coordinates or [0] * self.num_vars
    #     if parameters is None:
    #         parameters = parameters or [0] * self.num_params
    #     columns = []
    #     for variable in range(1, self.num_vars + 1):
    #         this_map = TpsaMap(**{
    #             coord: tpsa.derivative(variable) for coord, tpsa in self.items()
    #         })
    #         columns.append(this_map.evaluate(coordinates, parameters=parameters))
    #     return np.column_stack(columns)

    # def param_jacobian(
    #     self,
    #     coordinates: Sequence[Scalar] | None = None,
    #     *,
    #     parameters: Sequence[Scalar] | None = None,
    # ) -> np.ndarray:
    #     """Return first-order derivatives with respect to descriptor parameters.

    #     Parameters
    #     ----------
    #     coordinates
    #         Numerical coordinates at which to evaluate the parameter Jacobian.
    #     parameters
    #         Numerical values for the descriptor parameters.
    #     """
    #     if coordinates is None and parameters is None:
    #         return np.asarray([tpsa.param_grad() for tpsa in self])
    #     if coordinates is None:
    #         coordinates = coordinates or [0] * self.num_vars
    #     if parameters is None:
    #         parameters = parameters or [0] * self.num_params
    #     columns = []
    #     for param_idx in range(1, self.num_params + 1):
    #         this_map = TpsaMap(**{
    #             coord: tpsa.derivative(self.num_vars + param_idx)
    #             for coord, tpsa in self.items()
    #         })
    #         columns.append(this_map.evaluate(coordinates, parameters=parameters))
    #     return np.column_stack(columns)

    # def set_jacobian(self, jacobian) -> None:
    #     """Set the first-order coefficients with respect to descriptor variables."""
    #     jacobian = np.asarray(jacobian)
    #     expected_shape = (len(self), self.num_vars)
    #     if jacobian.shape != expected_shape:
    #         message = f'Jacobian must have shape {expected_shape}, got {jacobian.shape}'
    #         raise ValueError(message)
    #     for component, row in zip(self.values(), jacobian, strict=True):
    #         for variable, value in enumerate(row):
    #             monomial = [0] * self.descriptor.monomial_length
    #             monomial[variable] = 1
    #             self.set_coefficient(component, monomial, value)

    # def set_param_jacobian(self, jacobian) -> None:
    #     """Set first-order coefficients with respect to descriptor parameters."""
    #     jacobian = np.asarray(jacobian)
    #     expected_shape = (len(self), self.num_params)
    #     if jacobian.shape != expected_shape:
    #         message = f'Parameter Jacobian must have shape {expected_shape}, got {jacobian.shape}'
    #         raise ValueError(message)
    #     for component, row in zip(self.values(), jacobian, strict=True):
    #         for parameter, value in enumerate(row):
    #             monomial = [0] * self.descriptor.monomial_length
    #             monomial[self.num_vars + parameter] = 1
    #             self.set_coefficient(component, monomial, value)

    # def homogeneous(self, degree: int) -> TpsaMap[TpsaSeriesT]:
    #     """Return the homogeneous part of the requested degree."""
    #     result = TpsaMap(**{coord: tpsa.homogeneous(degree) for coord, tpsa in self.items()})
    #     return cast('TpsaMap[TpsaSeriesT]', result)

    # def truncate(self, degree: int) -> TpsaMap[TpsaSeriesT]:
    #     """Return the map truncated through the requested degree."""
    #     result = TpsaMap(**{coord: tpsa.truncate(degree) for coord, tpsa in self.items()})
    #     return cast('TpsaMap[TpsaSeriesT]', result)

    # def clear_order(self, degree: int) -> TpsaMap[TpsaSeriesT]:
    #     """Return a copy with one homogeneous degree removed."""
    #     result = TpsaMap(**{coord: tpsa.clear_order(degree) for coord, tpsa in self.items()})
    #     return cast('TpsaMap[TpsaSeriesT]', result)

    # @overload
    # def compose(self: TpsaMap[Tpsa], other: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    # @overload
    # def compose(self: TpsaMap[Tpsa], other: TpsaMap[ComplexTpsa]) -> TpsaMap[ComplexTpsa]: ...
    # @overload
    # def compose(
    #     self: TpsaMap[ComplexTpsa], other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    # ) -> TpsaMap[ComplexTpsa]: ...

    # def compose(
    #     self, other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    # ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
    #     """Return the composition ``self ∘ other``."""
    #     if not isinstance(other, TpsaMap):
    #         return NotImplemented
    #     return TpsaMap(
    #         compose(self.coords, other.coords),
    #         coord_names=self.coord_names,
    #     )

    # __matmul__ = compose

    # def inverse(self) -> TpsaMap[TpsaSeriesT]:
    #     """Return the formal inverse of the map."""
    #     if np.any(self.const_part != 0):
    #         err_mess = (
    #             'Map inversion requires a zero constant part; '
    #             'translate/recentre the map before inversion'
    #         )
    #         raise ValueError(err_mess)
    #     result = TpsaMap(
    #         inverse(self.coords),
    #         coord_names=self.coord_names,
    #     )
    #     return cast('TpsaMap[TpsaSeriesT]', result)

    # def partial_inverse(self, select: Sequence[SelectionValue]) -> TpsaMap[TpsaSeriesT]:
    #     """Return a partial inverse of the map."""
    #     if np.any(self.const_part != 0):
    #         err_mess = (
    #             'Map inversion requires a zero constant part; '
    #             'translate/recentre the map before inversion'
    #         )
    #         raise ValueError(err_mess)
    #     result = TpsaMap(
    #         partial_inverse(self.coords, select),
    #         coord_names=self.coord_names,
    #     )
    #     return cast('TpsaMap[TpsaSeriesT]', result)

    # @overload
    # def translate(self: TpsaMap[Tpsa], offsets: Sequence[SupportsFloat]) -> TpsaMap[Tpsa]: ...
    # @overload
    # def translate(
    #     self: TpsaMap[Tpsa], offsets: Sequence[SupportsComplex]
    # ) -> TpsaMap[ComplexTpsa]: ...
    # @overload
    # def translate(
    #     self: TpsaMap[ComplexTpsa], offsets: Sequence[Scalar]
    # ) -> TpsaMap[ComplexTpsa]: ...

    # def translate(self, offsets: Sequence[Scalar]) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
    #     """Translate the map arguments."""
    #     return TpsaMap(
    #         translate(self.coords, offsets),
    #         coord_names=self.coord_names,
    #     )

    # @overload
    # def lie_bracket(self: TpsaMap[Tpsa], other: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    # @overload
    # def lie_bracket(self: TpsaMap[Tpsa], other: TpsaMap[ComplexTpsa]) -> TpsaMap[ComplexTpsa]: ...
    # @overload
    # def lie_bracket(
    #     self: TpsaMap[ComplexTpsa], other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    # ) -> TpsaMap[ComplexTpsa]: ...

    # def lie_bracket(
    #     self, other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    # ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
    #     """Return the Lie bracket with another vector field."""
    #     if not isinstance(other, TpsaMap):
    #         return NotImplemented
    #     return TpsaMap(
    #         lie_bracket(self.coords, other.coords),
    #         coord_names=self.coord_names,
    #     )

    # @overload
    # def exp_poisson(self: TpsaMap[Tpsa], generator: Tpsa) -> TpsaMap[Tpsa]: ...
    # @overload
    # def exp_poisson(self: TpsaMap[Tpsa], generator: ComplexTpsa) -> TpsaMap[ComplexTpsa]: ...
    # @overload
    # def exp_poisson(
    #     self: TpsaMap[ComplexTpsa], generator: TpsaSeries
    # ) -> TpsaMap[ComplexTpsa]: ...
    # @overload
    # def exp_poisson(self: TpsaMap[Tpsa], generator: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    # @overload
    # def exp_poisson(
    #     self: TpsaMap[Tpsa], generator: TpsaMap[ComplexTpsa]
    # ) -> TpsaMap[ComplexTpsa]: ...
    # @overload
    # def exp_poisson(
    #     self: TpsaMap[ComplexTpsa], generator: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    # ) -> TpsaMap[ComplexTpsa]: ...

    # def exp_poisson(
    #     self,
    #     generator: TpsaSeries | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
    # ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
    #     """Return the action of the Lie exponential generated by ``generator``.

    #     In mathematical notation, ``map.exp_poisson(f)`` returns
    #         exp(:f:) map
    #     following the same argument ordering as MAD-NG ``damap:exppb(f)``.
    #     """
    #     generator_arg: TpsaSeries | Sequence[TpsaSeries]
    #     generator_arg = generator.coords if isinstance(generator, TpsaMap) else generator
    #     return TpsaMap(
    #         exp_poisson(self.coords, generator_arg),
    #         coord_names=self.coord_names,
    #     )

    # @overload
    # def log_poisson(self: TpsaMap[Tpsa], initial_guess: None = None) -> TpsaMap[Tpsa]: ...
    # @overload
    # def log_poisson(self: TpsaMap[Tpsa], initial_guess: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    # @overload
    # def log_poisson(
    #     self: TpsaMap[Tpsa], initial_guess: TpsaMap[ComplexTpsa]
    # ) -> TpsaMap[ComplexTpsa]: ...
    # @overload
    # def log_poisson(
    #     self: TpsaMap[ComplexTpsa],
    #     initial_guess: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None = None,
    # ) -> TpsaMap[ComplexTpsa]: ...

    # def log_poisson(
    #     self, initial_guess: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None = None
    # ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
    #     """Return the logarithmic Hamiltonian vector field."""
    #     initial_coords = None if initial_guess is None else initial_guess.coords
    #     return TpsaMap(
    #         log_poisson(self.coords, initial_coords),
    #         coord_names=self.coord_names,
    #     )

    # @overload
    # def log_generator(self: TpsaMap[Tpsa], initial_guess: None = None) -> Tpsa: ...
    # @overload
    # def log_generator(self: TpsaMap[Tpsa], initial_guess: TpsaMap[Tpsa]) -> Tpsa: ...
    # @overload
    # def log_generator(
    #     self: TpsaMap[Tpsa], initial_guess: TpsaMap[ComplexTpsa]
    # ) -> ComplexTpsa: ...
    # @overload
    # def log_generator(
    #     self: TpsaMap[ComplexTpsa],
    #     initial_guess: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None = None,
    # ) -> ComplexTpsa: ...

    # def log_generator(
    #     self, initial_guess: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None = None
    # ) -> TpsaSeries:
    #     """Return the scalar logarithmic generator."""
    #     initial_coords = None if initial_guess is None else initial_guess.coords
    #     return log_generator(self.coords, initial_coords)

    # @overload
    # def pullback(self: TpsaMap[Tpsa], function: Tpsa) -> Tpsa: ...
    # @overload
    # def pullback(self: TpsaMap[Tpsa], function: ComplexTpsa) -> ComplexTpsa: ...
    # @overload
    # def pullback(self: TpsaMap[ComplexTpsa], function: TpsaSeries) -> ComplexTpsa: ...

    # def pullback(self, function: TpsaSeries) -> TpsaSeries:
    #     """Return ``function ∘ self``."""
    #     map_ = cast(
    #         'TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]',
    #         self,
    #     )
    #     return _pullback_impl(function, map_)

    # def norm(self) -> float:
    #     """Return the MAD-NG map norm."""
    #     return norm(self.coords)

    # def __neg__(self) -> TpsaMap[TpsaSeriesT]:
    #     result = TpsaMap(
    #         [-value for value in self.coords],
    #         coord_names=self.coord_names,
    #     )
    #     return cast('TpsaMap[TpsaSeriesT]', result)

    # @overload
    # def __add__(self: TpsaMap[Tpsa], other: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    # @overload
    # def __add__(self: TpsaMap[Tpsa], other: TpsaMap[ComplexTpsa]) -> TpsaMap[ComplexTpsa]: ...
    # @overload
    # def __add__(
    #     self: TpsaMap[ComplexTpsa], other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    # ) -> TpsaMap[ComplexTpsa]: ...

    # def __add__(
    #     self, other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    # ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
    #     if not isinstance(other, TpsaMap):
    #         return NotImplemented
    #     left, right = _coerce_pair(self.coords, other.coords)
    #     if len(left) != len(right):
    #         message = 'Maps must have the same number of components'
    #         raise ValueError(message)
    #     return TpsaMap(
    #         [a + b for a, b in zip(left, right, strict=True)],
    #         coord_names=self.coord_names,
    #     )

    # @overload
    # def __sub__(self: TpsaMap[Tpsa], other: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    # @overload
    # def __sub__(self: TpsaMap[Tpsa], other: TpsaMap[ComplexTpsa]) -> TpsaMap[ComplexTpsa]: ...
    # @overload
    # def __sub__(
    #     self: TpsaMap[ComplexTpsa], other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    # ) -> TpsaMap[ComplexTpsa]: ...

    # def __sub__(
    #     self, other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    # ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
    #     if not isinstance(other, TpsaMap):
    #         return NotImplemented
    #     left, right = _coerce_pair(self.coords, other.coords)
    #     if len(left) != len(right):
    #         message = 'Maps must have the same number of components'
    #         raise ValueError(message)
    #     return TpsaMap(
    #         [a - b for a, b in zip(left, right, strict=True)],
    #         coord_names=self.coord_names,
    #     )

    # @overload
    # def __mul__(self: TpsaMap[Tpsa], scalar: SupportsFloat) -> TpsaMap[Tpsa]: ...
    # @overload
    # def __mul__(self: TpsaMap[Tpsa], scalar: SupportsComplex) -> TpsaMap[ComplexTpsa]: ...
    # @overload
    # def __mul__(self: TpsaMap[ComplexTpsa], scalar: Scalar) -> TpsaMap[ComplexTpsa]: ...

    # def __mul__(self, scalar: Scalar) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
    #     if not isinstance(scalar, (SupportsFloat, SupportsComplex)):
    #         return NotImplemented
    #     return TpsaMap(
    #         [scalar * value for value in self.coords],
    #         coord_names=self.coord_names,
    #     )

    # __rmul__ = __mul__
