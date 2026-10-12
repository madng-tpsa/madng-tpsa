"""Map operations for MAD-NG TPSA series.

The native MAD-NG library represents a map as an array of TPSA pointers rather
than as a separate C object. This module mirrors that design: the functional API
accepts sequences of :class:`Tpsa` / :class:`ComplexTpsa`, while :class:`TpsaMap`
provides a small Python container around such a sequence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Generic, SupportsComplex, SupportsFloat, cast, overload

import numpy as np

from ._typing import TpsaSeriesT
from .complex_tpsa import ComplexTpsa
from .functions import (
    _as_complex,
    _coerce_pair,
    _is_complex,
    _promote_complex,
    _pullback_impl,
    _series_tuple,
    _set_series_coefficient,
    _set_series_const_part,
    compose,
    evaluate,
    exp_poisson,
    inverse,
    lie_bracket,
    log_generator,
    log_poisson,
    map_order,
    max_nonzero_order,
    norm,
    partial_inverse,
    translate,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ._typing import Scalar, SelectionValue, TpsaSeries
    from .tpsa import Tpsa


class TpsaMap(Generic[TpsaSeriesT]):
    """A vector-valued TPSA map sharing one :class:`Descriptor`.

    ``TpsaMap`` owns only its component TPSA objects. It deliberately contains
    no accelerator-specific reference-particle or tracking state.
    """

    __slots__ = ('coords', 'coord_names')

    coords: list[TpsaSeriesT]
    coord_names: tuple[str, ...]

    @overload
    def __init__(
        self: TpsaMap[Tpsa],
        coords: Sequence[Tpsa] | Mapping[str, Tpsa],
        *,
        coord_names: Sequence[str] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: TpsaMap[ComplexTpsa],
        coords: Sequence[TpsaSeries] | Mapping[str, TpsaSeries],
        *,
        coord_names: Sequence[str] | None = None,
    ) -> None: ...

    def __init__(
        self,
        coords: Sequence[TpsaSeries] | Mapping[str, TpsaSeries],
        *,
        coord_names: Sequence[str] | None = None,
    ) -> None:
        if isinstance(coords, Mapping):
            if coord_names is None:
                coord_names = tuple(coords)
            coords = tuple(coords[name] for name in coord_names)

        series = _series_tuple(coords)
        if _is_complex(series):
            series = _promote_complex(series)
        self.coords = cast('list[TpsaSeriesT]', list(series))

        descriptor = series[0].descriptor
        if len(series) > descriptor.num_vars:
            message = (
                f'Map has {len(series)} components but descriptor has only '
                f'{descriptor.num_vars} variables'
            )
            raise ValueError(message)
        if coord_names is None:
            coord_names = descriptor.var_labels[: len(series)]
        coord_names = tuple(coord_names)
        if len(coord_names) != len(series):
            message = 'coord_names must contain one name per map component'
            raise ValueError(message)
        if len(set(coord_names)) != len(coord_names):
            message = 'coord_names must be unique'
            raise ValueError(message)
        self.coord_names = coord_names

    @classmethod
    def identity(
        cls,
        descriptor,
        *,
        values: Sequence[SupportsFloat] | None = None,
        coord_names: Sequence[str] | None = None,
    ) -> TpsaMap[Tpsa]:
        """Construct the identity TPSA map."""
        # Explicit real Tpsa components
        return TpsaMap(
            descriptor.vars(values),
            coord_names=coord_names,
        )

    @classmethod
    def from_monomial_coeffs(
        cls,
        descriptor,
        coefficients: Sequence[Mapping] | Mapping[str, Mapping],
        *,
        coord_names: Sequence[str] | None = None,
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Construct a TPSA map from monomial coefficient dictionaries."""
        if isinstance(coefficients, Mapping):
            if coord_names is None:
                coord_names = tuple(coefficients)
            coords = [descriptor.from_monomial_coeffs(coefficients[name]) for name in coord_names]
        else:
            coords = [descriptor.from_monomial_coeffs(component) for component in coefficients]
        return TpsaMap(coords, coord_names=coord_names)

    def to_complex(self) -> TpsaMap[ComplexTpsa]:
        """Return this map represented by complex TPSAs."""
        return TpsaMap(
            [_as_complex(value) for value in self.coords],
            coord_names=self.coord_names,
        )

    @property
    def descriptor(self):
        """Return the descriptor shared by all coordinates."""
        return self.coords[0].descriptor

    @property
    def order(self) -> int:
        """Return the maximum allocated order of the map."""
        return map_order(self.coords)

    @property
    def max_nonzero_order(self) -> int:
        """Return the highest populated order of the map."""
        return max_nonzero_order(self.coords)

    @property
    def num_vars(self) -> int:
        """Return the number of descriptor variables."""
        return self.descriptor.num_vars

    @property
    def num_params(self) -> int:
        """Return the number of descriptor parameters."""
        return self.descriptor.num_params

    @property
    def is_complex(self) -> bool:
        """Return whether the map contains complex TPSAs."""
        return any(isinstance(value, ComplexTpsa) for value in self.coords)

    @property
    def const_part(self) -> np.ndarray:
        """Return the constant vector of the map."""
        dtype = complex if self.is_complex else float
        return np.asarray([value.const_part for value in self.coords], dtype=dtype)

    def __len__(self) -> int:
        return len(self.coords)

    def __iter__(self) -> Iterator[TpsaSeriesT]:
        return iter(self.coords)

    def __getitem__(self, key: int | str) -> TpsaSeriesT:
        if isinstance(key, str):
            try:
                key = self.coord_names.index(key)
            except ValueError as exc:
                raise KeyError(key) from exc
        return self.coords[key]

    def __getattr__(self, name: str) -> TpsaSeriesT:
        if name in self.coord_names:
            return self[name]
        raise AttributeError(name)

    def __repr__(self) -> str:
        components = ', '.join(
            f'{name}={value!r}' for name, value in zip(self.coord_names, self.coords, strict=True)
        )
        return f'TpsaMap({components})'

    def copy(self) -> TpsaMap[TpsaSeriesT]:
        """Return an independent copy of the map."""
        result = TpsaMap(
            [value.copy() for value in self.coords],
            coord_names=self.coord_names,
        )
        return cast('TpsaMap[TpsaSeriesT]', result)

    def monomial_coeffs(self, coord: int | str | None = None, tol: float = 1e-14):
        """Return monomial coefficients for one or all coordinates."""
        if coord is not None:
            return self[coord].monomial_coeffs(tol)
        return {
            name: value.monomial_coeffs(tol)
            for name, value in zip(self.coord_names, self.coords, strict=True)
        }

    def coefficient(self, coord: int | str, monomials):
        """Return coefficients from one coordinate."""
        return self[coord].coefficient(monomials)

    def set_coefficient(self, coord: int | str, monomial, value) -> None:
        """Set a coefficient in one coordinate."""
        _set_series_coefficient(self[coord], monomial, value)

    def set_const_part(self, values: Sequence[Scalar]) -> None:
        """Set the constant part of every map component."""
        values = tuple(values)
        if len(values) != len(self):
            message = f'Expected {len(self)} values, got {len(values)}'
            raise ValueError(message)
        for component, value in zip(self.coords, values, strict=True):
            _set_series_const_part(component, value)

    def evaluate(self, coordinates, *, parameters=None) -> np.ndarray:
        """Evaluate the map at numerical coordinates."""
        return evaluate(self.coords, coordinates, parameters=parameters)

    evaluate_array = evaluate

    def jacobian(
        self,
        coordinates: Sequence[Scalar] | None = None,
        *,
        parameters: Sequence[Scalar] | None = None,
    ) -> np.ndarray:
        """Return the Jacobian matrix of the map."""
        if coordinates is None:
            return np.asarray([value.grad() for value in self.coords])

        columns = []
        for variable in range(1, self.descriptor.num_vars + 1):
            derivatives = tuple(value.derivative(variable) for value in self.coords)
            columns.append(evaluate(derivatives, coordinates, parameters=parameters))
        return np.column_stack(columns)

    def param_jacobian(self) -> np.ndarray:
        """Return first-order derivatives with respect to descriptor parameters."""
        return np.asarray([value.param_grad() for value in self.coords])

    def set_jacobian(self, jacobian) -> None:
        """Set the first-order coefficients with respect to descriptor variables."""
        jacobian = np.asarray(jacobian)
        expected_shape = (len(self), self.num_vars)
        if jacobian.shape != expected_shape:
            message = f'Jacobian must have shape {expected_shape}, got {jacobian.shape}'
            raise ValueError(message)
        for component, row in zip(self.coords, jacobian, strict=True):
            for variable, value in enumerate(row):
                monomial = [0] * self.descriptor.monomial_length
                monomial[variable] = 1
                _set_series_coefficient(component, monomial, value)

    def set_param_jacobian(self, jacobian) -> None:
        """Set first-order coefficients with respect to descriptor parameters."""
        jacobian = np.asarray(jacobian)
        expected_shape = (len(self), self.num_params)
        if jacobian.shape != expected_shape:
            message = f'Parameter Jacobian must have shape {expected_shape}, got {jacobian.shape}'
            raise ValueError(message)
        for component, row in zip(self.coords, jacobian, strict=True):
            for parameter, value in enumerate(row):
                monomial = [0] * self.descriptor.monomial_length
                monomial[self.num_vars + parameter] = 1
                _set_series_coefficient(component, monomial, value)

    def sensitivity(self, coord: int | str, parameter: int | str):
        """Return the first-order sensitivity of one output to one parameter."""
        if isinstance(parameter, str):
            try:
                parameter = self.descriptor.param_labels.index(parameter)
            except ValueError as exc:
                raise KeyError(parameter) from exc
        elif not 0 <= parameter < self.num_params:
            raise IndexError(parameter)
        return self[coord].param_grad()[parameter]

    def homogeneous(self, order: int) -> TpsaMap[TpsaSeriesT]:
        """Return the homogeneous part of the requested order."""
        result = TpsaMap(
            [value.homogeneous(order) for value in self.coords],
            coord_names=self.coord_names,
        )
        return cast('TpsaMap[TpsaSeriesT]', result)

    def truncate(self, order: int) -> TpsaMap[TpsaSeriesT]:
        """Return the map truncated through the requested order."""
        result = TpsaMap(
            [value.truncate(order) for value in self.coords],
            coord_names=self.coord_names,
        )
        return cast('TpsaMap[TpsaSeriesT]', result)

    def clear_order(self, order: int) -> TpsaMap[TpsaSeriesT]:
        """Return a copy with one homogeneous order removed."""
        result = TpsaMap(
            [value.clear_order(order) for value in self.coords],
            coord_names=self.coord_names,
        )
        return cast('TpsaMap[TpsaSeriesT]', result)

    @overload
    def compose(self: TpsaMap[Tpsa], other: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    @overload
    def compose(self: TpsaMap[Tpsa], other: TpsaMap[ComplexTpsa]) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def compose(
        self: TpsaMap[ComplexTpsa], other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[ComplexTpsa]: ...

    def compose(
        self, other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Return the composition ``self ∘ other``."""
        if not isinstance(other, TpsaMap):
            return NotImplemented
        return TpsaMap(
            compose(self.coords, other.coords),
            coord_names=self.coord_names,
        )

    __matmul__ = compose

    def inverse(self) -> TpsaMap[TpsaSeriesT]:
        """Return the formal inverse of the map."""
        if np.any(self.const_part != 0):
            err_mess = (
                'Map inversion requires a zero constant part; '
                'translate/recentre the map before inversion'
            )
            raise ValueError(err_mess)
        result = TpsaMap(
            inverse(self.coords),
            coord_names=self.coord_names,
        )
        return cast('TpsaMap[TpsaSeriesT]', result)

    def partial_inverse(self, select: Sequence[SelectionValue]) -> TpsaMap[TpsaSeriesT]:
        """Return a partial inverse of the map."""
        if np.any(self.const_part != 0):
            err_mess = (
                'Map inversion requires a zero constant part; '
                'translate/recentre the map before inversion'
            )
            raise ValueError(err_mess)
        result = TpsaMap(
            partial_inverse(self.coords, select),
            coord_names=self.coord_names,
        )
        return cast('TpsaMap[TpsaSeriesT]', result)

    @overload
    def translate(self: TpsaMap[Tpsa], offsets: Sequence[SupportsFloat]) -> TpsaMap[Tpsa]: ...
    @overload
    def translate(
        self: TpsaMap[Tpsa], offsets: Sequence[SupportsComplex]
    ) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def translate(
        self: TpsaMap[ComplexTpsa], offsets: Sequence[Scalar]
    ) -> TpsaMap[ComplexTpsa]: ...

    def translate(self, offsets: Sequence[Scalar]) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Translate the map arguments."""
        return TpsaMap(
            translate(self.coords, offsets),
            coord_names=self.coord_names,
        )

    @overload
    def lie_bracket(self: TpsaMap[Tpsa], other: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    @overload
    def lie_bracket(self: TpsaMap[Tpsa], other: TpsaMap[ComplexTpsa]) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def lie_bracket(
        self: TpsaMap[ComplexTpsa], other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[ComplexTpsa]: ...

    def lie_bracket(
        self, other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Return the Lie bracket with another vector field."""
        if not isinstance(other, TpsaMap):
            return NotImplemented
        return TpsaMap(
            lie_bracket(self.coords, other.coords),
            coord_names=self.coord_names,
        )

    @overload
    def exp_poisson(self: TpsaMap[Tpsa], generator: Tpsa) -> TpsaMap[Tpsa]: ...
    @overload
    def exp_poisson(self: TpsaMap[Tpsa], generator: ComplexTpsa) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def exp_poisson(self: TpsaMap[ComplexTpsa], generator: TpsaSeries) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def exp_poisson(self: TpsaMap[Tpsa], generator: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    @overload
    def exp_poisson(
        self: TpsaMap[Tpsa], generator: TpsaMap[ComplexTpsa]
    ) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def exp_poisson(
        self: TpsaMap[ComplexTpsa], generator: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[ComplexTpsa]: ...

    def exp_poisson(
        self,
        generator: TpsaSeries | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Return the action of the Lie exponential generated by ``generator``.

        In mathematical notation, ``map.exp_poisson(f)`` returns
            exp(:f:) map
        following the same argument ordering as MAD-NG ``damap:exppb(f)``.
        """
        generator_arg: TpsaSeries | Sequence[TpsaSeries]
        generator_arg = generator.coords if isinstance(generator, TpsaMap) else generator
        return TpsaMap(
            exp_poisson(self.coords, generator_arg),
            coord_names=self.coord_names,
        )

    @overload
    def log_poisson(self: TpsaMap[Tpsa], initial_guess: None = None) -> TpsaMap[Tpsa]: ...
    @overload
    def log_poisson(self: TpsaMap[Tpsa], initial_guess: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    @overload
    def log_poisson(
        self: TpsaMap[Tpsa], initial_guess: TpsaMap[ComplexTpsa]
    ) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def log_poisson(
        self: TpsaMap[ComplexTpsa],
        initial_guess: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None = None,
    ) -> TpsaMap[ComplexTpsa]: ...

    def log_poisson(
        self, initial_guess: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None = None
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Return the logarithmic Hamiltonian vector field."""
        initial_coords = None if initial_guess is None else initial_guess.coords
        return TpsaMap(
            log_poisson(self.coords, initial_coords),
            coord_names=self.coord_names,
        )

    @overload
    def log_generator(self: TpsaMap[Tpsa], initial_guess: None = None) -> Tpsa: ...
    @overload
    def log_generator(self: TpsaMap[Tpsa], initial_guess: TpsaMap[Tpsa]) -> Tpsa: ...
    @overload
    def log_generator(self: TpsaMap[Tpsa], initial_guess: TpsaMap[ComplexTpsa]) -> ComplexTpsa: ...
    @overload
    def log_generator(
        self: TpsaMap[ComplexTpsa],
        initial_guess: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None = None,
    ) -> ComplexTpsa: ...

    def log_generator(
        self, initial_guess: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None = None
    ) -> TpsaSeries:
        """Return the scalar logarithmic generator."""
        initial_coords = None if initial_guess is None else initial_guess.coords
        return log_generator(self.coords, initial_coords)

    @overload
    def pullback(self: TpsaMap[Tpsa], function: Tpsa) -> Tpsa: ...
    @overload
    def pullback(self: TpsaMap[Tpsa], function: ComplexTpsa) -> ComplexTpsa: ...
    @overload
    def pullback(self: TpsaMap[ComplexTpsa], function: TpsaSeries) -> ComplexTpsa: ...

    def pullback(self, function: TpsaSeries) -> TpsaSeries:
        """Return ``function ∘ self``."""
        map_ = cast(
            'TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]',
            self,
        )
        return _pullback_impl(function, map_)

    def norm(self) -> float:
        """Return the MAD-NG map norm."""
        return norm(self.coords)

    def __neg__(self) -> TpsaMap[TpsaSeriesT]:
        result = TpsaMap(
            [-value for value in self.coords],
            coord_names=self.coord_names,
        )
        return cast('TpsaMap[TpsaSeriesT]', result)

    @overload
    def __add__(self: TpsaMap[Tpsa], other: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    @overload
    def __add__(self: TpsaMap[Tpsa], other: TpsaMap[ComplexTpsa]) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def __add__(
        self: TpsaMap[ComplexTpsa], other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[ComplexTpsa]: ...

    def __add__(
        self, other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        if not isinstance(other, TpsaMap):
            return NotImplemented
        left, right = _coerce_pair(self.coords, other.coords)
        if len(left) != len(right):
            message = 'Maps must have the same number of components'
            raise ValueError(message)
        return TpsaMap(
            [a + b for a, b in zip(left, right, strict=True)],
            coord_names=self.coord_names,
        )

    @overload
    def __sub__(self: TpsaMap[Tpsa], other: TpsaMap[Tpsa]) -> TpsaMap[Tpsa]: ...
    @overload
    def __sub__(self: TpsaMap[Tpsa], other: TpsaMap[ComplexTpsa]) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def __sub__(
        self: TpsaMap[ComplexTpsa], other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[ComplexTpsa]: ...

    def __sub__(
        self, other: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        if not isinstance(other, TpsaMap):
            return NotImplemented
        left, right = _coerce_pair(self.coords, other.coords)
        if len(left) != len(right):
            message = 'Maps must have the same number of components'
            raise ValueError(message)
        return TpsaMap(
            [a - b for a, b in zip(left, right, strict=True)],
            coord_names=self.coord_names,
        )

    @overload
    def __mul__(self: TpsaMap[Tpsa], scalar: SupportsFloat) -> TpsaMap[Tpsa]: ...
    @overload
    def __mul__(self: TpsaMap[Tpsa], scalar: SupportsComplex) -> TpsaMap[ComplexTpsa]: ...
    @overload
    def __mul__(self: TpsaMap[ComplexTpsa], scalar: Scalar) -> TpsaMap[ComplexTpsa]: ...

    def __mul__(self, scalar: Scalar) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        if not isinstance(scalar, (SupportsFloat, SupportsComplex)):
            return NotImplemented
        return TpsaMap(
            [scalar * value for value in self.coords],
            coord_names=self.coord_names,
        )

    __rmul__ = __mul__
