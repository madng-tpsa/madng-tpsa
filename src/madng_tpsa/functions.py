"""Function wrappers for C functions."""

from __future__ import annotations

# from numbers import Integral
from typing import TYPE_CHECKING  # , overload

# import numpy as np
from ._cffi import ffi, lib
from .errors import TpsaError

if TYPE_CHECKING:
    # from collections.abc import Sequence
    from typing import Any  # , SupportsComplex, SupportsFloat

    # from ._typing import Scalar, SelectionValue, TpsaSeries
    # from .complex_tpsa import ComplexTpsa
    # from .maps import TpsaMap
    # from .tpsa import Tpsa


def _raise_mad_error() -> None:
    location = ffi.string(lib.madng_tpsa_last_error_location()).decode()
    message = ffi.string(lib.madng_tpsa_last_error_message()).decode()
    if location:
        raise TpsaError(f'GTPSA error in {location}: {message}')
    raise TpsaError(f'GTPSA error: {message}')


def _protected_call(family: str, function: Any, *, is_complex: bool, args: Any) -> None:
    kind = 'ctpsa' if is_complex else 'tpsa'
    wrapper = getattr(
        lib,
        f'madng_{kind}_protected_{family}_call',
    )
    status = wrapper(function, *args)
    if status:
        _raise_mad_error()


# def _series_tuple(values: Sequence[TpsaSeries] | TpsaMap) -> tuple[TpsaSeries, ...]:
#     from .complex_tpsa import ComplexTpsa
#     from .maps import TpsaMap
#     from .tpsa import Tpsa

#     if isinstance(values, TpsaMap):
#         values = values.coords
#     result = tuple(values)
#     if not result:
#         message = 'TPSA map cannot be empty'
#         raise ValueError(message)
#     for value in result:
#         if not isinstance(value, (Tpsa, ComplexTpsa)):
#             message = 'Map components must be Tpsa or ComplexTpsa'
#             raise TypeError(message)
#     descriptor = result[0].descriptor
#     if any(value.descriptor is not descriptor for value in result[1:]):
#         message = 'All map components must share the same Descriptor'
#         raise ValueError(message)
#     return result


# def _is_complex(values: Sequence[TpsaSeries]) -> bool:
#     from .complex_tpsa import ComplexTpsa

#     return any(isinstance(value, ComplexTpsa) for value in values)


# def _as_complex(value: TpsaSeries) -> ComplexTpsa:
#     from .complex_tpsa import ComplexTpsa

#     if isinstance(value, ComplexTpsa):
#         return value.copy()
#     return ComplexTpsa.from_tpsa(value)


# def _promote_complex(values: Sequence[TpsaSeries]) -> tuple[ComplexTpsa, ...]:
#     from .complex_tpsa import ComplexTpsa

#     return tuple(
#         value if isinstance(value, ComplexTpsa) else ComplexTpsa.from_tpsa(value)
#         for value in values
#     )


# def _coerce_pair(
#     left: Sequence[TpsaSeries] | TpsaMap, right: Sequence[TpsaSeries] | TpsaMap
# ) -> tuple[tuple[TpsaSeries, ...], tuple[TpsaSeries, ...]]:
#     left_series = _series_tuple(left)
#     right_series = _series_tuple(right)
#     if left_series[0].descriptor is not right_series[0].descriptor:
#         message = 'TPSA maps must share the same Descriptor'
#         raise ValueError(message)
#     if _is_complex(left_series) or _is_complex(right_series):
#         return _promote_complex(left_series), _promote_complex(right_series)
#     return left_series, right_series


# def _zeros_like(values: Sequence[TpsaSeries]) -> tuple[TpsaSeries, ...]:
#     from .complex_tpsa import ComplexTpsa

#     result = []
#     for value in values:
#         if isinstance(value, ComplexTpsa):
#             result.append(value.descriptor.complex_zero(order=value.order))
#         else:
#             result.append(value.descriptor.zero(order=value.order))
#     return tuple(result)


# def _parameter_identities(descriptor, *, is_complex: bool) -> tuple[TpsaSeries, ...]:
#     from .complex_tpsa import ComplexTpsa

#     params = descriptor.params()
#     if is_complex:
#         return tuple(ComplexTpsa.from_tpsa(value) for value in params)
#     return tuple(params)


# def _complete_substitution(values: Sequence[TpsaSeries]) -> tuple[TpsaSeries, ...]:
#     descriptor = values[0].descriptor
#     if len(values) != descriptor.num_vars:
#         message = (
#             f'A full map requires {descriptor.num_vars} coordinate components, got {len(values)}'
#         )
#         raise ValueError(message)
#     return (
#         *values,
#         *_parameter_identities(descriptor, is_complex=_is_complex(values)),
#     )


# def _pointer_array(values: Sequence[TpsaSeries]):
#     return ffi.new('void*[]', [value.ptr for value in values])


# def _normalise_selection(select: Sequence[SelectionValue], expected_length: int) -> list[int]:
#     if len(select) != expected_length:
#         message = f'select must contain {expected_length} entries, got {len(select)}'
#         raise ValueError(message)
#     result = []
#     for value in select:
#         if isinstance(value, (bool, np.bool_)):
#             result.append(int(value))
#             continue
#         if not isinstance(value, Integral):
#             message = 'select entries must be booleans or integers 0 or 1'
#             raise TypeError(message)
#         value = int(value)
#         if value not in (0, 1):
#             message = 'integer select entries must be 0 or 1'
#             raise ValueError(message)
#         result.append(value)
#     return result


# def _set_series_const_part(component: TpsaSeries, value: Scalar) -> None:
#     from .tpsa import Tpsa

#     converted = complex(value)
#     if isinstance(component, Tpsa):
#         if converted.imag != 0:
#             err_mess = 'Cannot assign a complex constant to a real TPSA map'
#             raise TypeError(err_mess)
#         component.set_const_part(converted.real)
#     else:
#         component.set_const_part(converted)


# def _set_series_coefficient(component: TpsaSeries, monomial, value: Scalar) -> None:
#     from .tpsa import Tpsa

#     if isinstance(component, Tpsa):
#         converted = complex(value)
#         if converted.imag != 0:
#             err_mess = 'Cannot assign a complex coefficient to a real TPSA map'
#             raise TypeError(err_mess)
#         component.set(monomial, converted.real)
#     else:
#         component.set(monomial, value)


# @overload
# def compose(
#     left: Sequence[Tpsa] | TpsaMap[Tpsa], right: Sequence[Tpsa] | TpsaMap[Tpsa]
# ) -> tuple[Tpsa, ...]: ...
# @overload
# def compose(
#     left: Sequence[Tpsa] | TpsaMap[Tpsa], right: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa]
# ) -> tuple[ComplexTpsa, ...]: ...
# @overload
# def compose(
#     left: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa],
#     right: (Sequence[Tpsa] | Sequence[ComplexTpsa] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]),
# ) -> tuple[ComplexTpsa, ...]: ...
# @overload
# def compose(
#     left: Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
#     right: Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
# ) -> tuple[TpsaSeries, ...]: ...


# def compose(
#     left: Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
#     right: Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
# ) -> tuple[TpsaSeries, ...]:
#     """Return ``left ∘ right``, i.e. ``left(right(z))``.

#     ``right`` must contain one component for every descriptor variable. Descriptor
#     parameters are automatically appended as identity substitutions.
#     """
#     left_series, right_series = _coerce_pair(left, right)
#     descriptor = left_series[0].descriptor

#     if len(left_series) > descriptor.num_vars:
#         message = (
#             f'Left map has {len(left_series)} components, but the descriptor '
#             f'has only {descriptor.num_vars} variables'
#         )
#         raise ValueError(message)
#     if len(right_series) != descriptor.num_vars:
#         message = f'Right map must have {descriptor.num_vars} components, got {len(right_series)}'
#         raise ValueError(message)

#     right_full = _complete_substitution(right_series)
#     result = _zeros_like(left_series)
#     is_complex = _is_complex(left_series)

#     _protected_call(
#         'map_binary',
#         lib.mad_ctpsa_compose if is_complex else lib.mad_tpsa_compose,
#         is_complex=is_complex,
#         args=(
#             len(left_series),
#             _pointer_array(left_series),
#             len(right_full),
#             _pointer_array(right_full),
#             _pointer_array(result),
#         ),
#     )
#     return result


# @overload
# def inverse(values: Sequence[Tpsa] | TpsaMap[Tpsa]) -> tuple[Tpsa, ...]: ...
# @overload
# def inverse(values: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa]) -> tuple[ComplexTpsa, ...]: ...
# @overload
# def inverse(values: Sequence[TpsaSeries]) -> tuple[TpsaSeries, ...]: ...


# def inverse(
#     values: Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
# ) -> tuple[TpsaSeries, ...]:
#     """Return the inverse of a full variable map centered at the origin.

#     Descriptor parameters are treated as independent parameters and carried
#     through the inversion unchanged.
#     """
#     series = _series_tuple(values)
#     descriptor = series[0].descriptor
#     if len(series) != descriptor.num_vars:
#         message = (
#             f'Inverse requires a full {descriptor.num_vars}-component map, got {len(series)}'
#         )
#         raise ValueError(message)

#     full_input = _complete_substitution(series)
#     coordinate_output = _zeros_like(series)
#     full_output = (
#         *coordinate_output,
#         *_parameter_identities(descriptor, is_complex=_is_complex(series)),
#     )
#     is_complex = _is_complex(series)

#     _protected_call(
#         'map_inverse',
#         lib.mad_ctpsa_minv if is_complex else lib.mad_tpsa_minv,
#         is_complex=is_complex,
#         args=(
#             len(full_input),
#             _pointer_array(full_input),
#             descriptor.num_vars,
#             _pointer_array(full_output),
#         ),
#     )
#     return tuple(full_output[: descriptor.num_vars])


# @overload
# def partial_inverse(
#     values: Sequence[Tpsa] | TpsaMap[Tpsa], select: Sequence[SelectionValue]
# ) -> tuple[Tpsa, ...]: ...
# @overload
# def partial_inverse(
#     values: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa], select: Sequence[SelectionValue]
# ) -> tuple[ComplexTpsa, ...]: ...
# @overload
# def partial_inverse(
#     values: Sequence[TpsaSeries], select: Sequence[SelectionValue]
# ) -> tuple[TpsaSeries, ...]: ...


# def partial_inverse(
#     values: Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
#     select: Sequence[SelectionValue],
# ) -> tuple[TpsaSeries, ...]:
#     """Return a partial inverse for selected variable rows of a full map.

#     ``select`` has one entry per descriptor variable. Parameters are never
#     selected for inversion. The map has to be a full variable map centered
#     at the origin.
#     """
#     series = _series_tuple(values)
#     descriptor = series[0].descriptor
#     if len(series) != descriptor.num_vars:
#         message = (
#             f'Partial inverse requires a full {descriptor.num_vars}-component map, '
#             f'got {len(series)}'
#         )
#         raise ValueError(message)

#     full_input = _complete_substitution(series)
#     coordinate_output = _zeros_like(series)
#     full_output = (
#         *coordinate_output,
#         *_parameter_identities(descriptor, is_complex=_is_complex(series)),
#     )
#     full_select = _normalise_selection(select, descriptor.num_vars)
#     full_select.extend([0] * descriptor.num_params)

#     select_array = ffi.new('int[]', full_select)
#     is_complex = _is_complex(series)

#     _protected_call(
#         'map_partial_inverse',
#         lib.mad_ctpsa_pminv if is_complex else lib.mad_tpsa_pminv,
#         is_complex=is_complex,
#         args=(
#             len(full_input),
#             _pointer_array(full_input),
#             descriptor.num_vars,
#             _pointer_array(full_output),
#             select_array,
#         ),
#     )
#     return tuple(full_output[: descriptor.num_vars])


# def evaluate(
#     values: Sequence[TpsaSeries] | TpsaMap,
#     coordinates: Sequence[Scalar],
#     *,
#     parameters: Sequence[Scalar] | None = None,
# ) -> np.ndarray:
#     """Evaluate a TPSA map at numerical coordinates and optional parameters."""
#     series = _series_tuple(values)
#     descriptor = series[0].descriptor

#     coordinate_values = tuple(coordinates)
#     if len(coordinate_values) != descriptor.num_vars:
#         message = (
#             f'Expected {descriptor.num_vars} coordinate values, '
#             f'got {len(coordinate_values)}'
#         )
#         raise ValueError(message)

#     if parameters is None:
#         parameter_values: tuple[Scalar, ...] = (0.0,) * descriptor.num_params
#     else:
#         parameter_values = tuple(parameters)
#         if len(parameter_values) != descriptor.num_params:
#             message = (
#                 f'Expected {descriptor.num_params} parameter values, got {len(parameter_values)}'
#             )
#             raise ValueError(message)

#     point = (*coordinate_values, *parameter_values)
#     is_complex = _is_complex(series) or any(complex(value).imag != 0.0 for value in point)
#     if is_complex and not _is_complex(series):
#         series = _promote_complex(series)

#     if is_complex:
#         input_array = ffi.new('double _Complex[]', [complex(value) for value in point])
#         output_array = ffi.new('double _Complex[]', len(series))
#     else:
#         input_array = ffi.new('double[]', [complex(value).real for value in point])
#         output_array = ffi.new('double[]', len(series))

#     _protected_call(
#         'map_eval',
#         lib.mad_ctpsa_eval if is_complex else lib.mad_tpsa_eval,
#         is_complex=is_complex,
#         args=(
#             len(series),
#             _pointer_array(series),
#             len(point),
#             input_array,
#             output_array,
#         ),
#     )

#     if is_complex:
#         return np.asarray([complex(output_array[ii]) for ii in range(len(series))])
#     return np.asarray([float(output_array[ii]) for ii in range(len(series))])


# @overload
# def translate(
#     values: Sequence[Tpsa] | TpsaMap[Tpsa], offsets: Sequence[SupportsFloat]
# ) -> tuple[Tpsa, ...]: ...
# @overload
# def translate(
#     values: Sequence[Tpsa] | TpsaMap[Tpsa],
#     offsets: Sequence[SupportsComplex],
# ) -> tuple[ComplexTpsa, ...]: ...
# @overload
# def translate(
#     values: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa], offsets: Sequence[Scalar]
# ) -> tuple[ComplexTpsa, ...]: ...
# @overload
# def translate(
#     values: Sequence[TpsaSeries], offsets: Sequence[Scalar]
# ) -> tuple[TpsaSeries, ...]: ...


# def translate(
#     values: Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa], offsets: Sequence[Scalar]
# ) -> tuple[TpsaSeries, ...]:
#     """Return the map after substituting ``z -> z + offsets``.

#     This is implemented through native composition so descriptor parameters remain
#     identity parameters. It therefore also works for parametric maps.
#     """
#     series = _series_tuple(values)
#     descriptor = series[0].descriptor
#     offsets = tuple(offsets)
#     if len(offsets) != descriptor.num_vars:
#         message = f'Expected {descriptor.num_vars} offsets, got {len(offsets)}'
#         raise ValueError(message)

#     is_complex = _is_complex(series) or any(complex(value).imag != 0.0 for value in offsets)
#     if is_complex and not _is_complex(series):
#         series = _promote_complex(series)

#     if is_complex:
#         substitutions: tuple[TpsaSeries, ...] = tuple(
#             descriptor.var(index, complex(value)) for index, value in enumerate(offsets, start=1)
#         )
#     else:
#         substitutions = tuple(
#             descriptor.var(index, complex(value).real)
#             for index, value in enumerate(offsets, start=1)
#         )
#     return compose(series, substitutions)


# @overload
# def vector_to_field(generator: Tpsa) -> tuple[Tpsa, ...]: ...
# @overload
# def vector_to_field(generator: ComplexTpsa) -> tuple[ComplexTpsa, ...]: ...
# @overload
# def vector_to_field(generator: TpsaSeries) -> tuple[TpsaSeries, ...]: ...


# def vector_to_field(generator: TpsaSeries) -> tuple[TpsaSeries, ...]:
#     """Convert a scalar generator to MAD-NG's Hamiltonian vector-field form.

#     This mirrors ``mad_[c]tpsa_vec2fld`` and returns ``G = -J grad(generator)``.
#     """
#     from .complex_tpsa import ComplexTpsa
#     from .tpsa import Tpsa

#     if not isinstance(generator, (Tpsa, ComplexTpsa)):
#         message = 'generator must be Tpsa or ComplexTpsa'
#         raise TypeError(message)
#     descriptor = generator.descriptor
#     if descriptor.num_vars % 2:
#         message = 'Hamiltonian vector fields require an even number of variables'
#         raise ValueError(message)

#     if isinstance(generator, ComplexTpsa):
#         result = tuple(
#             descriptor.complex_zero(order=generator.order) for _ in range(descriptor.num_vars)
#         )
#         is_complex = True
#     else:
#         result = tuple(descriptor.zero(order=generator.order) for _ in range(descriptor.num_vars))
#         is_complex = False

#     _protected_call(
#         'scalar_to_map',
#         lib.mad_ctpsa_vec2fld if is_complex else lib.mad_tpsa_vec2fld,
#         is_complex=is_complex,
#         args=(
#             descriptor.num_vars,
#             generator.ptr,
#             _pointer_array(result),
#         ),
#     )
#     return result


# @overload
# def field_to_vector(field: Sequence[Tpsa] | TpsaMap[Tpsa]) -> Tpsa: ...
# @overload
# def field_to_vector(field: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa]) -> ComplexTpsa: ...
# @overload
# def field_to_vector(field: Sequence[TpsaSeries]) -> TpsaSeries: ...


# def field_to_vector(
#     field: Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
# ) -> TpsaSeries:
#     """Convert a Hamiltonian vector field to its scalar generator."""
#     series = _series_tuple(field)
#     descriptor = series[0].descriptor
#     if len(series) != descriptor.num_vars:
#         message = f'Field must have {descriptor.num_vars} components, got {len(series)}'
#         raise ValueError(message)
#     if descriptor.num_vars % 2:
#         message = 'Hamiltonian vector fields require an even number of variables'
#         raise ValueError(message)

#     max_order = min(descriptor.order, max(value.order for value in series) + 1)
#     if _is_complex(series):
#         result: TpsaSeries = descriptor.complex_zero(order=max_order)
#         is_complex = True
#     else:
#         result = descriptor.zero(order=max_order)
#         is_complex = False

#     _protected_call(
#         'map_to_scalar',
#         lib.mad_ctpsa_fld2vec if is_complex else lib.mad_tpsa_fld2vec,
#         is_complex=is_complex,
#         args=(
#             len(series),
#             _pointer_array(series),
#             result.ptr,
#         ),
#     )
#     return result


# @overload
# def lie_bracket(
#     left: Sequence[Tpsa] | TpsaMap[Tpsa], right: Sequence[Tpsa] | TpsaMap[Tpsa]
# ) -> tuple[Tpsa, ...]: ...
# @overload
# def lie_bracket(
#     left: Sequence[Tpsa] | TpsaMap[Tpsa], right: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa]
# ) -> tuple[ComplexTpsa, ...]: ...
# @overload
# def lie_bracket(
#     left: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa],
#     right: (Sequence[Tpsa] | Sequence[ComplexTpsa] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]),
# ) -> tuple[ComplexTpsa, ...]: ...
# @overload
# def lie_bracket(
#     left: Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
#     right: Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
# ) -> tuple[TpsaSeries, ...]: ...


# def lie_bracket(
#     left: Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
#     right: Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
# ) -> tuple[TpsaSeries, ...]:
#     """Return the Lie bracket of two vector fields."""
#     left_series, right_series = _coerce_pair(left, right)
#     if len(left_series) != len(right_series):
#         message = 'Vector fields must have the same number of components'
#         raise ValueError(message)

#     result = _zeros_like(left_series)
#     is_complex = _is_complex(left_series)
#     _protected_call(
#         'map_pair',
#         lib.mad_ctpsa_liebra if is_complex else lib.mad_tpsa_liebra,
#         is_complex=is_complex,
#         args=(
#             len(left_series),
#             _pointer_array(left_series),
#             _pointer_array(right_series),
#             _pointer_array(result),
#         ),
#     )
#     return result


# @overload
# def exp_poisson(
#     values: Sequence[Tpsa] | TpsaMap[Tpsa], generator: Tpsa | Sequence[Tpsa] | TpsaMap[Tpsa]
# ) -> tuple[Tpsa, ...]: ...
# @overload
# def exp_poisson(
#     values: Sequence[Tpsa] | TpsaMap[Tpsa],
#     generator: (ComplexTpsa | Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa]),
# ) -> tuple[ComplexTpsa, ...]: ...
# @overload
# def exp_poisson(
#     values: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa],
#     generator: (TpsaSeries | Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]),
# ) -> tuple[ComplexTpsa, ...]: ...
# @overload
# def exp_poisson(
#     values: Sequence[TpsaSeries],
#     generator: (TpsaSeries | Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]),
# ) -> tuple[TpsaSeries, ...]: ...


# def exp_poisson(
#     values: Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
#     generator: (TpsaSeries | Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]),
# ) -> tuple[TpsaSeries, ...]:
#     """Apply MAD-NG's exponential Poisson-bracket map.

#     When ``generator`` is a scalar TPSA this mirrors MAD-NG ``damap:exppb(f)``.
#     MAD-NG converts the scalar generator as ``-vec2fld(f)`` before applying the
#     native map exponential.
#     """
#     from .complex_tpsa import ComplexTpsa
#     from .tpsa import Tpsa

#     series = _series_tuple(values)
#     descriptor = series[0].descriptor
#     if len(series) != descriptor.num_vars:
#         message = (
#             f'exp_poisson requires a full {descriptor.num_vars}-component map, got {len(series)}'
#         )
#         raise ValueError(message)

#     if isinstance(generator, (Tpsa, ComplexTpsa)):
#         if generator.descriptor is not descriptor:
#             message = 'Generator and map must share the same Descriptor'
#             raise ValueError(message)
#         is_complex = _is_complex(series) or isinstance(generator, ComplexTpsa)
#         if is_complex and not _is_complex(series):
#             series = _promote_complex(series)
#         if is_complex and isinstance(generator, Tpsa):
#             generator = ComplexTpsa.from_tpsa(generator)
#         field = tuple(-component for component in vector_to_field(generator))
#     else:
#         series, field = _coerce_pair(series, generator)
#         is_complex = _is_complex(series)
#         if len(field) != descriptor.num_vars:
#             message = (
#                 f'Generator field must have {descriptor.num_vars} components, got {len(field)}'
#             )
#             raise ValueError(message)

#     result = _zeros_like(series)
#     _protected_call(
#         'map_binary',
#         lib.mad_ctpsa_exppb if is_complex else lib.mad_tpsa_exppb,
#         is_complex=is_complex,
#         args=(
#             len(field),
#             _pointer_array(field),
#             len(series),
#             _pointer_array(series),
#             _pointer_array(result),
#         ),
#     )
#     return result


# def _log_poisson_impl(
#     values: Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
#     initial_guess: (Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None) = None,
# ) -> tuple[TpsaSeries, ...]:
#     series = _series_tuple(values)
#     descriptor = series[0].descriptor
#     if len(series) != descriptor.num_vars:
#         err_mess = (
#             f'log_poisson requires a full {descriptor.num_vars}-component map, got {len(series)}'
#         )
#         raise ValueError(err_mess)
#     initial_series = None
#     if initial_guess is not None:
#         series, initial_series = _coerce_pair(series, initial_guess)
#         if len(initial_series) != descriptor.num_vars:
#             err_mess = (
#                 f'Field `initial_guess` must have {descriptor.num_vars} components, '
#                 f'got {len(initial_series)}'
#             )
#             raise ValueError(err_mess)
#     result = _zeros_like(series)
#     is_complex = _is_complex(series)
#     initial_ptrs = ffi.NULL if initial_series is None else _pointer_array(initial_series)
#     _protected_call(
#         'map_pair',
#         lib.mad_ctpsa_logpb if is_complex else lib.mad_tpsa_logpb,
#         is_complex=is_complex,
#         args=(
#             len(series),
#             _pointer_array(series),
#             initial_ptrs,
#             _pointer_array(result),
#         ),
#     )
#     return result


# @overload
# def log_poisson(
#     values: Sequence[Tpsa] | TpsaMap[Tpsa],
#     initial_guess: Sequence[Tpsa] | TpsaMap[Tpsa] | None = None,
# ) -> tuple[Tpsa, ...]: ...
# @overload
# def log_poisson(
#     values: Sequence[Tpsa] | TpsaMap[Tpsa],
#     initial_guess: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa],
# ) -> tuple[ComplexTpsa, ...]: ...
# @overload
# def log_poisson(
#     values: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa],
#     initial_guess: (Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None) = None,
# ) -> tuple[ComplexTpsa, ...]: ...
# @overload
# def log_poisson(
#     values: Sequence[TpsaSeries],
#     initial_guess: (Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None) = None,
# ) -> tuple[TpsaSeries, ...]: ...


# def log_poisson(
#     values: Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
#     initial_guess: (Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None) = None,
# ) -> tuple[TpsaSeries, ...]:
#     """Return MAD-NG's vector-field logarithm of a map.

#     The result is the vector field accepted by the native ``exppb`` operation,
#     not yet the scalar Hamiltonian generator.
#     """
#     return _log_poisson_impl(values, initial_guess)


# @overload
# def log_generator(
#     values: Sequence[Tpsa] | TpsaMap[Tpsa],
#     initial_guess: Sequence[Tpsa] | TpsaMap[Tpsa] | None = None,
# ) -> Tpsa: ...
# @overload
# def log_generator(
#     values: Sequence[Tpsa] | TpsaMap[Tpsa],
#     initial_guess: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa],
# ) -> ComplexTpsa: ...
# @overload
# def log_generator(
#     values: Sequence[ComplexTpsa] | TpsaMap[ComplexTpsa],
#     initial_guess: (Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None) = None,
# ) -> ComplexTpsa: ...
# @overload
# def log_generator(
#     values: Sequence[TpsaSeries],
#     initial_guess: (Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None) = None,
# ) -> TpsaSeries: ...


# def log_generator(
#     values: Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
#     initial_guess: (Sequence[TpsaSeries] | TpsaMap[Tpsa] | TpsaMap[ComplexTpsa] | None) = None,
# ) -> TpsaSeries:
#     """Return the scalar generator corresponding to :func:`log_poisson`.

#     The minus sign mirrors MAD-NG's high-level ``damap:exppb(f)`` convention,
#     which converts a scalar ``f`` to ``-vec2fld(f)``.
#     """
#     return -field_to_vector(_log_poisson_impl(values, initial_guess))


# def _pullback_impl(
#     function: TpsaSeries, map_: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
# ) -> TpsaSeries:
#     if function.descriptor is not map_.descriptor:
#         message = 'Function and map must share the same Descriptor'
#         raise ValueError(message)
#     if len(map_) != map_.num_vars:
#         message = f'Pullback requires a full {map_.num_vars}-component map, got {len(map_)}'
#         raise ValueError(message)
#     return compose((function,), map_.coords)[0]


# @overload
# def pullback(function: Tpsa, map_: TpsaMap[Tpsa]) -> Tpsa: ...
# @overload
# def pullback(function: Tpsa, map_: TpsaMap[ComplexTpsa]) -> ComplexTpsa: ...
# @overload
# def pullback(
#     function: ComplexTpsa, map_: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]
# ) -> ComplexTpsa: ...
# @overload
# def pullback(function: TpsaSeries, map_: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]) -> TpsaSeries: ...


# def pullback(function: TpsaSeries, map_: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]) -> TpsaSeries:
#     """Return the pullback ``function ∘ map_``."""
#     return _pullback_impl(function, map_)


# def map_order(values: Sequence[TpsaSeries] | TpsaMap) -> int:
#     """Largest allocated TPSA order among map components."""
#     series = _series_tuple(values)
#     return max(value.order for value in series)


# def max_nonzero_order(values: Sequence[TpsaSeries] | TpsaMap) -> int:
#     """Largest populated TPSA order among map components."""
#     series = _series_tuple(values)
#     return max(value.max_nonzero_order for value in series)


# def norm(values: Sequence[TpsaSeries] | TpsaMap) -> float:
#     """MAD-NG map norm, equal to the sum of component coefficient norms."""
#     series = _series_tuple(values)
#     return float(sum(value.norm() for value in series))
