"""Wrapper for vector collections of TPSAs.

The TpsaVector class is a narrowed-down version of the TpsaTensor. Each
instance has exactly N internal TPSAs with N the dimensionality of the
underlying phase-space, represented as a vector (rank-1 tensor). This class
specialises into a TpsaVectorField (constructed from a scalar field f as
F = -S grad(f)) or a TpsaMap (after Lie-exponentiation M = exp(-:f:)).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, SupportsComplex, SupportsFloat, cast, overload

if TYPE_CHECKING:
    import builtins  # To be able to define method 'str' on the class
    from collections.abc import Mapping, Sequence
    from typing import Any, Literal, Self

    from ._typing import MapSeriesT, Scalar, TpsaSeries
    from .map import TpsaMap
    from .tpsa import Tpsa

from collections.abc import Sequence

import numpy as np

from ._cffi import ffi, lib
from ._typing import TpsaSeriesT
from .complex_tpsa import ComplexTpsa
from .descriptor import Descriptor
from .formatting import format_coord, format_polynomial
from .tensor import TpsaTensor, _align_descriptors
from .tpsa import Tpsa


class TpsaVector(TpsaTensor[TpsaSeriesT]):
    """A vector of TPSA-valued functions.

    Contains one TPSA component per descriptor variable, sharing a
    common descriptor. This is merely the geometric structure. Via
    its subclasses, the vector may represent a coordinate function
    (TpsaMap) or a vector field (TpsaVectorField).

    Scalar values represent constant polynomials, and unspecified
    components are zero.

    This class provides the common functionality for coordinate maps
    and vector fields, without assigning either interpretation to
    the components.

    Examples
    --------
    >>> d = Descriptor(variables=('x', 'px'), order=4)
    >>> x, px = d.vars()
    >>> V = TpsaVector(d)               # (0, 0)
    >>> V = TpsaVector([x, 0.2])        # (x, 0.2)
    >>> V = TpsaVector(d, x=0.1)        # (0.1, 0)

    See Also
    --------
    TpsaMap : Finite coordinate transformations.
    TpsaVectorField : Vector fields on phase space.
    """

    __slots__ = ('_coord_names',)
    _coord_names: tuple[builtins.str, ...]

    # ===================================
    # === Construction and Access API ===
    # ===================================

    @staticmethod
    def _scalar_component(
        descriptor: Descriptor,
        index: int | builtins.str,  # noqa: ARG004
        value: Scalar,
    ) -> Tpsa | ComplexTpsa:
        return descriptor.constant(value)

    def _post_init(self) -> None:
        """Initialise subclass-specific state after the TPSAs are constructed."""

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
        """Construct a TPSA vector from its components.

        A vector contains one TPSA component per descriptor variable.
        Components can be supplied as an ordered sequence or as named
        keyword arguments.

        Components may be TPSAs or numerical scalars. The interpretation
        of scalar values and unspecified components depends on the class;
        see the corresponding class docstring.

        Parameters
        ----------
        components : Descriptor or sequence of TPSAs or scalars, optional
            Components of the vector, ordered according to the descriptor
            variables.

            Alternatively, pass a `Descriptor` to construct the default
            vector for the class. If omitted, components can be supplied
            as named keyword arguments.

        descriptor : Descriptor, optional
            Descriptor defining the variables, parameters, and truncation
            order. If omitted, it is inferred from the supplied TPSA
            components.

        **coordinates : TPSA or scalar
            Named vector components, using descriptor variable labels as
            keywords. Each value can be a TPSA or a numerical scalar.

            Components not explicitly provided are initialised according
            to the class's default convention.

            A descriptor must be supplied explicitly if all provided
            components are scalars.

        Examples
        --------
        Construct a vector from TPSA components:

        >>> from madng_tpsa import Descriptor, TpsaVector
        >>> d = Descriptor(
        ...     variables=('x', 'px', 'y', 'py'),
        ...     parameters=('k',),
        ...     order=4,
        ... )
        >>> x, px, y, py = d.vars()
        >>> k = d.param('k')
        >>> V = TpsaVector([x, px + k * x**2, y, py - k * y**2])

        Provide components by name:

        >>> V = TpsaVector(px=px + k * x**2, py=py - k * y**2)

        The same constructor syntax is supported by `TpsaMap` and
        `TpsaVectorField`, with their respective scalar conventions.

        See Also
        --------
        TpsaMap : Coordinate maps, with identity-coordinate defaults.
        TpsaVectorField : Vector fields, with zero-component defaults.
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
                err_mess = 'Nested components require TpsaTensor'
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

        # Construct components
        coords = []
        for nn in names:
            value = coordinates.get(nn, 0.0)
            if isinstance(value, (Tpsa, ComplexTpsa)):
                coords.append(value)
            elif isinstance(value, (SupportsFloat, SupportsComplex)):
                coords.append(self._scalar_component(descriptor, nn, value))
            else:
                err_mess = 'All arguments must be Tpsa, ComplexTpsa, or Scalar'
                raise TypeError(err_mess)

        super().__init__(
            values=cast('Sequence[TpsaSeriesT]', coords),
            shape=(descriptor.num_vars,),
        )
        self._coord_names = names
        self._post_init()

    def _new_like(self, values: Sequence[TpsaSeriesT]) -> Self:
        return type(self)(values)

    @classmethod
    def _parse_dict(
        cls, dct: dict[builtins.str, Any]
    ) -> tuple[Descriptor, tuple[int], Sequence[TpsaSeries]]:
        """Parse and validate a dict representing the map."""
        descriptor, shape, tpsa_flat = super()._parse_dict(dct)
        if len(shape) != 1:
            err_mess = 'A TpsaVector always has to be rank 1!'
            raise ValueError(err_mess)
        return descriptor, shape, tpsa_flat

    @classmethod
    def from_dict(cls, dct: dict[builtins.str, Any]) -> Self:
        """Instantiate a TpsaVector from a dictionary representation."""
        _, _, tpsa_flat = cls._parse_dict(dct)
        return cls(tpsa_flat)

    def change_descriptor(
        self,
        descriptor: Descriptor,
        *,
        variable_map: Mapping[builtins.str, builtins.str] | None = None,
        parameter_map: Mapping[builtins.str, builtins.str] | None = None,
        truncate: bool = False,
    ) -> Self:
        """Convert a full vector, including its output-coordinate ordering."""
        rename = variable_map or {}
        output_names = tuple(rename.get(name, name) for name in self.component_names)
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
            for name, component in self.named_components.items()
        }
        return self._new_like([converted[name] for name in descriptor.variables])

    def extend_descriptor(self, descriptor: Descriptor, *, merge: bool = False) -> Self:
        """Extend a vector, inserting identity components (TpsaMap) or constants (TpsaVectorField).

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
            for name, component in self.named_components.items()
        }
        values = [
            converted[name]
            if name in converted
            else self._scalar_component(descriptor, name, 0j if self.is_complex else 0.0)
            for name in descriptor.variables
        ]
        return self._new_like(cast('Sequence[TpsaSeriesT]', values))

    def __getattr__(self, name: builtins.str) -> TpsaSeriesT:
        """Access the coordinate TPSAs by coordinate name."""
        try:
            names = object.__getattribute__(self, '_coord_names')
            i = names.index(name)
            return object.__getattribute__(self, '_flat')[self._offset + i]
        except (AttributeError, ValueError):
            raise AttributeError(name) from None

    def str(self, *, max_tpsa: int | None = None, max_terms: int | None = 3) -> builtins.str:
        """Return a string representation of the vector, specifying how many terms to show."""
        if max_tpsa is not None and max_tpsa < 0:
            message = 'max_tpsa must be non-negative or None'
            raise ValueError(message)
        if max_terms is not None and max_terms < 0:
            message = 'max_terms must be non-negative or None'
            raise ValueError(message)
        labels = self.descriptor.variables + self.descriptor.parameters
        entries = tuple(self.named_components.items())
        selected = entries[:max_tpsa]

        components: list[builtins.str] = []
        for coord, tpsa in selected:
            tt = format_polynomial(tpsa=tpsa, labels=labels, style='unicode', max_terms=max_terms)
            components.append(f'{format_coord(coord)} = {tt}')

        suffix = ', ...' if len(selected) < len(entries) else ''
        name = type(self).__name__

        return f'{name}({", ".join(components)}{suffix})'

    def show(self, *, max_tpsa: int | None = None, max_terms: int | None = None) -> None:
        """Print a representation of the vector, specifying how many terms to show."""
        print(self.str(max_tpsa=max_tpsa, max_terms=max_terms))  # noqa: T201

    def __repr__(self) -> builtins.str:
        """Return a compact, human-readable polynomial representation."""
        return self.str(max_tpsa=None, max_terms=3)

    # ==================
    # === Attributes ===
    # ==================

    @property
    def component_names(self) -> tuple[builtins.str, ...]:
        """Descriptor-ordered names of the full map outputs."""
        return self._coord_names

    @property
    def components(self) -> tuple[TpsaSeriesT, ...]:
        """Complete coordinate-output TPSAs, in descriptor order."""
        return self.flat

    @property
    def named_components(self) -> Mapping[builtins.str, TpsaSeriesT]:
        """Read-only mapping of output names to mutable TPSA components."""
        return MappingProxyType(dict(zip(self.component_names, self.components, strict=True)))

    # =======================
    # === TPSA Operations ===
    # =======================

    def _coordinate_index(self, coord: builtins.str | int) -> int:
        """Resolve a coordinate name or zero-based integer index."""
        if isinstance(coord, str):
            if coord in self.component_names:
                return self.component_names.index(coord)
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
        coord: builtins.str | int,
        monomials: Sequence[int] | Sequence[Sequence[int]] | np.ndarray,
    ) -> float | complex | np.ndarray:
        """Return coefficient(s) from one output series by coordinate name or index."""
        idx = self._coordinate_index(coord)
        arr = self._monomial_rows(monomials)
        return self.components[idx].coefficient(arr)

    def set_coefficient(
        self,
        coord: builtins.str | int,
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
        component = self.components[idx]
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

    def jacobian(
        self, *, wrt: Literal['variables', 'parameters', 'all'] = 'variables'
    ) -> TpsaTensor[TpsaSeriesT]:
        """Return the rank-two TPSA-valued Jacobian over selected inputs.

        The argument 'wrt' (with respect to) can be used to specify which
        derivatives to include: variables', 'parameters', or 'all'.
        """
        return self.derivative_tensor(1, wrt=wrt)

    def evaluate_jacobian(
        self,
        coordinates: Sequence[Scalar] | np.ndarray | None = None,
        *,
        parameters: Sequence[Scalar] | np.ndarray | None = None,
        wrt: Literal['variables', 'parameters', 'all'] = 'variables',
    ) -> np.ndarray:
        """Evaluate the map Jacobian at a numerical phase-space/parameter point.

        By default evaluate at the origin (all coordinates and parameters zero).
        The result has shape ``(num_outputs, num_inputs)``. The argument 'wrt'
        (with respect to) can be used to specify which derivatives to include in
        num_inputs: variables', 'parameters', or 'all'.
        """
        if wrt == 'parameters' and self.num_params == 0:
            dtype = complex if self.is_complex else float
            return np.empty((len(self), 0), dtype=dtype)
        if coordinates is None and parameters is None:
            if wrt not in ('variables', 'parameters', 'all'):
                err_mess = 'Unknown differentiation direction'
                raise ValueError(err_mess)
            if wrt == 'variables':
                return np.asarray([tpsa.grad() for tpsa in self.components])
            if wrt == 'parameters':
                return np.asarray([tpsa.param_grad() for tpsa in self.components])
            return np.asarray(
                [np.concatenate((tpsa.grad(), tpsa.param_grad())) for tpsa in self.components]
            )
        if coordinates is None:
            coordinates = [0.0] * self.num_vars
        if parameters is None:
            parameters = [0.0] * self.num_params
        return self.jacobian(wrt=wrt).evaluate(coordinates=coordinates, parameters=parameters)

    def hessian(
        self, *, wrt: Literal['variables', 'parameters', 'all'] = 'variables'
    ) -> TpsaTensor[TpsaSeriesT]:
        """Return the rank-three TPSA-valued Hessian over selected inputs.

        Entry ``[i, j, k]`` is the second derivative of output ``i`` with
        respect to inputs ``j`` and ``k`` (variables followed by parameters).
        The argument 'wrt' (with respect to) can be used to specify which
        derivatives to include: variables', 'parameters', or 'all'.
        """
        return self.derivative_tensor(2, wrt=wrt)

    def evaluate_hessian(
        self,
        coordinates: Sequence[Scalar] | np.ndarray | None = None,
        *,
        parameters: Sequence[Scalar] | np.ndarray | None = None,
        wrt: Literal['variables', 'parameters', 'all'] = 'variables',
    ) -> np.ndarray:
        """Evaluate the map Hessian at a numerical phase-space/parameter point.

        By default evaluate at the origin (all coordinates and parameters zero).
        The result has shape ``(num_outputs, num_inputs, num_inputs)``. The
        argument 'wrt' (with respect to) can be used to specify which
        derivatives to include in num_inputs: variables', 'parameters', or 'all'.
        """
        if coordinates is None:
            coordinates = [0.0] * self.num_vars
        if parameters is None:
            parameters = [0.0] * self.num_params
        return self.hessian(wrt=wrt).evaluate(coordinates=coordinates, parameters=parameters)

    def translate(
        self, offsets: Sequence[Scalar] | np.ndarray
    ) -> TpsaVector[Tpsa] | TpsaVector[ComplexTpsa]:
        """Return the input-translated vector ``V(z + offsets, k)``.

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
            from .map import TpsaMap

            substitutions = [
                self.descriptor.var(
                    index,
                    complex(value) if complex_needed else complex(value).real,
                )
                for index, value in enumerate(offsets, start=1)
            ]
            return cast(
                'TpsaVector[Tpsa] | TpsaVector[ComplexTpsa]',
                self.compose(TpsaMap(substitutions)),
            )

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
        return cast('TpsaVector[Tpsa] | TpsaVector[ComplexTpsa]', source._new_like(result))

    def norm(self) -> float:
        """Return the native vector norm (sum of component coefficient norms)."""
        if self.is_complex:
            return float(lib.mad_ctpsa_mnrm(self.num_vars, self.pointer_array))
        return float(lib.mad_tpsa_mnrm(self.num_vars, self.pointer_array))

    # ====================
    # === Typing Casts ===
    # ====================

    def to_complex(self) -> TpsaVector[ComplexTpsa]:
        """Promote the components, preserving the TpsaVector container. Only for typing."""
        return cast('TpsaVector[ComplexTpsa]', super().to_complex())

    @overload
    def __getitem__(self, key: int) -> TpsaSeriesT: ...

    @overload
    def __getitem__(self, key: object) -> TpsaSeriesT | TpsaTensor[TpsaSeriesT]: ...

    def __getitem__(self, key: object) -> TpsaSeriesT | TpsaTensor[TpsaSeriesT]:
        """Overload __getitem__. Only for typing."""
        return super().__getitem__(key)


class TpsaVectorField(TpsaVector[TpsaSeriesT]):
    """A vector field on phase space represented by TPSAs.

    A vector field assigns a tangent vector to each point in
    phase space. It describes an infinitesimal generator, rather
    than a finite coordinate transformation.

    Unlike `TpsaMap`, scalar values represent constant vector
    components, and unspecified components are zero.

    The constructor syntax is inherited from `TpsaVector`.

    Examples
    --------
    Construct the zero vector field:

    >>> from madng_tpsa import Descriptor, TpsaVectorField
    >>> d = Descriptor(variables=('x', 'px'), order=4)
    >>> F = TpsaVectorField(d)                # (0, 0)

    Construct a field from TPSA components:

    >>> x, px = d.vars()
    >>> F = TpsaVectorField([px, -x])        # (px, -x)

    Mix TPSAs and constant components:

    >>> F = TpsaVectorField([x, 0.2])       # (x, 0.2)

    Or provide components by name:

    >>> F = TpsaVectorField(d, x=0.1)       # (0.1, 0)

    See Also
    --------
    TpsaVector : General TPSA-valued vectors.
    TpsaMap : Finite coordinate transformations.
    """

    __slots__ = ()

    # ===================================
    # === Construction and Access API ===
    # ===================================

    @classmethod
    def from_generator(
        cls, generator: Tpsa | ComplexTpsa
    ) -> TpsaVectorField[Tpsa] | TpsaVectorField[ComplexTpsa]:
        """Construct the native Hamiltonian vector field ``F = -S grad(f)``.

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
        return cast('TpsaVectorField[Tpsa] | TpsaVectorField[ComplexTpsa]', TpsaVectorField(output))

    def to_generator(
        self, *, validate: bool = True, atol: float = 1e-12, rtol: float = 1e-10
    ) -> Tpsa | ComplexTpsa:
        """Recover a scalar generator from a Hamiltonian vector field ``int_0^x S F(x';0) dx'``.

        Use MAD-NG's native ``fld2vec`` to invert ``from_generator``:
        the map is interpreted as ``F = -S grad(f)``, with canonical
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
            rebuilt = TpsaVectorField.from_generator(result)
            for name, original, reconstructed in zip(
                self.component_names, self.flat, rebuilt.flat, strict=True
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

    # =======================
    # === TPSA Operations ===
    # =======================

    def lie_bracket(
        self, other: TpsaVectorField[MapSeriesT]
    ) -> TpsaVectorField[Tpsa] | TpsaVectorField[ComplexTpsa]:
        """Return the Lie bracket of two vector fields: [F, G]^i = F^j ∂_j G^i - G^j ∂_j F^i."""
        if not isinstance(other, TpsaVectorField):
            err_mess = f'Can only take Lie bracket with another TpsaVectorField! Got {type(other)}'
            raise TypeError(err_mess)
        left, right, complex_needed = _align_descriptors(
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
        return TpsaVectorField(result)

    def compose(self, other: TpsaMap[MapSeriesT]) -> TpsaVector[Tpsa] | TpsaVector[ComplexTpsa]:
        """Component substitution of a vector field with a map."""
        return TpsaVector(super().compose(other).flat)

    # ====================
    # === Typing Casts ===
    # ====================

    def to_complex(self) -> TpsaVectorField[ComplexTpsa]:
        """Promote the components, preserving the TpsaVectorField container. Only for typing."""
        return cast('TpsaVectorField[ComplexTpsa]', super().to_complex())
