"""Map operations for MAD-NG TPSA series.

The native MAD-NG library represents a map as an array of TPSA pointers rather
than as a separate C object. This module provides a small Python container
around such an array, and adds map operations as methods to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._typing import MapSeriesT, NormDefinition, Scalar, SelectionValue, TpsaSeries
    from .descriptor import Descriptor
    from .tpsa import Tpsa

from numbers import Integral

import numpy as np

from ._cffi import ffi, lib
from ._typing import TpsaSeriesT
from .complex_tpsa import ComplexTpsa
from .tensor import TpsaTensor, _align_descriptors
from .tpsa import Tpsa
from .vector import TpsaVector, TpsaVectorField


def symplectic_unit_matrix(n: int) -> np.ndarray:
    """Canonical symplectic matrix for n phase-space variables."""
    if n <= 0 or n % 2:
        err_mess = 'Phase-space dimension must be positive and even.'
        raise ValueError(err_mess)

    s2 = np.array([[0, 1], [-1, 0]])
    return np.kron(np.eye(n // 2), s2)


def _matrix_norm(matrix: np.ndarray, norm: NormDefinition):
    if norm == 'max_abs':
        return np.max(np.abs(matrix))
    if norm == 'frobenius':
        return np.linalg.norm(matrix, ord='fro')
    if norm == 'relative_frobenius':
        return float(np.linalg.norm(matrix, ord='fro') / np.sqrt(matrix.shape[0]))
    if norm == 'spectral':
        return np.linalg.norm(matrix, ord=2)
    err_mess = "Argument 'norm' needs to be one of ['max_abs', 'frobenius', "
    err_mess += "'relative_frobenius', 'spectral']"
    raise ValueError(err_mess)


class TpsaMap(TpsaVector[TpsaSeriesT]):
    """A coordinate transformation represented by TPSAs.

    A map associates a new coordinate vector with each point in
    phase space. Its components are interpreted as coordinate
    transformation functions.

    Unlike `TpsaVector`, scalar values represent offsets from
    the corresponding identity coordinates, rather than constant
    polynomials. Unspecified components retain their identity
    values.

    The constructor syntax is inherited from `TpsaVector`.

    Examples
    --------
    Construct an identity map:

    >>> from madng_tpsa import Descriptor, TpsaMap
    >>> d = Descriptor(
    ...     variables=('x', 'px', 'y', 'py'),
    ...     parameters=('k',),
    ...     order=4,
    ... )
    >>> M = TpsaMap(d)

    Mix TPSAs and scalar offsets:

    >>> x, px, y, py = d.vars()
    >>> M = TpsaMap([x, 0, 0.2, -0.1])

    This represents (x, px, y + 0.2, py - 0.1), not
    (x, 0, 0.2, -0.1).

    Specify scalar offsets by name:

    >>> M = TpsaMap(descriptor=d, x=0.1, px=-0.2)

    This represents (x + 0.1, px - 0.2, y, py).

    See Also
    --------
    TpsaVector : General TPSA-valued vectors.
    TpsaVectorField : Infinitesimal generators of transformations.
    """

    __slots__ = '_params'
    _params: tuple[TpsaSeries, ...]

    @staticmethod
    def _scalar_component(
        descriptor: Descriptor, index: int | str, value: Scalar
    ) -> Tpsa | ComplexTpsa:
        return descriptor.var(index, value)

    def _post_init(self) -> None:
        params = tuple(self.descriptor.params())
        if self.is_complex:
            params = tuple(ComplexTpsa.from_tpsa(t) for t in params)
        self._params = params

    @property
    def pointer_array_with_params(self) -> ffi.CData:
        """Return a full C array of pointers to the map's TPSAs, including parameters."""
        return ffi.new('void*[]', [t.ptr for t in (*self.flat, *self._params)])

    def non_symplectic_residual(
        self,
        coordinates: Sequence[Scalar] | np.ndarray | None = None,
        *,
        parameters: Sequence[Scalar] | np.ndarray | None = None,
        norm: NormDefinition = 'max_abs',
    ) -> float:
        """Norm of J.T @ S @ J - S at a phase-space point."""
        jj = self.evaluate_jacobian(coordinates=coordinates, parameters=parameters)
        ss = symplectic_unit_matrix(self.num_vars)
        res = jj.T @ ss @ jj - ss
        return _matrix_norm(res, norm)

    def non_symplectic_coeffs_by_degree(self, norm: NormDefinition = 'max_abs') -> dict[int, float]:
        """Maximum coefficient-matrix norm per phase-space degree.

        Tests formal symplecticity through order N-1.
        """
        n = self.num_vars
        ss = symplectic_unit_matrix(n)
        jj = self.jacobian()
        res = (jj.T @ ss @ jj - ss).truncate(self.truncation_order - 1)  # ty: ignore[unsupported-operator]

        # One numerical matrix for each monomial.
        dtype = complex if res.is_complex else float
        matrices = {}

        for idx, component in enumerate(res.flat):
            i, j = divmod(idx, n)

            for monomial, coeff in component.monomial_coeffs(tol=0).items():
                if monomial not in matrices:
                    matrices[monomial] = np.zeros((n, n), dtype=dtype)

                matrices[monomial][i, j] = coeff

        # Maximum norm per phase-space degree.
        result = dict.fromkeys(range(self.truncation_order), 0.0)
        for monomial, matrix in matrices.items():
            degree = sum(monomial[:n])
            if degree in result:
                result[degree] = max(
                    result[degree],
                    _matrix_norm(matrix, norm),
                )

        return result

    def compose(self, other: TpsaMap[MapSeriesT]) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Compose full coordinate maps, preserving the TpsaMap result type."""
        return cast('TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]', super().compose(other))

    # We define the composition operators here, because every composition
    # always involves a map. We use << (lshift) to represent regular
    # composition (one can imagine it as the opening of the map's round
    # brackets), and >> (rshift) to represent Dragt notation. The following
    # are possible, for T a tensor of general rank, M a map, F a vector
    # field, and f a scalar field:
    #    T << M    T.compose(M)  -> TpsaTensor (purely geometric)
    #    f << M    M.pullback(f) -> Tpsa
    #    N << M    N.compose(M)  -> TpsaMap
    #    F << M    F.compose(M)  -> TpsaVector (purely geometric)
    #    M >> T    T.compose(M)  -> TpsaTensor (purely geometric)
    #    M >> f    M.pullback(f) -> Tpsa
    #    M >> N    N.compose(M)  -> TpsaMap
    #    M >> F    F.compose(M)  -> TpsaVector (purely geometric)

    def __lshift__(self, other: TpsaMap) -> TpsaMap:
        """Use ordinary composition: ``M << N`` means ``M.compose(N)``."""
        if not isinstance(other, TpsaMap):
            return NotImplemented
        return self.compose(other)

    @overload
    def __rlshift__(self, other: TpsaVectorField) -> TpsaVector: ...

    @overload
    def __rlshift__(self, other: TpsaTensor) -> TpsaTensor: ...

    @overload
    def __rlshift__(self, other: Tpsa | ComplexTpsa) -> Tpsa | ComplexTpsa: ...

    def __rlshift__(
        self, other: TpsaVectorField | TpsaTensor | Tpsa | ComplexTpsa
    ) -> TpsaVector | TpsaTensor | Tpsa | ComplexTpsa:
        """Substitute a map into a tensor or scalar: ``X << M``."""
        if isinstance(other, (TpsaVectorField, TpsaTensor)):
            return other.compose(self)
        if isinstance(other, (Tpsa, ComplexTpsa)):
            return self.pullback(other)
        return NotImplemented

    @overload
    def __rshift__(self, other: TpsaMap) -> TpsaMap: ...

    @overload
    def __rshift__(self, other: TpsaVectorField) -> TpsaVector: ...

    @overload
    def __rshift__(self, other: TpsaTensor) -> TpsaTensor: ...

    @overload
    def __rshift__(self, other: Tpsa | ComplexTpsa) -> Tpsa | ComplexTpsa: ...

    def __rshift__(
        self, other: TpsaMap | TpsaVectorField | TpsaTensor | Tpsa | ComplexTpsa
    ) -> TpsaMap | TpsaVector | TpsaTensor | Tpsa | ComplexTpsa:
        """Use Dragt order: ``M >> X`` substitutes M into X.

        For X a map or tensor, compute X.compose(M). For a scalar
        observable f, compute the ordinary pullback f composed with M.
        """
        if isinstance(other, (TpsaMap, TpsaVectorField, TpsaTensor)):
            return other.compose(self)
        if isinstance(other, (Tpsa, ComplexTpsa)):
            return self.pullback(other)
        return NotImplemented

    def composition_power(self, n: int) -> TpsaMap:
        """Compose the map repeatedly by binary exponentiation, for n >= 0.

        Formal powers preserve the algebra under truncation for zero-constant
        maps. Nonzero constants can invalidate formal rearrangement identities.
        """
        if isinstance(n, bool) or not isinstance(n, Integral):
            error_message = 'Composition exponent must be an integer'
            raise TypeError(error_message)
        if n < 0:
            error_message = 'Composition exponent must be non-negative'
            raise ValueError(error_message)
        result = self._new_like(self.descriptor.vars())
        base = self
        while n:
            if n & 1:
                result = result.compose(base)
            n >>= 1
            if n:
                base = base.compose(base)
        return result

    def __xor__(self, exponent: int) -> TpsaMap:
        """``M ^ n`` is n-fold composition, not componentwise power."""
        return self.composition_power(exponent)

    def inverse(self) -> TpsaMap[TpsaSeriesT]:
        """Return the formal inverse map about a zero-constant expansion point.

        Descriptor parameters are retained as identity substitutions.
        The linear map must be invertible.
        """
        if np.any(self.const_part != 0):
            err_mess = 'Map inversion requires zero constant terms; recenter it first'
            raise ValueError(err_mess)
        jac = self.evaluate_jacobian()
        if np.linalg.matrix_rank(jac) < self.num_vars:
            err_mess = 'Cannot invert map: the linear Jacobian is singular'
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
        return cast('TpsaMap[TpsaSeriesT]', self._new_like(result))

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
        selected = np.flatnonzero(choices[: self.num_vars])
        if len(selected):
            jac = self.evaluate_jacobian()
            block = jac[np.ix_(selected, selected)]
            if np.linalg.matrix_rank(block) < len(selected):
                err_mess = 'Cannot partially invert map: selected Jacobian block is singular'
                raise ValueError(err_mess)

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
        return cast('TpsaMap[TpsaSeriesT]', self._new_like(result))

    def exp_poisson(self, generator: Tpsa | ComplexTpsa) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Apply the Lie operator exponential to this map: M.exp_poisson(f) = exp(-:f:) M."""
        if not isinstance(generator, (Tpsa, ComplexTpsa)):
            err_mess = 'Generator must be a scalar TPSA'
            raise TypeError(err_mess)
        left, right, complex_needed = _align_descriptors(
            generator, self, extend_left=False, extend_right=True
        )
        # MAD-NG's exppb works on *fields* and implements the exponentiated grad
        # Starting from a generator, transforming to a field, and then taking
        # the exponentiated grad amounts to the exponentiated Poisson bracket
        # of the original generator
        left = -TpsaVectorField.from_generator(left)
        return self._exppb(left, right, complex_needed=complex_needed)

    def exp_grad(
        self, field: TpsaVectorField[Tpsa] | TpsaVectorField[ComplexTpsa]
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
        """Apply the Lie grad exponential to this map: M.exp_grad(F) = exp(F^i ∂_i) M."""
        if not isinstance(field, TpsaVectorField):
            err_mess = 'Field must be a vector field'
            raise TypeError(err_mess)
        left, right, complex_needed = _align_descriptors(
            field, self, extend_left=True, extend_right=True
        )
        return self._exppb(left, right, complex_needed=complex_needed)

    @staticmethod
    def _exppb(
        left: TpsaVectorField[Tpsa] | TpsaVectorField[ComplexTpsa],
        right: TpsaMap[Tpsa] | TpsaMap[ComplexTpsa],
        *,
        complex_needed: bool,
    ) -> TpsaMap[Tpsa] | TpsaMap[ComplexTpsa]:
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
            left.pointer_array,
            right.num_vars,
            right.pointer_array,
            ffi.new('void*[]', [t.ptr for t in result]),
        )
        if status:
            left.flat[0]._raise_mad_error()
        return right._new_like(result)

    def log_poisson(
        self,
        initial_guess: TpsaVectorField[Tpsa] | TpsaVectorField[ComplexTpsa] | None = None,
        *,
        validate: bool = True,
        atol: float = 1e-12,
        rtol: float = 1e-10,
    ) -> TpsaVectorField[Tpsa] | TpsaVectorField[ComplexTpsa]:
        """Compute a native vector-field logarithm of this coordinate map.

        When ``validate=True`` (default), re-exponentiate the field and
        compare **every represented TPSA coefficient** against the input map.
        A native operation returning successfully is not proof of convergence.

        No symplecticity or Hamiltonian condition is required: a general
        vector field may be the logarithm of a non-symplectic coordinate map.
        ``initial_guess`` is a *vector-field* guess for the MAD-NG iteration,
        not a physical reference map.

        MAD-NG's logarithm is iterative and is most reliable for maps near
        the identity. Even when validation passes, it proves a numerical
        round trip at the retained TPSA order, not global logarithm existence
        or uniqueness. Set ``validate=False`` to avoid re-exponentiation when
        a higher-level normal-form algorithm checks reconstruction itself.
        """
        if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
            err_mess = 'Validation tolerances must be finite and non-negative'
            raise ValueError(err_mess)
        if initial_guess is not None and not isinstance(initial_guess, TpsaVectorField):
            err_mess = f'Map initial_guess must be a TpsaVectorField! Got {type(initial_guess)}'
            raise TypeError(err_mess)
        if initial_guess is None:
            left = self
            right = None
            complex_needed = left.is_complex
        else:
            left, right, complex_needed = _align_descriptors(
                self, initial_guess, extend_left=True, extend_right=True
            )

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
        field = TpsaVectorField(result)
        if validate:
            # Compare against `left`, not self: a supplied initial guess can
            # trigger descriptor alignment or real-to-complex promotion.
            reconstructed = TpsaMap(left.descriptor).exp_grad(field)
            for name, expected_series, actual_series in zip(
                left.component_names, left.flat, reconstructed.flat, strict=True
            ):
                expected = expected_series.monomial_coeffs(tol=0)
                actual = actual_series.monomial_coeffs(tol=0)
                for monomial in expected.keys() | actual.keys():
                    lhs = expected.get(monomial, 0)
                    rhs = actual.get(monomial, 0)
                    if not np.isclose(lhs, rhs, atol=atol, rtol=rtol):
                        err_mess = (
                            'Native log_poisson did not reconstruct the input map: '
                            f'coordinate {name}, monomial {monomial}, '
                            f'expected {lhs}, obtained {rhs}. '
                            'The logarithm may not have converged or the map '
                            'may lie outside the represented TPSA order.'
                        )
                        raise ValueError(err_mess)
        return field

    def log_generator(
        self,
        initial_guess: TpsaVectorField[Tpsa] | TpsaVectorField[ComplexTpsa] | None = None,
        *,
        validate: bool = True,
        atol: float = 1e-12,
        rtol: float = 1e-10,
    ) -> Tpsa | ComplexTpsa:
        """Return the scalar logarithmic generator using MAD-NG's sign.

        When ``validate=True``, reject non-Hamiltonian logarithmic fields or
        generators not reconstructable at this descriptor's order.
        """
        field = self.log_poisson(initial_guess, validate=validate, atol=atol, rtol=rtol)
        return -field.to_generator(validate=validate, atol=atol, rtol=rtol)

    def pullback(self, function: Tpsa | ComplexTpsa) -> Tpsa | ComplexTpsa:
        """Return ``function ∘ self`` (ordinary, not inverse, pullback)."""
        if not isinstance(function, (Tpsa, ComplexTpsa)):
            err_mess = 'Pullback requires a scalar TPSA'
            raise TypeError(err_mess)
        left, right, _ = _align_descriptors(
            self, function, extend_left=True, extend_right=False, reorder_variables=True
        )
        return TpsaTensor([right]).compose(left).flat[0]

    def to_complex(self) -> TpsaMap[ComplexTpsa]:
        """Promote the components, preserving the TpsaMap container. Only for typing."""
        return cast('TpsaMap[ComplexTpsa]', super().to_complex())
