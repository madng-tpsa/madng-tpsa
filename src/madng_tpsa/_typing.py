"""Type definitions for real TPSAs and complex TPSAs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, SupportsComplex, SupportsFloat

    import numpy as np
    from numpy.typing import NDArray

    from ._tpsa_base import _TpsaBase
    from .complex_tpsa import ComplexTpsa
    from .tpsa import Tpsa

    Scalar: TypeAlias = SupportsFloat | SupportsComplex
    SelectionValue: TypeAlias = bool | int | np.integer[Any]
    TpsaSeries: TypeAlias = Tpsa | ComplexTpsa
    # A positional derivative argument is one direction, optionally with a multiplicity.
    DerivativeDirection: TypeAlias = int | np.integer[Any] | str | Tpsa | ComplexTpsa
    DerivativeSpec: TypeAlias = DerivativeDirection | tuple[DerivativeDirection, int]
    DerivativeGroup: TypeAlias = Literal['vars', 'params', 'all']

    T = TypeVar('T')
    ArraySequence: TypeAlias = Sequence[T] | NDArray[Any]
    Nested: TypeAlias = 'ArraySequence[T | Nested[T]]'

# Need the following declared type variables at runtime.

# Bind to a shared base class, not a union: ty can resolve common members
# through this single interface. The forward reference avoids runtime imports.
TpsaSeriesT = TypeVar('TpsaSeriesT', bound='_TpsaBase[Any, Any]')

# An independent type variable for maps passed as operands.
MapSeriesT = TypeVar('MapSeriesT', bound='_TpsaBase[Any, Any]')

# Different matrix norms
NormDefinition = Literal['max_abs', 'frobenius', 'relative_frobenius', 'spectral']
