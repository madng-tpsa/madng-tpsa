"""Type definitions for real TPSAs and complex TPSAs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, SupportsComplex, SupportsFloat

    import numpy as np

    from ._tpsa_base import _TpsaBase
    from .complex_tpsa import ComplexTpsa
    from .tpsa import Tpsa

    Scalar: TypeAlias = SupportsFloat | SupportsComplex
    SelectionValue: TypeAlias = bool | int | np.integer[Any]
    TpsaSeries: TypeAlias = Tpsa | ComplexTpsa
    # A bare all-integer sequence of full descriptor length denotes exponent orders.
    # For explicit numeric coordinate indices, use a nested (index, multiplicity) pair.
    DerivativeSpec: TypeAlias = int | str | Tpsa | Sequence[int | str | tuple[int | str, int]]

    T = TypeVar('T')
    Nested: TypeAlias = 'Sequence[T | Nested[T]]'

# Need the following declared type variables at runtime.

# Bind to a shared base class, not a union: ty can resolve common members
# through this single interface. The forward reference avoids runtime imports.
TpsaSeriesT = TypeVar('TpsaSeriesT', bound='_TpsaBase[Any, Any]')

# An independent type variable for maps passed as operands.
MapSeriesT = TypeVar('MapSeriesT', bound='_TpsaBase[Any, Any]')

# Different matrix norms
NormDefinition = Literal['max_abs', 'frobenius', 'relative_frobenius', 'spectral']
