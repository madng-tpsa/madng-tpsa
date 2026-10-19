"""Type definitions for real TPSAs and complex TPSAs."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, SupportsComplex, SupportsFloat, TypeAlias

    import numpy as np

    from ._tpsa_base import _TpsaBase
    from .complex_tpsa import ComplexTpsa
    from .tpsa import Tpsa

    Scalar: TypeAlias = SupportsFloat | SupportsComplex
    SelectionValue: TypeAlias = bool | int | np.integer[Any]
    TpsaSeries: TypeAlias = Tpsa | ComplexTpsa

    T = TypeVar('T')
    Nested: TypeAlias = 'Sequence[T | Nested[T]]'

# Need the declared type variable at runtime for generic functions.
# Bind to a shared base class, not a union: ty can resolve common members
# through this single interface. The forward reference avoids runtime imports.
TpsaSeriesT = TypeVar('TpsaSeriesT', bound='_TpsaBase[Any, Any]')
