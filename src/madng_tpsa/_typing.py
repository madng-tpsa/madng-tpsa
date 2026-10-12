"""Type definitions for real TPSAs and complex TPSAs."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from typing import Any, SupportsComplex, SupportsFloat, TypeAlias

    import numpy as np

    from .complex_tpsa import ComplexTpsa

    # Only import when checking types to avoid circular imports
    from .tpsa import Tpsa

    Scalar: TypeAlias = SupportsFloat | SupportsComplex
    SelectionValue: TypeAlias = bool | int | np.integer[Any]
    TpsaSeries: TypeAlias = Tpsa | ComplexTpsa

# Need the declared type variable at runtime for generic functions.
# Define using string forward references as the classes don't exist yet.
TpsaSeriesT = TypeVar('TpsaSeriesT', 'Tpsa', 'ComplexTpsa')
