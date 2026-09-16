"""Formatting helpers for TPSA polynomials.

The public :data:`FormatStyle` alias selects code-like, mathematical, or table
output. :func:`format_polynomial` is used by the TPSA objects' ``format``
methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, no_type_check

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .tpsa import Tpsa

FormatStyle = Literal['code', 'math', 'table']


def format_polynomial(tpsa: Tpsa, labels: Sequence[str], style: FormatStyle) -> object:
    """Format ``tpsa`` using ``labels`` for variables and parameters."""
    coefficients = tpsa.to_dict()
    if style == 'code':
        return _format_code(coefficients, labels)
    if style == 'math':
        return _format_math(coefficients, labels)
    if style == 'table':
        return _format_table(coefficients, labels)
    message = "Style must be 'code', 'math', or 'table'"
    raise ValueError(message)


def _format_code(coeffs: dict[tuple[int, ...], float], labels: Sequence[str]) -> str:
    """Format a series into an expression such as ``'12 + 2 * x + 3 * x * y**2'``."""
    return _format_terms(
        coeffs,
        labels,
        separator=' * ',
        format_power=lambda label, exponent: f'{label}**{exponent}',
    )


@no_type_check
def _format_math(coeffs: dict[tuple[int, ...], float], labels: Sequence[str]) -> object:
    """Format a series as Jupyter-friendly Math object, e.g. ``Math('12 + 2 x + 3 x y^{2}')``."""
    from IPython.display import Math

    formatted = _format_terms(
        coeffs,
        labels,
        separator=' ',
        format_power=lambda label, exponent: rf'{label}^{{{exponent}}}',
    )
    return Math(formatted)


def _format_terms(
    coeffs: dict[tuple[int, ...], float],
    labels: Sequence[str],
    separator: str,
    format_power: Callable[[str, int], str],
) -> str:
    if not coeffs:
        return '0'

    chunks: list[str] = []
    for monomial, coefficient in coeffs.items():
        factors = _factors(monomial, labels, format_power)
        magnitude = abs(coefficient)
        if factors and magnitude == 1.0:
            term = separator.join(factors)
        elif factors:
            factor_text = separator.join(factors)
            term = f'{magnitude:g}{separator}{factor_text}'
        else:
            term = f'{magnitude:g}'

        if not chunks:
            chunks.append(f'-{term}' if coefficient < 0 else term)
        else:
            chunks.append(f' - {term}' if coefficient < 0 else f' + {term}')

    return ''.join(chunks)


def _factors(
    monomial: tuple[int, ...],
    labels: Sequence[str],
    format_power: Callable[[str, int], str],
) -> list[str]:
    factors = []
    for label, exponent in zip(labels, monomial, strict=True):
        if exponent == 1:
            factors.append(label)
        elif exponent > 1:
            factors.append(format_power(label, exponent))
    return factors


def _format_table(coeffs: dict[tuple[int, ...], float], labels: Sequence[str]) -> object:
    """Format a series into a rich ``Table``.

    E.g. for Tpsa({(0, 0, 0): 12.0, (1, 0, 0): 2.0, (1, 2, 0): 3.0}):
    ┏━━━┳━━━┳━━━┳━━━━━━━━━━━━━┓
    ┃ x ┃ y ┃ z ┃ coefficient ┃
    ┡━━━╇━━━╇━━━╇━━━━━━━━━━━━━┩
    │ 0 │ 0 │ 0 │          12 │
    │ 1 │ 0 │ 0 │           2 │
    │ 1 │ 2 │ 0 │           3 │
    └───┴───┴───┴─────────────┘
    """
    from rich.table import Table
    from rich.text import Text

    table = Table(show_header=True, title=None)
    for label in labels:
        table.add_column(label, justify='right')
    table.add_column('coefficient', justify='right')

    for monomial, coefficient in coeffs.items():
        table.add_row(
            *(str(exponent) for exponent in monomial),
            Text(f'{coefficient:g}', style='cyan'),
        )

    return table
