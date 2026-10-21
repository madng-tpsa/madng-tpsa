"""Formatting helpers for TPSA polynomials.

The public :data:`FormatStyle` alias selects code-like, mathematical, or table
output. :func:`format_polynomial` is used by the TPSA objects' ``format``
methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, no_type_check, overload

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, SupportsFloat

    from ._tpsa_base import _TpsaBase

FormatStyle = Literal['unicode', 'code', 'math', 'table']


_SUPERSCRIPT_MAP = {
    '0': '⁰',
    '1': '¹',
    '2': '²',
    '3': '³',
    '4': '⁴',
    '5': '⁵',
    '6': '⁶',
    '7': '⁷',
    '8': '⁸',
    '9': '⁹',
    '+': '⁺',
    '-': '⁻',
    '=': '⁼',
    '(': '⁽',
    ')': '⁾',
    'a': 'ᵃ',
    'b': 'ᵇ',
    'c': 'ᶜ',
    'd': 'ᵈ',
    'e': 'ᵉ',
    'f': 'ᶠ',
    'g': 'ᵍ',
    'h': 'ʰ',
    'i': 'ⁱ',
    'j': 'ʲ',
    'k': 'ᵏ',
    'l': 'ˡ',
    'm': 'ᵐ',
    'n': 'ⁿ',
    'o': 'ᵒ',
    'p': 'ᵖ',
    'r': 'ʳ',
    's': 'ˢ',
    't': 'ᵗ',
    'u': 'ᵘ',
    'v': 'ᵛ',
    'w': 'ʷ',
    'x': 'ˣ',
    'y': 'ʸ',
    'z': 'ᶻ',
}  # q is missing

_SUBSCRIPT_MAP = str.maketrans(
    {
        '0': '₀',
        '1': '₁',
        '2': '₂',
        '3': '₃',
        '4': '₄',
        '5': '₅',
        '6': '₆',
        '7': '₇',
        '8': '₈',
        '9': '₉',
        '+': '₊',
        '-': '₋',
        '=': '₌',
        '(': '₍',
        ')': '₎',
        'a': 'ₐ',
        'e': 'ₑ',
        'h': 'ₕ',
        'i': 'ᵢ',
        'j': 'ⱼ',
        'k': 'ₖ',
        'l': 'ₗ',
        'm': 'ₘ',
        'n': 'ₙ',
        'o': 'ₒ',
        'p': 'ₚ',
        's': 'ₛ',
        't': 'ₜ',
        'x': 'ₓ',
        'y': 'ᵧ',
        'z': '𝓏',
    }
)

_GREEK_MAP = {
    'Alpha': 'Α',
    'alpha': 'α',
    'Beta': 'Β',
    'beta': 'β',
    'Gamma': 'Γ',
    'gamma': 'γ',
    'Delta': 'Δ',
    'delta': 'δ',
    'Epsilon': 'Ε',
    'epsilon': 'ε',
    'Zeta': 'Ζ',
    'zeta': 'ζ',
    'Eta': 'Η',
    'eta': 'η',
    'Theta': 'Θ',
    'theta': 'θ',
    'Iota': 'Ι',
    'iota': 'ι',
    'Kappa': 'Κ',
    'kappa': 'κ',
    'Lambda': 'Λ',
    'lambda': 'λ',
    'Mu': 'Μ',
    'mu': 'μ',
    'Nu': 'Ν',
    'nu': 'ν',
    'Xi': 'Ξ',
    'xi': 'ξ',
    'Omicron': 'Ο',
    'omicron': 'ο',
    'Pi': 'Π',
    'pi': 'π',
    'Rho': 'Ρ',
    'rho': 'ρ',
    'Sigma': 'Σ',
    'sigma': 'σ',
    'varsigma': 'ς',
    'Tau': 'Τ',
    'tau': 'τ',
    'Upsilon': 'Υ',
    'upsilon': 'υ',
    'Phi': 'Φ',
    'phi': 'φ',
    'Chi': 'Χ',
    'chi': 'χ',
    'Psi': 'Ψ',
    'psi': 'ψ',
    'Omega': 'Ω',
    'omega': 'ω',
}


def to_superscript(s: str | int) -> str:
    """Move all characters in a string to superscript, if possible (keep basescript otherwise)."""
    if isinstance(s, int):
        s = str(s)
    return ''.join(_SUPERSCRIPT_MAP.get(c, c) for c in str(s))


def to_subscript(s: str | int) -> str:
    """Move all characters in a string to subscript, if possible (keep basescript otherwise)."""
    if isinstance(s, int):
        s = str(s)
    return ''.join(_SUBSCRIPT_MAP.get(c, c) for c in str(s))


def format_coeff(coef: SupportsFloat) -> str:
    """Format a coefficient with explicit exponential notation."""
    value = float(coef)
    formatted = f'{value:.3g}' if 0.1 <= abs(value) < 1000.0 else f'{value:.2e}'
    parts = formatted.split('e')
    if len(parts) == 1:
        return parts[0]
    base, exponent = parts
    exponent = int(exponent)
    if exponent == 0:
        return base
    return f'{base}·10{to_superscript(str(exponent))}'


def format_coord(coord: str) -> str:
    """Format a coordinate with Greek letters and potential subscripts for momenta."""
    new_coord = _GREEK_MAP.get(coord, coord)
    if new_coord == coord and coord.startswith('p') and len(coord) > 1:
        # Try momenta
        momentum = new_coord[1:]
        momentum = to_subscript(_GREEK_MAP.get(momentum, momentum))
        return f'p{momentum}'
    return new_coord


@overload
def format_polynomial(
    tpsa: _TpsaBase[Any, Any],
    labels: Sequence[str],
    style: Literal['unicode', 'code'],
    max_terms: int | None = None,
) -> str: ...


@overload
def format_polynomial(
    tpsa: _TpsaBase[Any, Any],
    labels: Sequence[str],
    style: FormatStyle,
    max_terms: int | None = None,
) -> object: ...


def format_polynomial(
    tpsa: _TpsaBase[Any, Any],
    labels: Sequence[str],
    style: FormatStyle,
    max_terms: int | None = None,
) -> object:
    """Format ``tpsa`` using ``labels`` for variables and parameters."""
    if max_terms is not None and max_terms < 0:
        message = 'max_terms must be non-negative or None'
        raise ValueError(message)

    coeffs = tpsa.monomial_coeffs()
    end = ''
    if max_terms is not None:
        coefficients = dict(list(coeffs.items())[:max_terms])
        if len(coefficients) < len(coeffs):
            end = ' + ...'
    else:
        coefficients = coeffs
    if style == 'unicode':
        if not coefficients and end:
            return '...'
        return _format_unicode(coefficients, labels) + end
    if style == 'code':
        if not coefficients and end:
            return '...'
        return _format_code(coefficients, labels) + end
    if style == 'math':
        return _format_math(coefficients, labels)
    if style == 'table':
        return _format_table(coefficients, labels)
    message = "Style must be 'unicode', 'code', 'math', or 'table'"
    raise ValueError(message)


def _format_unicode(coeffs: dict[tuple[int, ...], float | complex], labels: Sequence[str]) -> str:
    """Format a series into an expression such as ``'1.38·10⁻¹² + ζ + 2pζ - 3ζ² - 2ζpζ - pζ²'``."""
    formatted_labels = tuple(format_coord(ll) for ll in labels)
    return _format_terms(
        coeffs,
        formatted_labels,
        separator='·',
        format_power=lambda label, exponent: f'{label}{to_superscript(exponent)}',
        format_coeff=format_coeff,
    )


def _format_code(coeffs: dict[tuple[int, ...], float | complex], labels: Sequence[str]) -> str:
    """Format a series into an expression such as ``'12 + 2 * x + 3 * x * y**2'``."""
    return _format_terms(
        coeffs,
        labels,
        separator=' * ',
        format_power=lambda label, exponent: f'{label}**{exponent}',
    )


@no_type_check
def _format_math(coeffs: dict[tuple[int, ...], float | complex], labels: Sequence[str]) -> object:
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
    coeffs: dict[tuple[int, ...], float | complex],
    labels: Sequence[str],
    separator: str,
    format_power: Callable[[str, int], str],
    format_coeff: Callable[[float], str] | None = None,
) -> str:
    if not coeffs:
        return '0'

    chunks: list[str] = []
    for monomial, coefficient in coeffs.items():
        factors = _factors(monomial, labels, format_power)
        if isinstance(coefficient, complex):
            if coefficient.imag != 0:
                if coefficient.real == 0:
                    real = '0'
                else:
                    real = (
                        f'{coefficient.real:g}'
                        if format_coeff is None
                        else format_coeff(coefficient.real)
                    )
                imag = (
                    f'{abs(coefficient.imag):g}'
                    if format_coeff is None
                    else format_coeff(abs(coefficient.imag))
                )
                sign = '+' if coefficient.imag > 0 else '-'
                term = f'({real}{sign}{imag}j)'
                if factors:
                    term += separator + separator.join(factors)
                chunks.append(term if not chunks else f' + {term}')
                continue
            coefficient = coefficient.real

        magnitude = abs(coefficient)
        if factors and magnitude == 1.0:
            term = separator.join(factors)
        else:
            magnitude_text = f'{magnitude:g}' if format_coeff is None else format_coeff(magnitude)
            if factors:
                factor_text = separator.join(factors)
                term = f'{magnitude_text}{separator}{factor_text}'
            else:
                term = magnitude_text

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


def _format_table(coeffs: dict[tuple[int, ...], float | complex], labels: Sequence[str]) -> object:
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
