"""Formatting helpers for TPSA polynomials.

The public :data:`FormatStyle` alias selects code-like, mathematical, or table
output. :func:`format_polynomial` is used by the TPSA objects' ``format``
methods.
"""

from __future__ import annotations

from itertools import product
from math import prod
from typing import TYPE_CHECKING, Literal, TypeAlias, overload

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, SupportsFloat

    from ._tpsa_base import _TpsaBase

FormatStyle = Literal['unicode', 'code', 'math', 'table']
FormatStringStyle = Literal['unicode', 'code']
_Coefficients: TypeAlias = dict[tuple[int, ...], float | complex]


class LatexExpression(str):
    """LaTeX source with a rich-display hook, independent of IPython.

    Printing the expression returns raw LaTeX. Compatible notebook frontends
    render it as mathematics via the standard ``_repr_latex_`` protocol.
    """

    @property
    def data(self) -> str:
        """Return the raw LaTeX source (compatible with IPython's Math object)."""
        return str(self)

    def _repr_latex_(self) -> str:
        """Return display mathematics for compatible rich-display frontends."""
        return r'$\displaystyle ' + self + '$'


class PrintableTable(Table):
    """Rich table supporting Python's built-in print()."""

    def __str__(self) -> str:
        console = Console(color_system=None)
        with console.capture() as capture:
            console.print(self)
        return capture.get().rstrip('\n')


# q is missing
_SUPERSCRIPT_MAP = str.maketrans(
    {
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
    }
)

# b,c,d,f,g,q,r,u,v,w are missing
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
    return s.translate(_SUPERSCRIPT_MAP)


def to_subscript(s: str | int) -> str:
    """Move all characters in a string to subscript, if possible (keep basescript otherwise)."""
    if isinstance(s, int):
        s = str(s)
    return s.translate(_SUBSCRIPT_MAP)


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
    *,
    labels: Sequence[str] | None = None,
    style: FormatStringStyle = 'unicode',
    max_terms: int | None = None,
    tol: float | None = None,
) -> str: ...


@overload
def format_polynomial(
    tpsa: _TpsaBase[Any, Any],
    *,
    labels: Sequence[str] | None = None,
    style: Literal['table'],
    max_terms: int | None = None,
    tol: float | None = None,
) -> PrintableTable: ...


@overload
def format_polynomial(
    tpsa: _TpsaBase[Any, Any],
    *,
    labels: Sequence[str] | None = None,
    style: Literal['math'],
    max_terms: int | None = None,
    tol: float | None = None,
) -> LatexExpression: ...


@overload
def format_polynomial(
    tpsa: _TpsaBase[Any, Any],
    *,
    labels: Sequence[str] | None = None,
    style: FormatStyle,
    max_terms: int | None = None,
    tol: float | None = None,
) -> str | PrintableTable: ...


def format_polynomial(
    tpsa: _TpsaBase[Any, Any],
    *,
    labels: Sequence[str] | None = None,
    style: FormatStyle = 'unicode',
    max_terms: int | None = None,
    tol: float | None = None,
) -> str | PrintableTable:
    """Format ``tpsa`` using ``labels`` for variables and parameters."""
    _validate_limits(max_terms=max_terms)
    if labels is None:
        labels = tpsa.descriptor.var_names + tpsa.descriptor.param_names
    coeffs, truncated = _select_coefficients(tpsa, max_terms, tol)
    if style == 'table':
        table = _create_table(labels)
        _add_polynomial_rows(table, coeffs, truncated=truncated)
        return table
    if style == 'math':
        return LatexExpression(_format_expression(coeffs, labels, style, truncated=truncated))
    return _format_expression(coeffs, labels, style, truncated=truncated)


def _validate_limits(max_tpsa: int | None = None, max_terms: int | None = None) -> None:
    if max_tpsa is not None and max_tpsa < 0:
        message = 'max_tpsa must be non-negative or None'
        raise ValueError(message)
    if max_terms is not None and max_terms < 0:
        message = 'max_terms must be non-negative or None'
        raise ValueError(message)


def _select_coefficients(
    tpsa: _TpsaBase[Any, Any], max_terms: int | None, tol: float | None
) -> tuple[_Coefficients, bool]:
    coeffs = tpsa.monomial_coeffs() if tol is None else tpsa.monomial_coeffs(tol=tol)
    if max_terms is None or len(coeffs) <= max_terms:
        return coeffs, False
    return dict(list(coeffs.items())[:max_terms]), True


def _format_expression(
    coeffs: _Coefficients, labels: Sequence[str], style: FormatStyle, *, truncated: bool
) -> str:
    if style == 'unicode':
        expression = _format_unicode(coeffs, labels)
    elif style == 'code':
        expression = _format_code(coeffs, labels)
    elif style == 'math':
        expression = _format_latex(coeffs, labels)
    else:
        message = "Style must be 'unicode', 'code', 'math', or 'table'"
        raise ValueError(message)
    if not truncated:
        return expression
    ellipsis = r'\ldots' if style == 'math' else '...'
    return f'{expression} + {ellipsis}' if coeffs else ellipsis


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


def _format_coord_latex(coord: str) -> str:
    # latin_greek = {
    #     'Alpha': 'A',
    #     'Beta': 'B',
    #     'Epsilon': 'E',
    #     'Zeta': 'Z',
    #     'Eta': 'H',
    #     'Iota': 'I',
    #     'Kappa': 'K',
    #     'Mu': 'M',
    #     'Nu': 'N',
    #     'Omicron': 'O',
    #     'omicron': 'o',
    #     'Rho': 'P',
    #     'Tau': 'T',
    #     'Chi': 'X',
    # }
    # if coord in latin_greek:
    #     return latin_greek[coord]
    if coord in _GREEK_MAP:
        return '\\' + coord
    if coord.startswith('p') and (coord[1:] in _GREEK_MAP or coord[1:] in ('x', 'y', 'z', 't')):
        return rf'p_{{{_format_coord_latex(coord[1:])}}}'
    escaped = coord.translate(
        str.maketrans(
            {
                '\\': r'\backslash{}',
                '{': r'\{',
                '}': r'\}',
                '_': r'\_',
                '%': r'\%',
                '&': r'\&',
                '#': r'\#',
                '$': r'\$',
                '^': r'\^{}',
                '~': r'\~{}',
                ' ': r'\ ',
            }
        )
    )
    if len(coord) == 1 and coord.isascii() and coord.isalpha():
        return escaped
    return rf'\mathrm{{{escaped}}}'


def _format_latex_coeff(value: float) -> str:
    mantissa, sep, exponent = f'{value:g}'.partition('e')
    return rf'{mantissa}\times 10^{{{int(exponent)}}}' if sep else mantissa


def _format_latex(coeffs: _Coefficients, labels: Sequence[str]) -> str:
    return _format_terms(
        coeffs,
        tuple(_format_coord_latex(label) for label in labels),
        separator=' ',
        format_power=lambda label, exponent: rf'{label}^{{{exponent}}}',
        format_coeff=_format_latex_coeff,
        imaginary_unit=r'\mathrm{i}',
    )


def _format_terms(
    coeffs: dict[tuple[int, ...], float | complex],
    labels: Sequence[str],
    separator: str,
    format_power: Callable[[str, int], str],
    format_coeff: Callable[[float], str] | None = None,
    imaginary_unit: str = 'j',
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
                term = f'({real}{sign}{imag}{imaginary_unit})'
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


def _create_table(labels: Sequence[str], prefix: Sequence[str] = ()) -> PrintableTable:
    table = PrintableTable(show_header=True, title=None, header_style='bold deep_pink2')
    for label in prefix:
        table.add_column(label, justify='left', style='bold green', no_wrap=True)
    for label in labels:
        table.add_column(label, justify='right')
    table.add_column('coefficient', justify='right', style='cyan')
    return table


def _add_polynomial_rows(
    table: PrintableTable,
    coeffs: _Coefficients,
    *,
    indices: Sequence[str] = (),
    truncated: bool = False,
    end_section: bool = False,
) -> None:
    from rich.text import Text

    rows: list[tuple[str | Text, ...]] = [
        (*(str(exponent) for exponent in monomial), f'{coefficient:g}')
        for monomial, coefficient in coeffs.items()
    ]
    num_exponents = len(table.columns) - len(indices) - 1
    if truncated:
        rows.append(('',) * num_exponents + (Text('…', style='dim'),))
    elif not rows:
        rows.append(('0',) * num_exponents + (Text('0', style='cyan'),))

    for i, row in enumerate(rows):
        prefix = indices if i == 0 else ('',) * len(indices)
        table.add_row(*prefix, *row, end_section=end_section and i == len(rows) - 1)


def _format_nested(components: Sequence[str], shape: tuple[int, ...]) -> str:
    """Render a C-order prefix of a tensor with NumPy-like nesting."""

    def render(dims: tuple[int, ...], offset: int) -> str:
        if not dims:
            return components[offset] if offset < len(components) else '...'
        stride = prod(dims[1:])
        children: list[str] = []
        for i in range(dims[0]):
            start = offset + i * stride
            if stride and start >= len(components):
                children.append('...')
                break
            children.append(render(dims[1:], start))
        separator = ', ' if len(dims) == 1 else ',\n' + '\n' * (len(dims) - 2)
        body = separator.join(children)
        indented = '\n'.join(
            line if i == 0 or not line else ' ' + line for i, line in enumerate(body.split('\n'))
        )
        return '[' + indented + ']'

    return render(shape, 0)


def _latex_environment(name: str, rows: Sequence[str]) -> str:
    return rf'\begin{{{name}}}' + r' \\ '.join(rows) + rf'\end{{{name}}}'


def _format_latex_matrix(components: Sequence[str], shape: tuple[int, int]) -> str:
    nrows, ncols = shape
    if not nrows or not ncols:
        return _latex_environment('pmatrix', [])
    rows: list[str] = []
    for i in range(nrows):
        start = i * ncols
        if start >= len(components):
            rows.append(r'\vdots')
            break
        entries = list(components[start : start + ncols])
        if len(entries) < ncols:
            entries.append(r'\cdots')
        rows.append(' & '.join(entries))
    return _latex_environment('pmatrix', rows)


def _format_latex_tensor(components: Sequence[str], shape: tuple[int, ...]) -> str:
    if not shape:
        return components[0] if components else r'\ldots'
    if len(shape) == 1:
        return _format_latex_matrix(components, (shape[0], 1))
    matrix_shape = (shape[-2], shape[-1])
    if len(shape) == 2:
        return _format_latex_matrix(components, matrix_shape)
    stride = prod(matrix_shape)
    rows: list[str] = []
    if not prod(shape):
        return r'\varnothing'
    for i, prefix in enumerate(product(*(range(n) for n in shape[:-2]))):
        start = i * stride
        if start >= len(components):
            rows.append(r'\vdots')
            break
        label = ','.join(str(j) for j in prefix)
        matrix = _format_latex_matrix(components[start : start + stride], matrix_shape)
        rows.append(rf'T_{{{label},:,:}} &= {matrix}')
    return _latex_environment('aligned', rows)


def _format_tensor(
    components: Sequence[_TpsaBase[Any, Any]],
    shape: tuple[int, ...],
    labels: Sequence[str],
    *,
    names: Sequence[str] | None = None,
    style: FormatStyle = 'unicode',
    max_tpsa: int | None = None,
    max_terms: int | None = None,
    tol: float | None = None,
) -> str | PrintableTable:
    """Format a tensor or named vector without stringifying rich display objects."""
    _validate_limits(max_tpsa, max_terms)
    if style not in ('unicode', 'code', 'math', 'table'):
        message = "Style must be 'unicode', 'code', 'math', or 'table'"
        raise ValueError(message)
    selected = components[:max_tpsa]
    omitted = len(selected) < len(components)
    if style == 'table':
        prefix = ('output',) if names is not None else tuple(f'i{i}' for i in range(len(shape)))
        table = _create_table(labels, prefix)
        if omitted:
            table.caption = f'Showing {len(selected)} of {len(components)} components (… omitted)'
        indices = product(*(range(n) for n in shape))
        for i, (index, tpsa) in enumerate(zip(indices, selected, strict=False)):
            coeffs, truncated = _select_coefficients(tpsa, max_terms, tol)
            row_indices = (names[i],) if names is not None else tuple(str(j) for j in index)
            _add_polynomial_rows(
                table,
                coeffs,
                indices=row_indices,
                truncated=truncated,
                end_section=i < len(selected) - 1,
            )
        return table

    expressions = []
    for tpsa in selected:
        coeffs, truncated = _select_coefficients(tpsa, max_terms, tol)
        expressions.append(_format_expression(coeffs, labels, style, truncated=truncated))

    if names is not None:
        if style == 'math':
            rows = [
                rf'{_format_coord_latex(name)} &= {expression}'
                for name, expression in zip(names[: len(selected)], expressions, strict=True)
            ]
            if omitted:
                rows.append(r'\vdots')
            return LatexExpression(_latex_environment('aligned', rows))
        display_names = [format_coord(name) if style == 'unicode' else name for name in names]
        rows = [
            f'{name} = {expression}'
            for name, expression in zip(display_names[: len(selected)], expressions, strict=True)
        ]
        if omitted:
            rows.append('...')
        return '\n'.join(rows)
    if style == 'math':
        return LatexExpression(_format_latex_tensor(expressions, shape))
    return _format_nested(expressions, shape)
