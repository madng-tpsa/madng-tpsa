"""Tests for polynomial formatting."""

import pytest

from madng_tpsa import ComplexTpsa, Descriptor, Tpsa, TpsaMap, TpsaTensor, format_polynomial


def _polynomial():
    desc = Descriptor(variables=['x', 'y'], order=2)
    t = Tpsa.from_dict(
        {
            '__class__': 'Tpsa',
            'descriptor': desc.to_dict(),
            'truncation_order': desc.order,
            '[0, 0]': -6.0,
            '[1, 0]': 2.0,
            '[0, 2]': 3.0,
        }
    )
    return desc, t


def test_format_polynomial_as_code():
    desc, t = _polynomial()

    assert desc.format_polynomial(t, style='code') == '-6 + 2 * x + 3 * y**2'
    assert t.format() == '-6 + 2 * x + 3 * y**2'


def test_format_polynomial_as_math():
    ipython_display = pytest.importorskip('IPython.display')

    _, t = _polynomial()
    formatted = t.format(style='math')

    assert isinstance(formatted, ipython_display.Math)
    assert formatted.data == r'-6 + 2 x + 3 y^{2}'


def test_format_polynomial_as_table():
    rich_text = pytest.importorskip('rich.text')

    _, t = _polynomial()
    table = t.format(style='table')

    assert [column.header for column in table.columns] == ['x', 'y', 'coefficient']
    assert table.rows[0].style is None
    assert isinstance(table.columns[-1]._cells[0], rich_text.Text)
    assert table.columns[-1]._cells[0].style == 'cyan'


def test_format_polynomial_rejects_different_descriptor():
    desc, _ = _polynomial()
    other = Descriptor(1, 2).zero()

    with pytest.raises(ValueError, match='different descriptor'):
        desc.format_polynomial(other)


def test_format_polynomial_rejects_unknown_style():
    _, t = _polynomial()

    with pytest.raises(ValueError, match='Style'):
        t.format(style='bad')


def test_unicode_formatting_of_complex_coefficients_in_tensor_and_map():
    d = Descriptor(variables=('q', 'p'), order=3)
    q, p = d.vars()
    z = ComplexTpsa.from_tpsa(q)
    polynomial = (2 + 3j) * z**2 + (1 - 2j) * z + 1j * p - 4

    text = format_polynomial(polynomial, d.variables, style='unicode')
    assert '(2+3j)·q²' in text
    assert '(1-2j)·q' in text
    assert '(0+1j)·p' in text
    assert '-4' in text
    assert '(2+3j) * q**2' in format_polynomial(polynomial, d.variables, style='code')

    tensor = TpsaTensor([polynomial, q])
    mapping = TpsaMap([polynomial, p])
    for obj in (tensor, mapping):
        assert '(2+3j)·q²' in obj.str(max_terms=None)
        assert 'j)' in repr(obj)


def test_max_terms_zero_and_negative_limits():
    d, t = _polynomial()
    labels = d.variables
    assert format_polynomial(t, labels, style='unicode', max_terms=0) == '...'
    assert format_polynomial(t, labels, style='code', max_terms=0) == '...'
    assert format_polynomial(d.zero(), labels, style='unicode', max_terms=0) == '0'
    with pytest.raises(ValueError, match='max_terms'):
        format_polynomial(t, labels, style='unicode', max_terms=-1)
    assert 'values=[, ...]' not in TpsaTensor([t]).str(max_tpsa=1, max_terms=0)
