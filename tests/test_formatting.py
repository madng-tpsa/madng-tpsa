"""Tests for polynomial formatting."""

import pytest

import madng_tpsa


def _polynomial():
    desc = madng_tpsa.Descriptor(variables=['x', 'y'], order=2)
    t = madng_tpsa.Tpsa.from_dict(
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
    other = madng_tpsa.Descriptor(1, 2).zero()

    with pytest.raises(ValueError, match='different descriptor'):
        desc.format_polynomial(other)


def test_format_polynomial_rejects_unknown_style():
    _, t = _polynomial()

    with pytest.raises(ValueError, match='Style'):
        t.format(style='bad')
