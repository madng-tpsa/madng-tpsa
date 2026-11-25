"""Tests for polynomial and tensor formatting."""

from typing import TYPE_CHECKING, cast

import pytest
from rich.console import Console

from madng_tpsa import (
    ComplexTpsa,
    Descriptor,
    Tpsa,
    TpsaMap,
    TpsaTensor,
    TpsaVector,
    TpsaVectorField,
    format_polynomial,
)
from madng_tpsa.formatting import LatexExpression

if TYPE_CHECKING:
    from madng_tpsa.formatting import FormatStyle


def _polynomial():
    desc = Descriptor(var_names=['x', 'y'], order=2)
    return Tpsa.from_dict(
        {
            '__class__': 'Tpsa',
            'descriptor': desc.to_dict(),
            'truncation_order': desc.order,
            '[0, 0]': -6.0,
            '[1, 0]': 2.0,
            '[0, 2]': 3.0,
        }
    )


def test_format_polynomial_as_code():
    t = _polynomial()

    assert format_polynomial(t, style='code') == '-6 + 2 * x + 3 * y**2'
    assert t.format(style='code') == '-6 + 2 * x + 3 * y**2'


def test_format_polynomial_as_math():
    t = _polynomial()
    formatted = t.format(style='math')

    assert isinstance(formatted, LatexExpression)
    assert isinstance(formatted, str)
    assert formatted.data == r'-6 + 2 x + 3 y^{2}'
    assert str(formatted) == formatted.data
    assert formatted._repr_latex_() == r'$\displaystyle -6 + 2 x + 3 y^{2}$'


def test_math_ipython_display():
    formatters = pytest.importorskip('IPython.core.formatters')
    t = _polynomial()
    for obj in (t, TpsaTensor([t]), TpsaMap([t, t])):
        formatted = obj.format(style='math')
        mime, _ = formatters.DisplayFormatter().format(formatted)
        assert mime['text/latex'] == formatted._repr_latex_()


def test_format_polynomial_as_table():
    t = _polynomial()
    table = t.format(style='table')

    assert [column.header for column in table.columns] == ['x', 'y', 'coefficient']
    assert table.rows[0].style is None


def test_format_polynomial_rejects_unknown_style():
    t = _polynomial()

    with pytest.raises(ValueError, match='Style'):
        t.format(style=cast('FormatStyle', 'bad'))


def test_unicode_formatting_of_complex_coefficients_in_tensor_and_map():
    d = Descriptor(var_names=('q', 'p'), order=3)
    q, p = d.vars()
    z = ComplexTpsa.from_tpsa(q)
    polynomial = (2 + 3j) * z**2 + (1 - 2j) * z + 1j * p - 4

    text = format_polynomial(polynomial, labels=d.var_names, style='unicode')
    assert '(2+3j)·q²' in text
    assert '(1-2j)·q' in text
    assert '(0+1j)·p' in text
    assert '-4' in text
    assert '(2+3j) * q**2' in format_polynomial(polynomial, labels=d.var_names, style='code')

    tensor = TpsaTensor([polynomial, q])
    mapping = TpsaMap([polynomial, p])
    for obj in (tensor, mapping):
        assert '(2+3j)·q²' in obj.format(max_terms=None)
        assert 'j)' in repr(obj)


def test_max_terms_zero_and_negative_limits():
    t = _polynomial()
    d = t.descriptor
    labels = d.var_names
    assert format_polynomial(t, labels=labels, style='unicode', max_terms=0) == '...'
    assert format_polynomial(t, labels=labels, style='code', max_terms=0) == '...'
    assert format_polynomial(d.zero(), labels=labels, style='unicode', max_terms=0) == '0'
    with pytest.raises(ValueError, match='max_terms'):
        format_polynomial(t, labels=labels, style='unicode', max_terms=-1)
    assert TpsaTensor([t]).format(max_tpsa=1, max_terms=0) == '[...]'


def test_tensor_text_structure_and_views():
    d = Descriptor(var_names=('x', 'y'), order=2)
    x, y = d.vars()
    tensor = TpsaTensor([[x, y**2], [x * y, -y]])
    assert tensor.format() == '[[x, y²],\n [x·y, -y]]'
    assert tensor.format(style='code') == '[[x, y**2],\n [x * y, -y]]'
    assert tensor.T.format() == '[[x, x·y],\n [y², -y]]'
    reversed_tensor = tensor[:, ::-1]
    assert isinstance(reversed_tensor, TpsaTensor)
    assert reversed_tensor.format() == '[[y², x],\n [-y, x·y]]'
    assert tensor.format(max_tpsa=0) == '[...]'
    assert tensor.format(max_tpsa=1) == '[[x, ...],\n ...]'
    assert tensor.format(max_tpsa=2) == '[[x, y²],\n ...]'
    assert tensor.format(max_tpsa=3) == '[[x, y²],\n [x·y, ...]]'
    rank3 = TpsaTensor([[[x, y]], [[y, x]]])
    assert rank3.format() == '[[[x, y]],\n\n [[y, x]]]'
    assert 'components=[x, y², x·y, ...]' in repr(tensor)


@pytest.mark.parametrize('container', [TpsaVector, TpsaMap, TpsaVectorField])
def test_named_vector_formatting(container):
    d = Descriptor(var_names=('x', 'px'), order=2)
    x, px = d.vars()
    vector = container([x + px**2, px])
    assert vector.format() == 'x = x + pₓ²\npₓ = pₓ'
    assert vector.format(style='code', max_tpsa=1) == 'x = x + px**2\n...'
    assert vector.format(max_tpsa=0) == '...'
    table = vector.format(style='table')
    assert [column.header for column in table.columns] == ['output', 'x', 'px', 'coefficient']
    assert list(table.columns[0].cells) == ['x', '', 'px']
    assert table.rows[1].end_section


def test_tensor_table_indices_and_rendering():
    d = Descriptor(var_names=('x', 'y'), order=2)
    x, y = d.vars()
    tensor = TpsaTensor([[x + y, y**2], [d.zero(), -x]])
    table = tensor.T.format(style='table')
    assert [column.header for column in table.columns] == ['i0', 'i1', 'x', 'y', 'coefficient']
    assert list(table.columns[0].cells) == ['0', '', '0', '1', '1']
    assert list(table.columns[1].cells) == ['0', '', '1', '0', '1']
    assert [str(cell) for cell in table.columns[-1].cells] == ['1', '1', '0', '1', '-1']
    assert table.rows[1].end_section
    assert table.rows[2].end_section
    console = Console(record=True, width=100)
    with console.capture() as capture:
        console.print(table)
    rendered = capture.get()
    assert 'coefficient' in rendered
    assert 'Table object' not in rendered


def test_table_truncation_and_zero():
    t = _polynomial()
    table = TpsaTensor([t, t.descriptor.zero()]).format(style='table', max_terms=0)
    assert [str(cell) for cell in table.columns[-1].cells] == ['…', '0']
    table = TpsaTensor([t, t]).format(style='table', max_tpsa=0)
    assert table.row_count == 0
    assert 'Showing 0 of 2' in str(table.caption)
    assert t.format(style='table', max_terms=1).row_count == 2


def test_math_tensor_vector_and_truncation():
    d = Descriptor(var_names=('x', 'px'), order=2)
    x, px = d.vars()
    tensor = TpsaTensor([[x, px**2], [x * px, -px]])
    math = tensor.format(style='math')
    assert isinstance(math, LatexExpression)
    assert math.data == r'\begin{pmatrix}x & p_{x}^{2} \\ x p_{x} & -p_{x}\end{pmatrix}'
    assert r'\vdots' in tensor.format(style='math', max_tpsa=2).data
    assert r'\cdots' in tensor.format(style='math', max_tpsa=1).data
    vector = TpsaMap([x + px**2, px])
    assert vector.format(style='math').data == (
        r'\begin{aligned}x &= x + p_{x}^{2} \\ p_{x} &= p_{x}\end{aligned}'
    )
    assert r'\vdots' in vector.format(style='math', max_tpsa=1).data
    assert r'\ldots' in (x + px).format(style='math', max_terms=1).data
    assert (x + px).format(style='math', max_terms=0).data == r'\ldots'
    rank3 = TpsaTensor([[[x, px]], [[px, x]]])
    assert r'T_{0,:,:}' in rank3.format(style='math').data
    assert r'T_{1,:,:}' in rank3.format(style='math').data


def test_explicit_tolerance_and_parameter_labels():
    d = Descriptor(var_names=('x', 'px'), param_names=('k_1',), order=2)
    x, _ = d.vars()
    k = d.param('k_1')
    t = 1e-15 * x + k
    assert '1e-15 * x' in t.format(style='code', tol=1e-16)
    assert t.format(style='code') == 'k_1'
    table = TpsaTensor([t]).format(style='table', tol=1e-16)
    assert [column.header for column in table.columns] == ['i0', 'x', 'px', 'k_1', 'coefficient']
    assert table.row_count == 2
    assert r'\mathrm{k\_1}' in t.format(style='math').data


@pytest.mark.parametrize('style', ['unicode', 'code', 'table', 'math'])
def test_container_format_validates_limits_even_without_components(style):
    t = _polynomial()
    tensor = TpsaTensor([t])
    with pytest.raises(ValueError, match='max_tpsa'):
        tensor.format(style=style, max_tpsa=-1)
    with pytest.raises(ValueError, match='max_terms'):
        tensor.format(style=style, max_tpsa=0, max_terms=-1)
