Formatting
==========

:func:`~madng_tpsa.formatting.format_polynomial` formats real and complex TPSA
coefficients as Unicode, code-like expressions, IPython mathematics, or Rich
tables. Unicode expressions render genuinely complex coefficients as, for
example, ``(2+3j)·q²``. The optional ``max_terms`` controls truncation of the
displayed terms (``None`` means no limit; ``0`` hides all terms).

Tensor formatting preserves the logical shape for text output. Named vectors,
vector fields and maps retain coordinate-labelled equations. The optional
``max_tpsa`` limits the displayed components in C-order; omitted components
and polynomial terms are indicated by ellipses.

Use ordinary ``print`` for Unicode, code, and coefficient tables,
and a compatible notebook frontend for mathematics::

    from IPython.display import display

    print(tensor.format(style='unicode'))
    print(tensor.format(style='code'))
    print(tensor.format(style='table'))
    display(tensor.format(style='math'))

Tables include output-coordinate or tensor-index columns. Math output uses
column vectors, matrices, or labelled matrix slices for higher ranks.
Math formatting returns a :class:`~madng_tpsa.formatting.LatexExpression`,
a string containing raw LaTeX that also implements the rich-display protocol.
It does not require IPython. A compatible frontend such as Jupyter renders
the expression; in a terminal, use ``print(tensor.format(style='math'))``
to inspect the raw LaTeX (also accessible through the ``data`` property).

.. autodata:: madng_tpsa.formatting.FormatStyle

.. autofunction:: madng_tpsa.formatting.format_polynomial

.. autofunction:: madng_tpsa.formatting.format_coeff

.. autofunction:: madng_tpsa.formatting.format_coord

.. autofunction:: madng_tpsa.formatting.to_superscript

.. autofunction:: madng_tpsa.formatting.to_subscript
