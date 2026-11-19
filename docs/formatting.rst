Formatting
==========

:func:`~madng_tpsa.formatting.format_polynomial` formats real and complex TPSA
coefficients as Unicode, code-like expressions, IPython mathematics, or Rich
tables. Unicode expressions render genuinely complex coefficients as, for
example, ``(2+3j)·q²``. The optional ``max_terms`` controls truncation of the
displayed terms (``None`` means no limit; ``0`` hides all terms).

.. autodata:: madng_tpsa.formatting.FormatStyle

.. autofunction:: madng_tpsa.formatting.format_polynomial

.. autofunction:: madng_tpsa.formatting.format_coeff

.. autofunction:: madng_tpsa.formatting.format_coord

.. autofunction:: madng_tpsa.formatting.to_superscript

.. autofunction:: madng_tpsa.formatting.to_subscript
