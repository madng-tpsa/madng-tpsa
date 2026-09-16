madng-tpsa
==========

Python bindings for MAD-NG's Generalised Truncated Power Series Algebra (GTPSA).
It provides real and complex truncated multivariate power series with a small,
Pythonic interface backed by the MAD-NG C library.

Quick start
-----------

Start by defining the algebraic space: its variables, optional parameters, and
maximum order. Series created from the same
:class:`~madng_tpsa.descriptor.Descriptor` can then be combined algebraically
and differentiated.

.. code-block:: python

   from madng_tpsa import Descriptor

   descriptor = Descriptor(variables=['x', 'y'], order=3)
   x, y = descriptor.vars()
   f = x**2 + 2 * y

   f.format()  # '2 * y + x**2'
   f.derivative('y')  # Tpsa({(0, 0): 2.0})
   f.monomial_coeffs()  # {(0, 1): 2.0, (2, 0): 1.0}

NumPy compatibility
-------------------

TPSA objects implement the NumPy ufunc protocol for supported elementary
functions, so ufuncs preserve the series rather than converting it to a scalar.
The same applies to supported SciPy special-function ufuncs. They return
:class:`~madng_tpsa.tpsa.Tpsa` objects for real inputs.

.. code-block:: python

   import numpy as np
   import scipy.special

   np.sin(f)  # A Tpsa containing the truncated sine series
   scipy.special.erf(f)  # A Tpsa containing the truncated error function

Complex series
---------------

:class:`~madng_tpsa.complex_tpsa.ComplexTpsa` has the corresponding
complex-coefficient API. Promote a real series with
:meth:`~madng_tpsa.complex_tpsa.ComplexTpsa.from_tpsa`, or use a function whose
mathematical result is complex. In particular, ``scipy.special.wofz`` promotes a
real TPSA result to :class:`~madng_tpsa.complex_tpsa.ComplexTpsa` to match SciPy's
scalar behaviour.

.. code-block:: python

   import scipy.special
   from madng_tpsa import ComplexTpsa

   z = ComplexTpsa.from_tpsa(x, y)  # x + 1j * y
   np.exp(z)  # A ComplexTpsa
   scipy.special.wofz(f)  # A ComplexTpsa

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   self
   api
