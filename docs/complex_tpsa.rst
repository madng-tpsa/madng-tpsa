ComplexTpsa
===========

:class:`~madng_tpsa.complex_tpsa.ComplexTpsa` is the complex-coefficient
counterpart of :class:`~madng_tpsa.tpsa.Tpsa`.

.. code-block:: python

   from madng_tpsa import ComplexTpsa, Descriptor

   descriptor = Descriptor(variables=['x', 'y'], order=4)
   x, y = descriptor.vars()

   z = ComplexTpsa.from_tpsa(x, y)   # x + 1j*y
   f = z**2 + 2*x

Mixed real/complex operations promote their result when required.

Complex TPSAs support the same coefficient, calculus, parameter-gradient,
Poisson-bracket, and order-manipulation concepts as real TPSAs.

A useful application is the complex phasor representation used in nonlinear
normal-form calculations.

In-place mutation is different from promotion: an existing real native TPSA
cannot silently become a complex native TPSA. APIs mutating a real map should
therefore reject genuinely complex coefficients unless the map is first
promoted.

API reference
-------------

.. autoclass:: madng_tpsa.complex_tpsa.ComplexTpsa
   :members:
   :inherited-members:
   :special-members: __neg__, __add__, __sub__, __mul__, __truediv__, __pow__, __getitem__, __setitem__
