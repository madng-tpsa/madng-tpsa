Tpsa
====

A :class:`~madng_tpsa.tpsa.Tpsa` is a real truncated multivariate Taylor
series backed by MAD-NG's GTPSA implementation.

Descriptors, variables and parameters
-------------------------------------

Every TPSA belongs to one :class:`~madng_tpsa.descriptor.Descriptor`.
The descriptor fixes the independent variables, optional external parameters,
and truncation orders.

.. code-block:: python

   descriptor = Descriptor(
       variables=['x', 'px'],
       order=4,
       parameters=['k1', 'k2'],
       param_order=2,
   )

   x, px = descriptor.vars()
   k1, k2 = descriptor.params()

Full monomial tuples contain variable exponents followed by parameter
exponents. For the descriptor above, ``(2, 1, 0, 1)`` means
:math:`x^2 p_x k_2`.

Expansion values
----------------

.. code-block:: python

   x = descriptor.var('x', 1.5)

The resulting TPSA has constant part ``1.5`` and first-order coefficient one in
``x``. Formal TPSA variables can therefore be interpreted as deviations from
the selected expansion point.

Coefficients
------------

.. code-block:: python

   f = 2 + x*px + k1*x + k2
   f.const_part
   f[(1, 1, 0, 0)]
   f.coefficient((1, 1, 0, 0))
   f.monomial_coeffs()
   f.to_dict()

Stored numbers are Taylor coefficients, not raw higher derivatives.

Invalid monomial lengths and monomials outside the descriptor or allocated
TPSA order are rejected before entering the native GTPSA API.

Differentiation and parameter derivatives
-----------------------------------------

.. code-block:: python

   dfdx = f.derivative('x')
   dfdk1 = f.derivative('k1')
   variable_gradient = f.grad()
   parameter_gradient = f.param_grad()

``grad()`` and ``param_grad()`` are first-order coefficients at the expansion
point. They are not replacements for ``derivative(...)`` when the full
polynomial derivative is required.

Order operations
----------------

.. code-block:: python

   cubic = f.homogeneous(3)
   through_cubic = f.truncate(3)
   without_quadratic = f.clear_order(2)

``truncation_order`` is the allocated order; ``degree`` is the highest order
currently populated by a non-zero coefficient.

Poisson brackets
----------------

For variables ordered ``(q1, p1, q2, p2, ...)``:

.. code-block:: python

   descriptor = Descriptor(variables=['q', 'p'], order=5)
   q, p = descriptor.vars()
   assert q.poisson_bracket(p).const_part == 1

Descriptor parameters are excluded from the canonical pairs.

Formatting
----------

.. code-block:: python

   f.format('code')
   f.format('math')
   f.format('table')

``'math'`` returns an ``IPython.display.Math`` object intended for notebooks.

API reference
-------------

.. autoclass:: madng_tpsa.tpsa.Tpsa
   :members:
