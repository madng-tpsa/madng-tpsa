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
   print(f)
   # Tpsa({(0, 0, 0, 0): 2.0, (0, 1, 0, 0): 1.5, (0, 0, 1, 0): 1.5,
   #       (0, 0, 0, 1): 1.0, (1, 1, 0, 0): 1.0, (1, 0, 1, 0): 1.0})
   print(f.const_part)  # 2.0
   print(f[(1, 1, 0, 0)])  # 1.0
   print(f.coefficient((1, 1, 0, 0)))  # 1.0
   print(f.monomial_coeffs())
   # {(0, 0, 0, 0): 2.0, (0, 1, 0, 0): 1.5, (0, 0, 1, 0): 1.5,
   #  (0, 0, 0, 1): 1.0, (1, 1, 0, 0): 1.0, (1, 0, 1, 0): 1.0}
   print(f.to_dict())
   # {'__class__': 'Tpsa',
   #  'descriptor': {'__class__': 'Descriptor',
   #   'variables': ('x', 'px'),
   #   'parameters': ('k1', 'k2'),
   #   'order': 4,
   #   'param_order': 2,
   #   'max_orders': None},
   #  'truncation_order': 4,
   #  '[0, 0, 0, 0]': 2.0,
   #  '[0, 1, 0, 0]': 1.5,
   #  '[0, 0, 1, 0]': 1.5,
   #  '[0, 0, 0, 1]': 1.0,
   #  '[1, 1, 0, 0]': 1.0,
   #  '[1, 0, 1, 0]': 1.0}

Stored numbers are Taylor coefficients, not raw higher derivatives.

Invalid monomial lengths and monomials outside the descriptor or allocated
TPSA order are rejected before entering the native GTPSA API.

Differentiation and parameter derivatives
-----------------------------------------

.. code-block:: python

   print(f.derivative('x'))
   # Tpsa({(0, 1, 0, 0): 1.0, (0, 0, 1, 0): 1.0})
   print(f.derivative('k1'))
   # Tpsa({(0, 0, 0, 0): 1.5, (1, 0, 0, 0): 1.0})
   print(f.grad())  # [0.0, 1.5]
   print(f.param_grad())  # [1.5, 1.0]

``grad()`` and ``param_grad()`` are first-order coefficients at the expansion
point. They are not replacements for ``derivative(...)`` when the full
polynomial derivative is required.

Order operations
----------------

.. code-block:: python

   print(f.homogeneous(3))  # Tpsa({})
   print(f.truncate(3))
   # Tpsa({(0, 0, 0, 0): 2.0, (0, 1, 0, 0): 1.5, (0, 0, 1, 0): 1.5,
   #       (0, 0, 0, 1): 1.0, (1, 1, 0, 0): 1.0, (1, 0, 1, 0): 1.0})
   print(f.clear_order(2))
   # Tpsa({(0, 0, 0, 0): 2.0, (0, 1, 0, 0): 1.5, (0, 0, 1, 0): 1.5,
   #       (0, 0, 0, 1): 1.0})

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
