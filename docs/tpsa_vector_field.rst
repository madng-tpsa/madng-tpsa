TpsaVectorField
===============

:class:`~madng_tpsa.vector.TpsaVectorField` represents a TPSA Hamiltonian
vector field. It subclasses :class:`~madng_tpsa.vector.TpsaVector` with one
component per *descriptor variable*. All components share the same descriptor;
mixed real and complex series are promoted to a complex vector.

Construction and coordinate ordering
------------------------------------

.. code-block:: python

   from madng_tpsa import Descriptor, TpsaVectorField

   d = Descriptor(variables=('q', 'p'), parameters=('k',), order=5)
   q, p = d.vars()
   k = d.param('k')
   identity = TpsaVectorField(d)
   shifted = TpsaVectorField([q + k, p], descriptor=d)
   named = TpsaVectorField(p=p, q=q + k)  # reordered into descriptor variable order
   offset = TpsaVectorField(d, q=0.1)     # first output is 0.1

   assert named.component_names == ('q', 'p')
   assert named.q == q + k
   assert named.p == p

Each field has exactly ``d.num_vars`` outputs, ordered according to
``d.variables``. The sequence constructor is **positional**; the keyword
constructor uses names and orders them by the descriptor. Unknown names are
rejected. Scalar component values are *constant output polynomials*. Missing
keyword components are set to zero. The descriptor can be inferred from TPSA
inputs, or supplied explicitly for scalar-only construction.

``field.components`` returns the tuple of component TPSAs; ``field.component_names``
returns the tuple of coordinates; ``field.named_components`` is a read-only
*mapping of names to mutable TPSA objects*. Use positional indexing
(``field[0]``) or ``field.q`` for access. The names do not establish canonical
conjugacy.

Conversion and inspection
-------------------------

.. code-block:: python

   named.const_part                # NumPy array of output constants
   named.coefficient('q', (1, 0, 1))
   named.set_coefficient('q', (1, 0, 1), 2.0)
   complex_field = named.to_complex()
   copy = named.copy()
   saved = named.to_dict()
   restored = TpsaVectorField.from_dict(saved)

``set_coefficient`` updates the selected component in place. ``const_part`` is
a writable array-shaped property.

``change_descriptor`` converts a complete field to another compatible descriptor,
including the output coordinate ordering; ``extend_descriptor`` inserts identity
components for newly introduced variables. Both use descriptor variable names,
not merely the number of named_components. ``homogeneous()``, ``truncate()`` and
``clear_order()`` operate on every component without modifying the original.

Evaluation, Jacobians, Hessians
-------------------------------

.. code-block:: python

   m = TpsaVectorField([q**2 + k*q + p, q*p + k], descriptor=d)
   m.evaluate((2., 3.), parameters=(4.,))  # array([15., 10.])
   m.evaluate_jacobian((2., 3.), parameters=(4.,))
   # array([[8., 1.],
   #        [3., 2.]])
   J = m.jacobian(wrt='variables')  # Shape (2, 2), TPSA-valued
   print(J)
   # TpsaTensor(shape=(2, 2), size=4, values=[2·q + k, 1, p, ...])
   H = m.hessian(wrt='all')         # Shape (2, 3, 3), TPSA-valued
   values = m.evaluate_hessian((2., 3.), parameters=(4.,), wrt='all')

Each Jacobian and Hessian is evaluated in the original descriptor named_components.

The inherited generic methods
:meth:`~madng_tpsa.tensor.TpsaTensor.partial_derivative` and
:meth:`~madng_tpsa.tensor.TpsaTensor.derivative_tensor` cover arbitrary mixed
partial derivatives and preserve parameter dependence. See :doc:`tpsa_tensor`
and ``examples/05_tpsamap_derivatives.py``.

Translation
-----------

.. code-block:: python

   field = TpsaVectorField([q + k, p], descriptor=d)
   translated = field.translate((1., 0.))  # field(q + 1, p; k)

Translation shifts the **inputs** rather than adding to the outputs. Native
MAD-NG translation is used for parameter-free descriptors; with parameters,
composition with their identity substitutions preserves symbolic dependence.

Hamiltonian fields and Lie maps
-------------------------------

Hamiltonian operations assume an **even** number of descriptor variables,
ordered in canonical pairs ``(q1, p1, q2, p2, ...)``. Names are not checked for
physical meaning. Descriptor parameters are not canonical named_components.

MAD-NG's native scalar-to-field convention is
:math:`G=-S\nabla f`, implemented by
:meth:`~madng_tpsa.vector.TpsaVectorField.from_generator`. For Hamiltonian fields,
:meth:`~madng_tpsa.vector.TpsaVectorField.to_generator` reverses this conversion
up to an additive function of the descriptor parameters. With its default
``validate=True`` it rejects vector fields that cannot be reconstructed from
a scalar generator at the available TPSA order.

.. code-block:: python

   f = k * q**3 / 3
   field = TpsaVectorField.from_generator(f)
   recovered = field.to_generator()     # f, with parameter-only terms removed

``lie_bracket()`` acts on vector fields.

API reference
-------------

.. autoclass:: madng_tpsa.vector.TpsaVectorField
   :members:
   :inherited-members:
   :special-members: __getitem__, __setitem__, __getattr__, __len__, __iter__, __contains__, __neg__, __add__, __sub__, __mul__, __truediv__, __matmul__, __lshift__, __rshift__, __xor__
