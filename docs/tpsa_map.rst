TpsaMap
=======

:class:`~madng_tpsa.map.TpsaMap` represents a full vector-valued TPSA map.
It subclasses :class:`~madng_tpsa.tensor.TpsaTensor` with one component per
*descriptor variable*. All components share the same descriptor; mixed real
and complex series are promoted to a complex map.

Construction and coordinate ordering
------------------------------------

.. code-block:: python

   from madng_tpsa import Descriptor, TpsaMap

   d = Descriptor(variables=('q', 'p'), parameters=('k',), order=5)
   q, p = d.vars()
   k = d.param('k')
   identity = TpsaMap(d)
   shifted = TpsaMap([q + k, p], descriptor=d)
   named = TpsaMap(p=p, q=q + k)  # reordered into descriptor variable order
   offset = TpsaMap(d, q=0.1)     # first output is q + 0.1

   assert named.coord_names == ('q', 'p')
   assert named.q == q + k
   assert named.p == p

Each map has exactly ``d.num_vars`` outputs, ordered according to
``d.variables``. The sequence constructor is **positional**; the keyword
constructor uses names and orders them by the descriptor. Unknown names are
rejected. Scalar component values are *offsets from the corresponding identity
coordinate*, not constant output polynomials. Missing keyword components retain
the identity. The descriptor can be inferred from TPSA inputs, or supplied
explicitly for scalar-only construction.

``map.coords`` returns the tuple of component TPSAs; ``map.coordinates`` is a
read-only *mapping of names to mutable TPSA objects*. Use ``map.keys()``,
``map.values()``, ``map.items()``, positional indexing (``map[0]``), or
``map.q`` for access. The names do not establish canonical conjugacy.

Conversion and inspection
-------------------------

.. code-block:: python

   named.const_part                # NumPy array of output constants
   named.coefficient('q', (1, 0, 1))
   named.set_coefficient('q', (1, 0, 1), 2.0)
   complex_map = named.to_complex()
   copy = named.copy()
   saved = named.to_dict()
   restored = TpsaMap.from_dict(saved)

``set_coefficient`` updates the selected component in place. ``const_part`` is
a writable array-shaped property; ``set_const_part`` is deprecated.

``change_descriptor`` converts a complete map to another compatible descriptor,
including the output coordinate ordering; ``extend_descriptor`` inserts identity
components for newly introduced variables. Both use descriptor variable names,
not merely the number of coordinates. ``homogeneous()``, ``truncate()`` and
``clear_order()`` operate on every component without modifying the original.

Evaluation, Jacobians, Hessians
------------------------------

.. code-block:: python

   m = TpsaMap([q**2 + k*q + p, q*p + k], descriptor=d)
   m.evaluate((2., 3.), parameters=(4.,))
   m.evaluate_jacobian((2., 3.), parameters=(4.,))
   J = m.jacobian(wrt='variables')  # Shape (2, 2), TPSA-valued
   H = m.hessian(wrt='all')         # Shape (2, 3, 3)
   values = m.evaluate_hessian((2., 3.), parameters=(4.,), wrt='all')

Each Jacobian and Hessian is evaluated in the original descriptor coordinates.

The inherited generic methods
:meth:`~madng_tpsa.tensor.TpsaTensor.partial_derivative` and
:meth:`~madng_tpsa.tensor.TpsaTensor.derivative_tensor` cover arbitrary mixed
partial derivatives and preserve parameter dependence. See :doc:`tpsa_tensor`
and ``examples/05_tpsamap_derivatives.py``.

Composition, translation and pullback
-------------------------------------

``left @ right`` and ``left.compose(right)`` mean :math:`left\circ right`.
Descriptor parameters are automatically carried through substitution unchanged:

.. math::

   f(q,p;k)\circ g(q,p;k) = f(g_q(q,p;k),g_p(q,p;k);k).

.. code-block:: python

   left = TpsaMap([q + k, p], descriptor=d)
   right = TpsaMap([2*q, p], descriptor=d)
   combined = left @ right          # (2*q + k, p)
   translated = left.translate((1., 0.))  # left(q + 1, p; k)
   observable = q**2 + p
   transformed = right.pullback(observable)  # observable composed with right

Translation shifts the **inputs** rather than adding to the outputs. Native
MAD-NG translation is used for parameter-free descriptors; with parameters,
composition with their identity substitutions preserves symbolic dependence.

Inversion
---------

.. code-block:: python

   m = TpsaMap([q + p**2, p], descriptor=d)
   m_inv = m.inverse()
   partly_inverted = m.partial_inverse((True, False))

These use the native formal inverse around an expansion point with zero
*constant coefficients*. The required linear block must be invertible. Note
that zero numerical constant coefficients do not exclude parameter-dependent
terms; parameters are treated symbolically. Inversion is not a general-purpose
numerical solver at an arbitrary point.

Hamiltonian fields and Lie maps
------------------------------

Hamiltonian operations assume an **even** number of descriptor variables,
ordered in canonical pairs ``(q1, p1, q2, p2, ...)``. Names are not checked for
physical meaning. Descriptor parameters are not canonical coordinates.

MAD-NG's native scalar-to-field convention is
:math:`G=-J\nabla f`, implemented by
:meth:`~madng_tpsa.map.TpsaMap.from_generator`. For Hamiltonian fields,
:meth:`~madng_tpsa.map.TpsaMap.to_generator` reverses this conversion up to
an additive function of the descriptor parameters. With its default
``validate=True`` it rejects vector fields that cannot be reconstructed from
a scalar generator at the available TPSA order.

.. code-block:: python

   f = k * q**3 / 3
   field = TpsaMap.from_generator(f)
   recovered = field.to_generator()     # f, with parameter-only terms removed
   lie_map = TpsaMap(d).exp_poisson(f)
   same_lie_map = TpsaMap(d).exp_poisson(-field)
   logarithmic_field = lie_map.log_poisson()
   logarithmic_scalar = lie_map.log_generator()

The *minus sign* in ``exp_poisson(-field)`` is deliberate: scalar inputs follow
MAD-NG's high-level exponential convention, whereas field inputs are passed
to the native exponential directly. A scalar field is not interchangeable
with the raw ``from_generator`` output without this sign change.

``lie_bracket()`` acts on vector fields; ``exp_poisson()`` and
``log_poisson()`` act on the corresponding Lie operators. The native logarithm
uses an iterative algorithm and is principally intended for maps close to the
identity. A normal return from MAD-NG does **not** prove that it converged.

By default, ``log_poisson(validate=True)`` checks the **full retained TPSA
coefficient map** after re-exponentiating the returned vector field. The
comparison uses ``atol=1e-12`` and ``rtol=1e-10`` by default, and raises
``ValueError`` if the map cannot be reconstructed at those tolerances. The
check does not require symplecticity or even a Hamiltonian vector field.

.. code-block:: python

   field = lie_map.log_poisson()  # Checks exp_poisson(field) == lie_map
   unchecked = lie_map.log_poisson(validate=False)

Use ``validate=False`` when a normal-form analysis deliberately performs its
**own** higher-level map reconstruction or tolerance and order comparison.
It avoids a potentially expensive extra exponential, but the caller then
owns convergence assessment. We deliberately keep ``validate=True`` as the
public API default, including in ``log_generator()``, which *also* checks
Hamiltonian integrability. Passing an ``initial_guess`` supplies the native
iteration with a vector-field guess, not a reference coordinate map.

Truncation can prevent exact generator recovery; a successful numerical
round trip is not a proof of existence or uniqueness of a global logarithm.

A TPSA map known to order :math:`N` generally allows verification of its
symplectic condition only through order :math:`N-1` because derivatives lower
polynomial order. ``TpsaMap`` supplies the algebraic primitives for normal-form
work, not eigenvector normalisation, resonance classification or RDT extraction.

Subclass behaviour
------------------

Generic map operations may produce a base ``TpsaMap`` instead of preserving
an arbitrary subclass whose auxiliary state has no defined transformation.

API reference
-------------

.. autoclass:: madng_tpsa.map.TpsaMap
   :members:
   :inherited-members:
