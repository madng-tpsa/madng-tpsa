TpsaTensor
==========

:class:`~madng_tpsa.tensor.TpsaTensor` is an arbitrary-rank, row-major
container of real or complex TPSA components. A tensor's axes describe its
array shape; all polynomial components share one descriptor. A rank-one
``TpsaMap`` is a specialised subclass for full coordinate maps.

.. code-block:: python

   from madng_tpsa import Descriptor, TpsaTensor

   d = Descriptor(var_names=('q', 'p'), param_names=('k',), order=4)
   q, p = d.vars()
   k = d.param('k')
   tensor = TpsaTensor([[q**2 + k*p, p], [q*p, k*q]])
   assert tensor.shape == (2, 2)

   partial = tensor.partial_derivative('q', 'k')
   assert partial.shape == (2, 2)
   hessian = tensor.derivative_tensor(2, wrt='all')
   assert hessian.shape == (2, 2, 3, 3)

Derivatives
-----------

``partial_derivative(*wrt, orders=None)`` preserves the tensor's shape and class.
Each positional argument specifies **one direction** (a name, 1-based index,
or a unit first-order real/complex TPSA), or a ``(direction, multiplicity)``
pair. For ``(q, p; k)``, these calls all mean
:math:`\partial^3 / \partial q^2\partial k`:

.. code-block:: python

   tensor.partial_derivative(('q', 2), 'k')
   tensor.partial_derivative((1, 2), 3)
   tensor.partial_derivative(q, q, k)
   tensor.partial_derivative(orders=(2, 0, 1))

An exponent vector is supplied **only** through ``orders=``; a two-integer
positional tuple means ``(1-based index, multiplicity)`` instead. You cannot
mix ``orders=`` and positional directions. Invalid names, negative
multiplicities, booleans, and unsupported specifications raise errors.
Derivatives beyond the descriptor's representable order evaluate to zero.

``derivative_tensor(order, wrt='vars')`` appends
``order`` derivative axes, each of length equal to the number of selected
directions (specified with ``wrt``, which is ``'vars'``,
``'params'``, or ``'all'``). Original axes are left unchanged. When
``wrt='all'``, variables precede parameters in each derivative axis.

For performance and storage, the result has ``tensor.size * n**order``
component TPSAs, so high ranks can be expensive even when higher derivatives
vanish. The default ``max_components=1_000_000`` prevents unexpectedly
allocating larger tensors; a call may opt out with ``max_components=None``
only when memory requirements have been assessed. A lower limit can be
supplied for memory-sensitive work. The guard counts *TPSA objects*, not
coefficient storage: even a smaller number of high-order series can use
substantial memory.

Storage and conversion
----------------------

``tensor[i]`` returns a lower-rank *view* (sharing the underlying mutable TPSA
objects); ``tensor[i, j]`` may return a single TPSA. ``flat`` returns the
flattened components of the current tensor or view; ``shape``, ``rank`` and
``size`` describe that view. ``copy()`` returns independently allocated
components.

.. code-block:: python

   row = tensor[0]              # Shared view of the first row
   snapshot = row.copy()        # Independent tensor
   data = tensor.to_dict()      # JSON-friendly coefficient payload
   restored = TpsaTensor.from_dict(data)
   complex_tensor = tensor.to_complex()
   evaluated = tensor.evaluate((1., 2.), param_values=(3.,))

``const_part`` can be read or assigned using scalar broadcasting or a NumPy
array matching ``shape``. ``homogeneous()``, ``truncate()`` and
``clear_order()`` return new tensors. ``change_descriptor()`` converts component
polynomials to a compatible descriptor.

``compose(map)`` substitutes the full coordinate map into every component,
retaining original tensor shape. Descriptor parameters are carried through as
identity substitutions. The operator ``<<`` represents regular composition
(one can imagine it as the opening of the map's round brackets), while the
operator ``>>`` represents Dragt-style notation. The following are possible,
for ``T`` a tensor of general rank, ``M`` a map, and ``f`` a scalar function:

.. code-block:: python

   T << M  #    T(M)
   f << M  #    f(M)  (pullback defined on M)
   N << M  #    N(M)

   M >> T  #    T(M)
   M >> f  #    f(M)  (pullback defined on M)
   M >> N  #    N(M)

Tensor ``+``, ``-``, scalar ``*`` and scalar ``/`` are
componentwise operations.


Polynomial substitution
-----------------------

``substitute(var_subs=None, param_subs=None, /, **substitutions)`` simultaneously
replaces descriptor variables and parameters in every tensor component. It
returns a new tensor with the original shape and concrete built-in container
type.

The first two arguments are positional-only and refer to the input variables
and parameters, respectively, not to the tensor's output components. Each
argument accepts:

* ``None``: leave that entire group unchanged.
* A scalar or a single real or complex TPSA: broadcast to every input in the
  group. Scalars represent constant polynomials, not offsets.
* A sequence or one-dimensional NumPy array: one scalar or TPSA per input,
  in the original descriptor's ``var_names`` or ``param_names`` order.
  The length must equal ``d.num_vars`` or ``d.num_params``, respectively.

.. code-block:: python

   import numpy as np

   variables_zero = tensor.substitute(0.)      # Keep parameter dependence
   nominal = tensor.substitute(None, 0.)       # Keep variable dependence
   fixed = tensor.substitute(0., 1.)           # Fix both groups
   constants = tensor.substitute(0., 0.)       # Constant TPSAs, not an array
   rescaled = tensor.substitute([2*q, p])      # Substitute all variables
   shifted = tensor.substitute(None, [k+0.1])  # Substitute all parameters
   broadcast = tensor.substitute(q)            # Both input variables become q
   array_input = tensor.substitute(np.zeros(d.num_vars))

Alternatively, use keyword arguments to replace individual named variables
or parameters. All unspecified inputs retain their identity substitutions.

.. code-block:: python

   nominal = tensor.substitute(k=0)
   shifted = tensor.substitute(k=k+0.1)
   rescaled = tensor.substitute(q=2*q, k=0.5)
   swapped = tensor.substitute(q=p, p=q)

Substitutions are simultaneous, not sequential: both ``tensor.substitute(p=q,
q=p)`` and ``tensor.substitute([p, q])`` exchange the two input coordinates.
Non-``None`` positional groups cannot be combined with named substitutions
(``TypeError``). Incorrect group lengths or array dimensions raise
``ValueError``. The positional-only arguments leave all descriptor labels
available for named substitution, even labels such as ``var_subs``.

Substitution TPSAs must have the same set of variable labels as the original
descriptor; different label ordering is supported. Compatible descriptors are
unified, including additional parameters introduced by substitutions, and
complex substitutions promote real tensors to complex tensors. Unknown
substitution names raise ``KeyError``.

Unlike ``evaluate()``, this operation returns symbolic TPSA polynomials.
Unlike ``compose()``, it allows explicit substitution of parameters.
Setting a parameter to zero removes its polynomial dependence, but does not
remove the parameter from the descriptor. Use ``change_descriptor()`` to
remove parameters that are no longer needed.

All results remain subject to the common descriptor's truncation limits.
Calling ``substitute()`` or ``substitute(None, None)`` returns an independent
copy. Broadcasting over an empty group (for example, a descriptor without
parameters) is a no-op. An empty sequence is valid only for an empty group.


Strided views and tensor contractions
-------------------------------------

Basic slicing, negative steps, ``None``, ellipsis, ``transpose()``,
``swapaxes()`` and ``.T`` produce shared-storage views without copying TPSAs.
The ``flat`` property enumerates a view in logical row-major order, even
when its backing storage is noncontiguous. Native kernels receive a packed
pointer array for noncontiguous views. Cache entries are invalidated when
components are replaced. ``x in tensor`` tests *object identity*, not
polynomial equality.

Use ``@`` for rank-one/rank-two matrix multiplication, ``inner()`` for
last-axis contraction, ``trace()`` for a diagonal sum, ``tensordot()`` to
select paired axes and ``einsum('ij,jk->ik', A, B)`` for explicit-index
contractions. ``einsum`` accepts only explicit ASCII single-letter subscripts
(no ellipsis or batched broadcasting in this implementation).
A complete contraction returns a scalar TPSA instead of a rank-zero tensor.

API reference
-------------

.. autoclass:: madng_tpsa.tensor.TpsaTensor
   :members:
   :inherited-members:
   :special-members: __getitem__, __setitem__, __len__, __iter__, __contains__, __neg__, __add__, __sub__, __mul__, __truediv__, __matmul__, __lshift__, __rrshift__
