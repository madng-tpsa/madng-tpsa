TpsaTensor
==========

:class:`~madng_tpsa.tensor.TpsaTensor` is an arbitrary-rank, row-major
container of real or complex TPSA components. A tensor's axes describe its
array shape; all polynomial components share one descriptor. A rank-one
``TpsaMap`` is a specialised subclass for full coordinate maps.

.. code-block:: python

   from madng_tpsa import Descriptor, TpsaTensor

   d = Descriptor(variables=('q', 'p'), parameters=('k',), order=4)
   q, p = d.vars()
   k = d.param('k')
   tensor = TpsaTensor([[q**2 + k*p, p], [q*p, k*q]])
   assert tensor.shape == (2, 2)

   partial = tensor.partial_derivative((('q', 1), 'k'))
   assert partial.shape == (2, 2)
   hessian = tensor.derivative_tensor(2, wrt='all')
   assert hessian.shape == (2, 2, 3, 3)

Derivatives
-----------

``partial_derivative(spec)`` preserves the input tensor's shape and class.
Directions can be specified using names, one-based integer indices, a
representable single-monomial TPSA, or a full multi-index of exponents. A
sequence of ``(direction, multiplicity)`` pairs and individual names is also
accepted. For ``(q, p; k)``, all of the following describe
:math:`\partial^3 / \partial q^2\partial k`:

.. code-block:: python

   tensor.partial_derivative((('q', 2), 'k'))
   tensor.partial_derivative(((1, 2), 3))
   tensor.partial_derivative((2, 0, 1))

A bare all-integer sequence is interpreted as a **full exponent vector**, not
a sequence of coordinate indices. Indices used as individual directions are
one-based in descriptor variable/parameter order. Invalid names, lengths,
negative orders, booleans, or unsupported specifications raise an exception.
Derivatives beyond the descriptor's representable order evaluate to zero.

``derivative_tensor(order, wrt='variables')`` appends
``order`` derivative axes, each of length equal to the number of selected
directions (specified with ``wrt``, which is  ``'variables'``,
``'parameters'``, or ``'all'``). Original axes are left unchanged. When
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
   evaluated = tensor.evaluate((1., 2.), parameters=(3.,))

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
