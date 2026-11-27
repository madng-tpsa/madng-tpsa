TpsaVector
==========

.. automodule:: madng_tpsa.vector

Polynomial substitution
-----------------------

:meth:`~madng_tpsa.vector.TpsaVector.substitute` substitutes input variables
and parameters in every vector component while preserving the vector type.
For a vector ``v``, use ``v.substitute(0.)`` to set all variables to zero,
``v.substitute(None, 0.)`` to set all parameters to zero, or named arguments
for individual inputs. See :doc:`tpsa_tensor` for the full substitution API.

API reference
-------------

.. autoclass:: madng_tpsa.vector.TpsaVector
   :members:
   :inherited-members:
