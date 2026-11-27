Examples
========

The repository contains runnable examples in ``examples``. Run one with, for
example:

.. code-block:: console

   python examples/01_tpsa_basics.py

``01_tpsa_basics.py`` covers construction, arithmetic, coefficients, order
operations and formatting.

.. literalinclude:: ../examples/01_tpsa_basics.py
   :language: python

``02_tpsa_calculus_and_poisson.py`` covers differentiation, parameters,
integration and Poisson brackets.

.. literalinclude:: ../examples/02_tpsa_calculus_and_poisson.py
   :language: python

``03_tpsamap_basics.py`` covers map construction, evaluation, Jacobians and
coefficient manipulation.

.. literalinclude:: ../examples/03_tpsamap_basics.py
   :language: python

``04_tpsamap_composition_and_lie.py`` covers composition, inversion,
parameter-dependent maps, named and positional substitutions, and Lie operations.

.. literalinclude:: ../examples/04_tpsamap_composition_and_lie.py
   :language: python

``05_tpsamap_derivatives.py`` covers map and tensor Jacobians, Hessians,
parameter derivatives and mixed partials.

.. literalinclude:: ../examples/05_tpsamap_derivatives.py
   :language: python

``06_public_api_walkthrough.py`` surveys conversion, serialization, views,
mathematical functions, tensor substitutions (including NumPy arrays), and
additional public map/tensor operations.

.. literalinclude:: ../examples/06_public_api_walkthrough.py
   :language: python

``07_log_fgrad_validation.py`` demonstrates validated general vector-field
logarithms, and distinguishes them from Hamiltonian generators.

.. literalinclude:: ../examples/07_log_fgrad_validation.py
   :language: python
