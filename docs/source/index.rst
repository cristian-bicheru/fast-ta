===============================
Fast TA |version| Documentation
===============================

Fast TA is an optimized, high-level technical analysis library used to 
compute technical indicators on financial datasets. It is written entirely
in C, and uses `AVX`_ vectorization as well. Fast TA is built with the `NumPy C API`_.

.. _AVX: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions
.. _NumPy C API: https://docs.scipy.org/doc/numpy/reference/c-api.html


Additional Support
==================
* Ask or search for questions in `StackOverflow using the fast-ta tag`_.
* Report bugs in our `issue tracker`_.

.. _StackOverflow using the fast-ta tag: https://stackoverflow.com/tags/fast-ta
.. _issue tracker: https://github.com/cristian-bicheru/fast-ta/issues


Introduction
============
.. toctree::
   :maxdepth: 2
   :caption: Introduction
   :hidden:

   misc/install

:doc:`misc/install`
   Get Fast TA installed on your computer.


Indicators
==========
.. toctree::
   :maxdepth: 2
   :caption: Indicators
   :hidden:

   api/momentum
   api/volume

:doc:`api/momentum`
   View the documentation for the momentum indicators.
:doc:`api/volume`
   View the documentation for the volume indicators.
:doc:`api/volatility`
   View the documentation for the volatility indicators.

Contributing
============
.. toctree::
   :maxdepth: 2
   :caption: Contributing
   :hidden:

   contributing/testing

:doc:`contributing/testing`
   Learn how to test and benchmark the Fast TA library.


License
=======
.. toctree::
   :maxdepth: 2
   :caption: License
   :hidden:

   misc/license

:doc:`misc/license`
   The Fast TA Library is licensed under the MIT License.

Credits
=======
Developed with passion by the `Fast TA Team`_.

.. _Fast TA Team: https://github.com/cristian-bicheru/fast-ta/graphs/contributors
