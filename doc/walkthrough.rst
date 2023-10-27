Walkthrough the package
=======================

The fglib2 package contains two modules.
The first one is the :mod:`fglib2.distributions` module, which contains tabular distributions over `discrete random
variables. <https://random-events.readthedocs.io/en/latest/#variables>`_. The second one is the :mod:`fglib2.graphs`
which contains the definition of factor graphs.

Distributions
-------------
The :mod:`fglib2.distributions` module contains the :class:`fglib2.distributions.Multinomial` class, which implements a
tabular distribution over discrete random variables. The following example highlights the main features of the class.


.. nbgallery::

    notebooks/distributions

