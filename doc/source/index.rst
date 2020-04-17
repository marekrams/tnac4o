.. otn2d documentation master file, created by
   sphinx-quickstart on Tue Mar 17 06:24:43 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to otn2d's documentation!
=================================

**otn2d** is an open-source package to heuristically solve Ising-type optimization problems defined on quasi-2d lattices, including, for instance, the chimera graph.
It employs tensor network contractions to calculate marginal probabilities and identify the most probable states according to Gibbs distribution.
By identifying spin-glass droplets, it allows reconstructing the low-energy spectrum of the model. It can also be used for Random Markov Fields defined on 2d lattice.

**otn2d** is based on the paper M. M. Rams, M. Mohseni and B. Gardas, *"Optimization and discovery of spin–glass droplets with approximate tensor networks contractions"*,
https://arxiv.org/abs/1811.06518


.. toctree::
   :maxdepth: 2
   :caption: Package description:

   otn2d
   aux

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
