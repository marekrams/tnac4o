.. tnac4o documentation master file, created by
   sphinx-quickstart on Tue Mar 17 06:24:43 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to tnac4o's documentation!
==================================

**Tensor network approximate contractions for optimalization (tnac4o)** is an open-source package to heuristically solve Ising-type optimization problems defined on quasi-2d lattices, including, for instance, the chimera graph.
It employs tensor network contractions to calculate marginal probabilities and identify the most probable states according to Gibbs distribution.
By identifying spin-glass droplets, it allows reconstructing the low-energy spectrum of the model. It can also be used for Random Markov Fields defined on 2d lattice.

**tnac4o** is based on the paper M. M. Rams, M. Mohseni, D. Eppens, K. Jałowiecki and B. Gardas, 
*"Approximate optimization, sampling and spin-glass droplets discovery with tensor networks"*, `Phys. Rev. E 104, 025308 (2021) <https://doi.org/10.1103/PhysRevE.104.025308>`_, `arXiv:1811.06518 <https://arxiv.org/abs/1811.06518>`_

.. toctree::
   :maxdepth: 2
   :caption: Package description:

   tnac4o
   aux

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
