otn2d
============
M. M. Rams, M. Mohseni and B. Gardas, "Optimization and discovery of spin–glass droplets with approximate tensor networks contractions", 
https://arxiv.org/abs/1811.06518

**otn2d** is an open source package to heuristically solve optimization problems defined on quasi-2d lattice (including chimera graph).
It employs tensor network contractions to calculate marginal probabilites to find most probably states acconrding to Gibbs distribution.
Package can be also used for Random Markov Fields defined on 2d lattice.

Installation
-------------

.. code-block:: shell-session

   pip install .

Usage example
--------------

See folder \examples for scripts solving Ising problems defined on chimera graphs. 
See -h for some options, e.g.:

.. code-block:: shell-session

   python 01_search_gs_droplet.py -h

      
Documentation
--------------

Build using sphinx. 

.. code-block:: shell-session

    cd docs && make html

The generated documentation is found at `docs/build/html/index.html`