otn2d
============
M. M. Rams, M. Mohseni and B. Gardas, "Optimization and discovery of spin–glass droplets with approximate tensor networks contractions", 
https://arxiv.org/abs/1811.06518

**otn2d** is an open-source package to heuristically solve Ising-type optimization problems defined on quasi-2d lattices, such, as e.g., the chimera graph.
It employs tensor network contractions to calculate marginal probabilities to identify the most probable states (according to Gibbs distribution).
It can also be used for Random Markov Fields defined on 2d lattice.

Installation
-------------

   pip install .

Usage example
--------------

See folder \examples for scripts solving Ising problems defined on the chimera graph. We include a set of hard _droplet instances_ defined on chimera graphs of sizes _L_=128,512,1152,2048 in folder \instances. For instance, to find the ground state of instance 1 for _L_=128 run:
   ```
   python 01_search_gs_droplet.py -L 128 -ins 1
   ```
To see some of the other available options run:
   ```
   python 01_search_gs_droplet.py -h
   ```
See the documentation for further details.

Documentation
--------------

Build using sphinx. 
   ```
   cd docs && make html
   ```
The generated documentation can be found at `docs/build/html/index.html`
