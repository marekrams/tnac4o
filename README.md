Tensor network approximate contractions 4 optimalization (tnac4o)
============
M. M. Rams, M. Mohseni and B. Gardas, "Optimization and discovery of spin–glass droplets with approximate tensor networks contractions",
https://arxiv.org/abs/1811.06518

**tnac4o** is an open-source package to heuristically solve Ising-type optimization problems defined on quasi-2d lattices, including, for instance, the chimera graph.
It employs tensor network contractions to calculate marginal probabilities and identify the most probable states according to Gibbs distribution.
By identifying spin-glass droplets, it allows one to reconstruct the low-energy spectrum of the model.
It can also be used for Random Markov Fields defined on a 2d lattice.

Installation
------------
In the main folder run:
   ```
   pip install .
   ```
Make sure you are using python 3.6 or newer. 

Usage examples
--------------

See folder [examples](examples) for a set of scripts showing basic applications of the package.
Most of the examples concern with Ising problems defined on a chimera graph. We include a set of hard _droplet instances_ defined on chimera graphs of sizes _L_=128,512,1152,2048, see folder [instances](instances). 

For example, to find the ground state of instance number 1 for _L_=128 run:
   ```
   python e01_search_gs_droplet_instances.py -L 128 -ins 1
   ```
To see some of the other available options run:
   ```
   python e01_search_gs_droplet_instances.py -h
   ```
See the documentation for further details.

Other examples include:

Sampling from a Gibbs distribution, here at inverse temperature _beta_=1
   ```
   python e02_sample_droplet_instances.py -L 128 -ins 1 -b 1
   ```
   
Searching for a structure of low-energy excitations and saving the result to a file (in folder [examples/results](examples/results))
   ```
   python e03_search_spectrum_droplet_instances.py -L 128 -ins 1 -s
   ```

Loading the solution from the previous script, and reconstructing the low-energy states:
   ```
   python e04_load_spectrum_droplet_instances.py -L 128 -ins 1
   ```

Finally, for a minimal example of a problem defined as a Random Markov Field see:
   ```
   python e05_minimal_RMF.py
   ```

Documentation
-------------

Build using sphinx.
   ```
   cd doc && make html
   ```
The generated documentation can be found at `doc/build/html/index.html`
