import logging
import numpy as np
from otn2d import otn2d


def main():
    """
    Runs a script with a minimal example defining RMF.
    """

    # defining a model
    Nx = 5
    Ny = 3
    N = np.zeros((3,5), dtype=int)+3
    fun = {1: np.array([[0, 1, 1],[1, 0, 1],[1, 1, 0]]),  ## penalty for neighbouring spins differing
           2: np.array([-1.5, 0, 1.5]),
           3: np.array([1.25, 0, -1.25])}

    # fac = { (ny1, nx1, ny2, nx2): fun index,  (ny, nx): fun index}
    # dictionary of couplings
    fac = {(0, 0, 0, 1):1, (0, 1, 0, 2):1, (0, 2, 0, 3):1, (0, 3, 0, 4):1,
           (1, 0, 1, 1):1, (1, 1, 1, 2):1, (1, 2, 1, 3):1, (1, 3, 1, 4):1,
           (2, 0, 2, 1):1, (2, 1, 2, 2):1, (2, 2, 2, 3):1, (2, 3, 2, 4):1,
           (0, 0, 1, 0):1, (1, 0, 2, 0):1,
           (0, 1, 1, 1):1, (1, 1, 2, 1):1,
           (0, 2, 1, 2):1, (1, 2, 2, 2):1,
           (0, 3, 1, 3):1, (1, 3, 2, 3):1,
           (0, 4, 1, 4):1, (1, 4, 2, 4):1,
           (0, 0):2, (0, 1):2, (0, 2):2, (0, 3):2, (0, 4):2,
           (1, 0):3, (1, 1):3, (1, 2):3, (1, 3):3, (1, 4):3,
           (2, 0):2, (2, 1):2, (2, 2):2, (2, 3):2, (2, 4):2}
    J = {'fun':fun, 'fac': fac, 'N':N, 'Nx':Nx, 'Ny':Ny}

    logging.basicConfig(level='INFO')

    ins = otn2d.otn2d(mode='RMF', Nx=Nx, Ny=Ny, J=J, beta=4)
    # search low enegy spectrum
    Eng = ins.search_low_energy_spectrum(M=128, relative_P_cutoff=1e-8, Dmax=32, max_dEng=3.1)
    # decodes it into states
    Eng = ins.decode_low_energy_states(max_dEng=3.1, max_states=100)
    return ins


if __name__ == "__main__":
    ins=main()
    ins.show_solution(state=False)
    print('Energies of the states found:')
    print(ins.energy) 

