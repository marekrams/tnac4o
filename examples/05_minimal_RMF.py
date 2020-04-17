import logging
import argparse
import numpy as np
import otn2d


def main(rot=0, beta=3,
         D=40,
         M=1024, relative_P_cutoff=1e-12,
         excitations_encoding=1,
         dE=1., hd=0,
         max_states=100,
         precondition=False):
    """
    Runs a minimal example of Random Markov Field.
    """

    # defining a model
    Nx = 5
    Ny = 3
    N = np.zeros((3, 5), dtype=int)+3
    fun = {1: np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),  # penalty for neighbouring spins differing
           2: np.array([-1.5, 0, 1.5]),
           3: np.array([1.25, 0, -1.25])}

    # fac = { (ny1, nx1, ny2, nx2): fun index,  (ny, nx): fun index}
    # dictionary of couplings
    fac = {(0, 0, 0, 1): 1, (0, 1, 0, 2): 1, (0, 2, 0, 3): 1, (0, 3, 0, 4): 1,
           (1, 0, 1, 1): 1, (1, 1, 1, 2): 1, (1, 2, 1, 3): 1, (1, 3, 1, 4): 1,
           (2, 0, 2, 1): 1, (2, 1, 2, 2): 1, (2, 2, 2, 3): 1, (2, 3, 2, 4): 1,
           (0, 0, 1, 0): 1, (1, 0, 2, 0): 1,
           (0, 1, 1, 1): 1, (1, 1, 2, 1): 1,
           (0, 2, 1, 2): 1, (1, 2, 2, 2): 1,
           (0, 3, 1, 3): 1, (1, 3, 2, 3): 1,
           (0, 4, 1, 4): 1, (1, 4, 2, 4): 1,
           (0, 0): 2, (0, 1): 2, (0, 2): 2, (0, 3): 2, (0, 4): 2,
           (1, 0): 3, (1, 1): 3, (1, 2): 3, (1, 3): 3, (1, 4): 3,
           (2, 0): 2, (2, 1): 2, (2, 2): 2, (2, 3): 2, (2, 4): 2}
    J = {'fun': fun, 'fac': fac, 'N': N, 'Nx': Nx, 'Ny': Ny}

    logging.basicConfig(level='INFO')

    ins = otn2d.otn2d(mode='RMF', Nx=Nx, Ny=Ny, J=J, beta=4)

    #  rotates graph
    if rot > 0:
        ins.rotate_graph(rot=rot)

    # if using excitations_encoding = 2 or 3
    # adds small noise to remove accidental degeneracies
    if excitations_encoding > 1:
        ins.add_noise(amplitude=1e-7)

    # applies preconditioning using balancing heuristics
    if precondition:
        ins.precondition(mode='balancing')

    # search low enegy spectrum (return lowest energy. full data stored in ins)
    Eng = ins.search_low_energy_spectrum(excitations_encoding=excitations_encoding,
                                         M=M, relative_P_cutoff=relative_P_cutoff, Dmax=D, max_dEng=dE, lim_hd=hd)
    # decodes it into states
    Eng = ins.decode_low_energy_states(max_dEng=dE, max_states=max_states)
    return ins


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=int, default=0,
                        help="Rotate graph by 90 deg r times. Default is 0. Used to try to search/contract from different sides.")
    parser.add_argument("-b", type=float, default=3,
                        help="Inverse temperature. Default is set at 3.")
    parser.add_argument("-D", type=int, default=32,
                        help="Maximal bond dimension of boundary MPS used to contract PEPS.")
    parser.add_argument("-M", type=int, default=2**7,
                        help="Maximal number of partial states kept during branch and bound search.")
    parser.add_argument("-P", type=float, default=1e-12,
                        help="Cuttof on the range of relative probabilities kept during branch and bound search.")
    parser.add_argument("-dE", type=float, default=3.1,
                        help="Limit on excitation energy.")
    parser.add_argument("-hd", type=int, default=0,
                        help="Lower limit of Hamming distance between states (while merging). Outputs less states.")
    parser.add_argument("-max_st", type=int, default=2**20,
                        help="Limit total number of low energy states which is being reconstructed.")
    parser.add_argument("-ee", type=int, default=1, choices=[1, 2, 3],
                        help="Strategy used to compress droplets. For excitations_encoding = 2 or 3 small noise is added to the couplings slighly modyfings energies.")
    parser.add_argument('-pre', dest='pre', action='store_true', help="Do not use preconditioning.")
    parser.set_defaults(pre=False)
    args = parser.parse_args()

    ins = main(rot=args.r,
               beta=args.b,
               D=args.D,
               M=args.M, relative_P_cutoff=args.P,
               excitations_encoding=args.ee,
               dE=args.dE, hd=args.hd,
               max_states=args.max_st,
               precondition=args.pre)

    ins.show_solution(state=False)
    print('Energies of the found low-energy states:')
    print('(for ee = 2 or 3 small noise of O(1e-7) was added to couplings, to distinguish degenerated solutions):')
    print(ins.energy)

    # to display excitation tree, uncomment the lines below
    print()
    print('Tree of droplets (intendation shows hierarchy):')
    print('dEng : variable | change (xor) of viariable')
    ins.exc_print()
