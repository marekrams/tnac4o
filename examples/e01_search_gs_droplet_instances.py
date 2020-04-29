import logging
import argparse
import time
import tnac4o
import os


def search_gs_droplet(L=128, instance=1,
                      rot=0, beta=3,
                      D=48,
                      M=1024, relative_P_cutoff=1e-8,
                      precondition=True):
    '''
    Runs a script searching for a ground state of a droplet instance.

    Instances are located in the folder ./../instances/.
    Reasonable (but not neccesarily optimal) values of parameters for those instances are set as default.
    Some can be changed using options in this script. See documentation for more information.
    '''

    # Initialize global logging level to INFO.
    logging.basicConfig(level='INFO')

    # filename of the instance of interest
    if L == 128:
        Nx, Ny, Nc = 4, 4, 8
        filename_in = os.path.join(os.path.dirname(__file__),
                      './../instances/Chimera_droplet_instances/chimera128_spinglass_power/%03d.txt' % instance)
    elif L == 512:
        Nx, Ny, Nc = 8, 8, 8
        filename_in = os.path.join(os.path.dirname(__file__),
                      './../instances/Chimera_droplet_instances/chimera512_spinglass_power/%03d.txt' % instance)
    elif L == 1152:
        Nx, Ny, Nc = 12, 12, 8
        filename_in = os.path.join(os.path.dirname(__file__),
                      './../instances/Chimera_droplet_instances/chimera1152_spinglass_power/%03d.txt' % instance)
    elif L == 2048:
        Nx, Ny, Nc = 16, 16, 8
        filename_in = os.path.join(os.path.dirname(__file__),
                      './../instances/Chimera_droplet_instances/chimera2048_spinglass_power/%03d.txt' % instance)

    # load Jij couplings
    J = tnac4o.load_Jij(filename_in)

    # those instances are defined with spin numering starting with 1
    # change to 0-base indexing
    J = tnac4o.Jij_f2p(J)

    # round J to multiplies of 1/75 for those instances
    # as couplings were saved with 6 digit precision
    J = tnac4o.round_Jij(J, 1 / 75)

    # initializes solver
    ins = tnac4o.tnac4o(mode='Ising', Nx=Nx, Ny=Ny, Nc=Nc, J=J, beta=beta)
    ins.logger.info('Analysing droplet instance %1d on chimera graph of %1d sites.' % (instance, L))
    # rotates graph to contract from different side/edge
    if rot > 0:
        ins.rotate_graph(rot=rot)

    # applies preconditioning using balancing heuristics
    if precondition:
        ins.precondition(mode='balancing')

    # search ground state (return lowest energy, full data stored in ins)
    Eng = ins.search_ground_state(M=M, relative_P_cutoff=relative_P_cutoff, Dmax=D)
    return ins


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", type=int, choices=[128, 512, 1152, 2048], default=128,
                        help="Size of the chimera graph. Default is 128 (C4).")
    parser.add_argument("-ins", type=int, choices=range(1, 101), metavar="[1-100]", default=1,
                        help="Instance number (1-100). Default is 1.")
    parser.add_argument("-r", type=int, default=0,
                        help="Rotate graph by 90 deg r times. Default is 0. \
                              Used to try to search/contract from different sides.")
    parser.add_argument("-b", type=float, default=3,
                        help="Inverse temperature. Default is set at 3.")
    parser.add_argument("-D", type=int, default=48,
                        help="Maximal bond dimension of boundary MPS used to contract PEPS.")
    parser.add_argument("-M", type=int, default=2**10,
                        help="Maximal number of partial states kept during branch-and-bound search.")
    parser.add_argument("-P", type=float, default=1e-8,
                        help="Cuttof on the range of relative probabilities kept during branch-and-bound search.")
    parser.add_argument('-no-pre', dest='pre', action='store_false', help="Do not use preconditioning.")
    parser.set_defaults(pre=True)
    args = parser.parse_args()

    keep_time = time.time()
    ins = search_gs_droplet(L=args.L, instance=args.ins, rot=args.r, beta=args.b,
                            D=args.D, M=args.M, relative_P_cutoff=args.P, precondition=args.pre)
    ins.logger.info('Total time : %.2f seconds', time.time() - keep_time)

    # display solution on screen
    ins.show_solution(state=False)
    print('Solution [1 -> spin up: si=+1; 0 -> spin down: si=-1]:')
    print(ins.binary_states())
