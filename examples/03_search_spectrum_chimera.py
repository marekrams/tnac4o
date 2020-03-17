import logging
import argparse
import time
from otn2d import otn2d


def search_spectrum_droplet(L, ins, rot, beta, M, relative_P_cutoff, ver, dE, hd, max_states):
    ''' Runs a script searching for ground state of droplet instances.
        Instances are located in the folder "./../instances/"
    '''
    if args.L == 128:
        Nx, Ny, Nc = 4, 4, 8
        filename_in = ('./../instances/Chimera_droplet_instances/chimera128_spinglass_power/%03d.txt' % args.ins)
    elif args.L == 512:
        Nx, Ny, Nc = 8, 8, 8
        filename_in = ('./../instances/Chimera_droplet_instances/chimera512_spinglass_power/%03d.txt' % args.ins)
    elif args.L == 1152:
        Nx, Ny, Nc = 12, 12, 8
        filename_in = ('./../instances/Chimera_droplet_instances/chimera1152_spinglass_power/%03d.txt' % args.ins)
    elif args.L == 2048:
        Nx, Ny, Nc = 16, 16, 8
        filename_in = ('./../instances/Chimera_droplet_instances/chimera2048_spinglass_power/%03d.txt' % args.ins)

    # Initialize global logging level to INFO.
    logging.basicConfig(level='INFO')

    J = otn2d.load_Jij(filename_in)
    # those instances are defined with spin numering starting with 1
    J = otn2d.Jij_f2p(J)
    # round J to 1/75 for those instances
    J = [[x[0], x[1], round(75.*x[2])/75.] for x in J]

    #  initialize solver
    ins = otn2d.otn2d(mode='Ising', Nx=Nx, Ny=Ny, Nc=Nc, J=J, beta=args.b)

    #  rotates graph
    if args.r > 0:
        ins.rotate_graph(rot=args.r)

    if args.ver > 1:
        ins.add_noise(amplitude=1e-7)
        
    #  applies preconditioning using balancing heuristics
    ins.precondition(mode='balancing')

    # search ground state
    Eng = ins.search_low_energy_spectrum(ver=ver, M=M, relative_P_cutoff=relative_P_cutoff, max_dEng=dE, lim_hd=hd)

    # decode tree of droplets to generate all low-energy states
    Eng = ins.decode_low_energy_states(max_dEng=args.dE, max_states=max_states)

    # show energies
    return ins

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", type=int, choices=[128, 512, 1152, 2048], default=128,
                        help="Size of chimera graph. Default is 128 (C4).")
    parser.add_argument("-ins", type=int, choices=range(1, 101), metavar="[1-100]", default=1,
                        help="Instance number (1-100). Default is 1.")
    parser.add_argument("-r", type=int, default=0,
                        help="Rotate graph by 90 deg r times. Default is 0. Used to try to search/contract from different sides.")
    parser.add_argument("-b", type=float, default=4,
                        help="Inverse temperature.")
    parser.add_argument("-M", type=int, default=2**10,
                        help="Maximal number of partial states kept during branch and bound search.")
    parser.add_argument("-P", type=float, default=1e-6,
                        help="Cuttof on the range of relative probabilities kept during branch and bound search.")
    parser.add_argument("-dE", type=float, default=1.0,
                        help="Limit on excitation energy.")
    parser.add_argument("-hd", type=int, default=0,
                        help="Lower limit of Hamming distance between some states. Outputs less states.")
    parser.add_argument("-max_states", type=int, default=2**20,
                        help="Limit total number of low energy states which is being reconstructed.")
    parser.add_argument("-ver", type=int, default=1, choices=[1, 2, 3],
                        help="Strategy used to compress droplets. For ver=2 or 3 adds a very small noise to the couplings slighly modyfings energies.")

    args = parser.parse_args()
    
    keep_time = time.time()
    ins = search_spectrum_droplet(L=args.L, ins=args.ins, rot=args.r, beta=args.b, M=args.M, relative_P_cutoff=args.P, ver=args.ver, dE=args.dE, hd=args.hd,  max_states=args.max_states)
    ins.logger.info('Total time : %.2f seconds', time.time() - keep_time)

    # display solution on screen
    ins.show_solution(state=False)
 
    # translates low energy states to bit_strings
    bit_strings = ins.binary_states()
    print('Number of states:', len(bit_strings))

    # display excitation energies    
    print(ins.energy-ins.energy[0])

    # display excitation tree    
    ins.exc_print()



