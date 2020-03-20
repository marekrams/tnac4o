import logging
import argparse
import time
from otn2d import otn2d


def search_spectrum_droplet(L=128, ins=1,
                            rot=0, beta=3,
                            M=1024, relative_P_cutoff=1e-6,
                            excitations_encoding=1,
                            dE=1., hd=0,
                            precondition=True):
    """
    Runs a script searching for ground state of `droplet instances`.

    Instances are located in the folder ./../instances/
    Reasonable (but not neccesarily optimal) values of parameters are used by default.
    Some can be chaged using options in this script.
    """

    # Initialize global logging level to INFO.
    logging.basicConfig(level='INFO')

    # filename of the instance of interest
    if L == 128:
        Nx, Ny, Nc = 4, 4, 8
        filename_in = ('./../instances/Chimera_droplet_instances/chimera128_spinglass_power/%03d.txt' % ins)
    elif L == 512:
        Nx, Ny, Nc = 8, 8, 8
        filename_in = ('./../instances/Chimera_droplet_instances/chimera512_spinglass_power/%03d.txt' % ins)
    elif L == 1152:
        Nx, Ny, Nc = 12, 12, 8
        filename_in = ('./../instances/Chimera_droplet_instances/chimera1152_spinglass_power/%03d.txt' % ins)
    elif L == 2048:
        Nx, Ny, Nc = 16, 16, 8
        filename_in = ('./../instances/Chimera_droplet_instances/chimera2048_spinglass_power/%03d.txt' % ins)

    # load Jij couplings
    J = otn2d.load_Jij(filename_in)

    # those instances are defined with spin numering starting with 1
    # change to 0-base indexing
    J = otn2d.Jij_f2p(J)

    # round J to multiplies of 1/75 for those instances
    J = [[x[0], x[1], round(75.*x[2])/75.] for x in J]

    #  initialize solver
    ins = otn2d.otn2d(mode='Ising', Nx=Nx, Ny=Ny, Nc=Nc, J=J, beta=beta)

    #  rotates graph
    if args.r > 0:
        ins.rotate_graph(rot=args.r)

    # if using excitations_encoding = 2 or 3
    # adds small noise to remove accidental degeneracies
    if excitations_encoding > 1:
        ins.add_noise(amplitude=1e-7)

    # applies preconditioning using balancing heuristics
    if precondition:
        ins.precondition(mode='balancing')

    # search for low energy spectrum
    Eng = ins.search_low_energy_spectrum(excitations_encoding=excitations_encoding, M=M, relative_P_cutoff=relative_P_cutoff, max_dEng=dE, lim_hd=hd)

    return ins


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", type=int, choices=[128, 512, 1152, 2048], default=128,
                        help="Size of chimera graph. Default is 128 (C4).")
    parser.add_argument("-ins", type=int, choices=range(1, 101), metavar="[1-100]", default=1,
                        help="Instance number (1-100). Default is 1.")
    parser.add_argument("-r", type=int, default=0,
                        help="Rotate graph by 90 deg r times. Default is 0. Used to try to search/contract from different sides.")
    parser.add_argument("-b", type=float, default=3,
                        help="Inverse temperature. Default is set at 3.")
    parser.add_argument("-M", type=int, default=2**10,
                        help="Maximal number of partial states kept during branch and bound search.")
    parser.add_argument("-P", type=float, default=1e-6,
                        help="Cuttof on the range of relative probabilities kept during branch and bound search.")
    parser.add_argument("-dE", type=float, default=1.0,
                        help="Limit on excitation energy.")
    parser.add_argument("-hd", type=int, default=0,
                        help="Lower limit of Hamming distance between states (while merging). Outputs less states.")
    parser.add_argument("-max_st", type=int, default=2**20,
                        help="Limit total number of low energy states which is being reconstructed.")
    parser.add_argument("-ee", type=int, default=1, choices=[1, 2, 3],
                        help="Strategy used to compress droplets. For excitations_encoding = 2 or 3 small noise is added to the couplings slighly modyfings energies.")
    parser.add_argument('-no-pre', dest='pre', action='store_false', help="Do not use preconditioning.")
    parser.set_defaults(pre=True)
    parser.add_argument("-s", dest='s', action='store_true', help="Saves results to file in ./temp/")
    parser.set_defaults(s=False)
    args = parser.parse_args()

    keep_time = time.time()
    ins = search_spectrum_droplet(L=args.L, ins=args.ins,
                                  rot=args.r,
                                  beta=args.b,
                                  M=args.M, relative_P_cutoff=args.P,
                                  excitations_encoding=args.ee,
                                  dE=args.dE, hd=args.hd,
                                  precondition=args.pre)
    ins.logger.info('Total time : %.2f seconds', time.time() - keep_time)

    # saves solution to file
    # saves before decoding excitations
    if args.s:
        file_name = './temp/L=%1d_ins=%03d_r=%1d_beta=%0.2f_M=%1d_P=%0.2e_ee=%1d_dE=%0.3f_hd=%1d_pre=%1d.npy' \
                    % (args.L, args.ins, args.r, args.b, args.M, args.P, args.ee, args.dE, args.hd, args.pre)
        ins.save(file_name)

    # display solution on screen
    ins.show_solution(state=False)

    # decode low energy spectrum
    keep_time_decode = time.time()
    Eng = ins.decode_low_energy_states(max_dEng=args.dE, max_states=args.max_st)
    ins.logger.info('Decoding spectrum elapse time : %.2f seconds', time.time() - keep_time_decode)

    # translates low energy states to bit_strings
    bit_strings = ins.binary_states()
    print('Number of states:', len(bit_strings))

    # display excitation energies
    print()
    print('Excitation energies:')
    print(ins.energy-ins.energy[0])

    # display excitation tree
    # print()
    # print('Tree of droplets (intendation shows hierarchy):')
    # print('dEng : clusters | change (xor) in cluster')
    # ins.exc_print()
