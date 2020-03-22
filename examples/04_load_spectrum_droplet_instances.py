import logging
import argparse
import time
from otn2d import otn2d


def main():
    """
    Runs a script loading solution for low energy spectrum, and decoding it.
    """
    # Initialize global logging level to INFO.

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
    args = parser.parse_args()

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

    # load Jij couplings
    J = otn2d.load_Jij(filename_in)

    # those instances are defined with spin numering starting with 1
    # change to 0-base indexing
    J = otn2d.Jij_f2p(J)

    # round J to multiplies of 1/75 for those instances
    J = [[x[0], x[1], round(75.*x[2])/75.] for x in J]

    logging.basicConfig(level='INFO')

    # file to load
    file_name = './temp/L=%1d_ins=%03d_r=%1d_beta=%0.2f_M=%1d_P=%0.2e_ee=%1d_dE=%0.3f_hd=%1d_pre=%1d.npy' \
                % (args.L, args.ins, args.r, args.b, args.M, args.P, args.ee, args.dE, args.hd, args.pre)

    # load instance
    try:
        ins = otn2d.load(file_name)
    except FileNotFoundError:
        print('First run script 03_*.py with option `-s` and other parameters identical to generate a solution.')
        quit()

    print('Decomposing excitation structure into low energy states.')
    keep_time_decode = time.time()
    Eng = ins.decode_low_energy_states(max_dEng=args.dE, max_states=args.max_st)
    bit_strings = ins.binary_states()
    ins.logger.info('Decoding spectrum elapse time : %.2f seconds', time.time() - keep_time_decode)

    ins.show_solution()

    Eng = otn2d.energy_Jij(J, bit_strings)
    error = max(abs(ins.energy - Eng))

    print('Consistency of different ways to calculate energies.')
    print('For ee = 2 or 3 expected difference is ~1e-6 due to applied noise.')
    print('Difference = ', error)

    return error


if __name__ == "__main__":
    main()
