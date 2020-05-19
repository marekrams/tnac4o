# Copyright 2020 Marek M. Rams, Masoud Mohseni, Bartlomiej Gardas. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import argparse
import time
import tnac4o
import numpy as np
import os


def gibbs_sampling(L=128, instance=1,
                   rot=0, beta=3,
                   D=48,
                   M=128,
                   precondition=True):
    '''
    Runs a script sampling from Gibbs distribution for a droplet instance.

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

    # initialize solver
    ins = tnac4o.tnac4o(mode='Ising', Nx=Nx, Ny=Ny, Nc=Nc, J=J, beta=beta)
    ins.logger.info('Analysing droplet instance %1d on chimera graph of %1d sites.' % (instance, L))

    # rotates graph
    if rot > 0:
        ins.rotate_graph(rot=rot)

    # applies preconditioning using balancing heuristics
    if precondition:
        ins.precondition(mode='balancing')

    # perform sampling (return energies, full data stored in ins)
    Eng = ins.gibbs_sampling(M=M, Dmax=D)
    return ins


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", type=int, choices=[128, 512, 1152, 2048], default=128,
                        help="Size of chimera graph. Default is 128 (C4).")
    parser.add_argument("-ins", type=int, choices=range(1, 101), metavar="[1-100]", default=1,
                        help="Instance number (1-100). Default is 1.")
    parser.add_argument("-r", type=int, default=0,
                        help="Rotate graph by 90 deg r times. Default is 0. \
                              Used to try to search/contract from different sides.")
    parser.add_argument("-b", type=float, default=3,
                        help="Inverse temperature. Default is set at 3.")
    parser.add_argument("-D", type=int, default=48,
                        help="Maximal bond dimension of boundary MPS used to contract PEPS.")
    parser.add_argument("-M", type=int, default=2**7,
                        help="Number of sampled states. Default is 128.")
    parser.add_argument('-no-pre', dest='pre', action='store_false',
                        help="Do not use preconditioning.")
    parser.set_defaults(pre=True)
    parser.add_argument("-s", dest='s', action='store_true',
                        help="Save sampled states to txt file in ./results/")
    parser.set_defaults(s=False)
    args = parser.parse_args()

    keep_time = time.time()
    ins = gibbs_sampling(L=args.L, instance=args.ins, rot=args.r, beta=args.b, D=args.D, M=args.M, precondition=args.pre)
    ins.logger.info('Total time : %.2f seconds', time.time() - keep_time)

    # display solution on screen
    ins.show_solution(state=False)

    # output samples as bit-strings
    bit_strings = ins.binary_states()

    if args.s:
        # save it to ./results/*.txt
        filename = os.path.join(os.path.dirname(__file__),
                   './results/gibbs_L=%1d_ins=%03d_r=%1d_beta=%0.2f_D=%1d_M=%1d_pre=%1d.txt'
                   % (args.L, args.ins, args.r, args.b, args.D, args.M, args.pre))
        f = open(filename, 'w')
        print("# One line per state; First column is the energy, the rest is a state; \
                1 = spin up = si=+1; 0 = spin down = si=-1", file=f)
        for ii in range(len(ins.energy)):
            st = np.zeros(ins.L + 1)
            st[1:] = bit_strings[ii]
            st[0] = ins.energy[ii]
            np.savetxt(f, np.c_[np.reshape(st, (1, ins.L + 1))], fmt=' '.join(['%4.6f'] + ['%i'] * ins.L), delimiter=' ')
        f.close()
