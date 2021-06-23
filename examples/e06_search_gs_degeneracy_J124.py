# Copyright 2021 Marek M. Rams, Masoud Mohseni, Bartlomiej Gardas. All Rights Reserved.
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
import os
import numpy as np


def search_gs_J124(C=8, instance=1,
                    rot=0, beta=0.75,
                    D=48,
                    M=2**12, relative_P_cutoff=1e-8,
                    precondition=True):
    '''
    Runs a script searching for a ground state of a J124 instance defined on a chimera graph.
    The ground state degeneracy is counted at no extra cost.

    Instances are located in the folder ./../instances/.
    Reasonable (but not neccesarily optimal) values of parameters for those instances are set as default.
    Some can be changed using options in this script. See documentation for more information.
    '''

    # Initialize global logging level to INFO.
    logging.basicConfig(level='INFO')

    # filename of the instance of interest
    if C == 8:
        Nx, Ny, Nc = 8, 8, 8
        filename_in = os.path.join(os.path.dirname(__file__),
                      './../instances/Chimera_J124/C=8_J124/%03d.txt' % instance)
    elif C == 12:
        Nx, Ny, Nc = 12, 12, 8
        filename_in = os.path.join(os.path.dirname(__file__),
                      './../instances/Chimera_J124/C=12_J124/%03d.txt' % instance)
    elif C == 16:
        Nx, Ny, Nc = 16, 16, 8
        filename_in = os.path.join(os.path.dirname(__file__),
                      './../instances/Chimera_J124/C=16_J124/%03d.txt' % instance)

    # load Jij couplings
    J = tnac4o.load_Jij(filename_in)

    # those instances are defined with spin numering starting with 1
    # change to 0-base indexing
    J = tnac4o.Jij_f2p(J)

    # initializes solver
    ins = tnac4o.tnac4o(mode='Ising', Nx=Nx, Ny=Ny, Nc=Nc, J=J, beta=beta)
    ins.logger.info('Analysing J124 instance %1d on chimera graph of %1d sites.' % (instance, Nx * Ny * Nc))
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
    parser.add_argument("-C", type=int, choices=[8, 12, 16], default=8,
                        help="Size of the chimera graph. Default is C8 (N=512).")
    parser.add_argument("-ins", type=int, choices=range(1, 201), metavar="[1-200]", default=1,
                        help="Instance number (1-100). Default is 1.")
    parser.add_argument("-b", type=float, default=0.75,
                        help="Inverse temperature. Default is set at 3.")
    parser.add_argument("-D", type=int, default=48,
                        help="Maximal bond dimension of boundary MPS used to contract PEPS.")
    parser.add_argument("-M", type=int, default=2**12,
                        help="Maximal number of partial states kept during branch-and-bound search.")
    parser.add_argument("-P", type=float, default=1e-8,
                        help="Cuttof on the range of relative probabilities kept during branch-and-bound search.")
    parser.add_argument("-s", dest='s', action='store_true',
                        help="Save results to txt file in ./results/")
    parser.add_argument('-no-pre', dest='pre', action='store_false', help="Do not use preconditioning.")
    parser.set_defaults(pre=True)
    args = parser.parse_args()

    Engs, degs = [], []
    for rot in range(4):
        keep_time = time.time()
        ins = search_gs_J124(C=args.C, instance=args.ins, rot=rot, beta=args.b,
                            D=args.D, M=args.M, relative_P_cutoff=args.P, precondition=args.pre)
        ins.logger.info('Rotation %1d; Total time : %.2f seconds', rot, time.time() - keep_time)
        ins.show_solution(state=False)
        Engs.append(ins.energy)
        degs.append(ins.degeneracy)

    Eng = min(Engs)
    best = tuple(ii for ii, E in enumerate(Engs) if E == Eng)
    deg = max(degs[ii] for ii in best)

    print('Best found energy and its degeneracy for J124 instances on chimera graph C%1d, instance %1d' %(args.C, args.ins))
    print('Energy = %1d' % Eng)
    print('Degeneracy = %1d' % deg)

    if args.s:
        # save it to ./results/*.txt
        filename = os.path.join(os.path.dirname(__file__),
                   './results/J124_C=%1d_ins=%03d_beta=%0.2f_D=%1d_M=%1d_pre=%1d.txt'
                   % (args.C, args.ins, args.b, args.D, args.M, args.pre))
        np.savetxt(filename, np.array([Eng, deg], dtype=int), delimiter=' ', header='Energy and degeneracy', fmt='%1d')
