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

from e01_search_gs_droplet_instances import search_gs_droplet
from e02_sample_droplet_instances import gibbs_sampling
from e03_search_spectrum_droplet_instances import search_spectrum_droplet
from e05_minimal_RMF import minimal_RMF
from e06_search_gs_degeneracy_J124 import search_gs_J124
import tnac4o
import os


def test_e01():
    """ Minimal test of ground state search. """
    # expected energy for droplet instance 1 for L=128
    expected_energy = -210.93333333

    ins = search_gs_droplet(rot=0, D=8, precondition=True)
    assert(abs(expected_energy - ins.energy[0]) < 1e-5)

    ins = search_gs_droplet(rot=3, D=8, precondition=False)
    assert(abs(expected_energy - ins.energy[0]) < 1e-5)


def test_e02():
    """ Minimal test of Gibbs sampling. """
    M = 128
    ins0 = gibbs_sampling(M=M, precondition=False, rot=0)
    assert(M == len(ins0.states))
    ins1 = gibbs_sampling(M=M, precondition=False, rot=1)
    filename_in = os.path.join(os.path.dirname(__file__),
                               './../instances/Chimera_droplet_instances/chimera128_spinglass_power/001.txt')

    # test consistency of energies
    J = tnac4o.load_Jij(filename_in)
    J = tnac4o.Jij_f2p(J)
    J = tnac4o.round_Jij(J, 1 / 75)
    st0 = ins0.binary_states()
    st1 = ins1.binary_states()
    E0 = tnac4o.energy_Jij(J, st0)
    E1 = tnac4o.energy_Jij(J, st1)

    err0 = max(abs(ins0.energy - E0))
    err1 = max(abs(ins1.energy - E1))
    assert(max(err0, err1) < 1e-6)


def test_e03():
    """ Minimal test of low-energy spectrum search. """
    # expected number of states with dE<1 for droplet instance 1 for L=128
    expected_number_of_states = 31
    ins0 = search_spectrum_droplet(D=16, excitations_encoding=1, rot=0, precondition=False, dE=1.)
    ins1 = search_spectrum_droplet(D=16, excitations_encoding=1, rot=1, precondition=False, dE=1.)
    ins2 = search_spectrum_droplet(D=16, excitations_encoding=2, rot=2, precondition=False, dE=1.)
    ins3 = search_spectrum_droplet(D=16, excitations_encoding=3, rot=3, precondition=False, dE=1.)

    ins0.decode_low_energy_states(max_dEng=1.)
    ins1.decode_low_energy_states(max_dEng=1.)
    ins2.decode_low_energy_states(max_dEng=1.)
    ins3.decode_low_energy_states(max_dEng=1.)

    assert(len(ins0.energy) == expected_number_of_states)
    assert(len(ins1.energy) == expected_number_of_states)
    assert(len(ins2.energy) == expected_number_of_states)
    assert(len(ins3.energy) == expected_number_of_states)

    err0 = max(abs(ins0.energy - ins1.energy))
    err1 = max(abs(ins1.energy - ins2.energy))
    err2 = max(abs(ins1.energy - ins3.energy))

    assert(max(err0, err1, err2) < 1e-4)

    st0 = ins0.binary_states()
    st1 = ins1.binary_states()
    st2 = ins2.binary_states()
    st3 = ins3.binary_states()

    filename_in = os.path.join(os.path.dirname(__file__),
                               './../instances/Chimera_droplet_instances/chimera128_spinglass_power/001.txt')
    J = tnac4o.load_Jij(filename_in)
    J = tnac4o.Jij_f2p(J)
    J = tnac4o.round_Jij(J, 1 / 75)

    E0 = tnac4o.energy_Jij(J, st0)
    E1 = tnac4o.energy_Jij(J, st1)
    E2 = tnac4o.energy_Jij(J, st2)
    E3 = tnac4o.energy_Jij(J, st3)

    err0 = max(abs(E1 - E0))
    err1 = max(abs(E2 - E0))
    err2 = max(abs(E3 - E0))

    assert(max(err0, err1, err2) < 1e-4)


def test_e05():
    """ Minimal test of RMF example. """
    # expected number of states with dE<3.1 for this problem
    expected_number_of_states = 26
    ins0 = minimal_RMF(excitations_encoding=1, rot=0)
    ins1 = minimal_RMF(excitations_encoding=1, rot=1)
    ins2 = minimal_RMF(excitations_encoding=2, rot=2)
    ins3 = minimal_RMF(excitations_encoding=3, rot=3)

    assert(len(ins0.energy) == expected_number_of_states)
    assert(len(ins1.energy) == expected_number_of_states)
    assert(len(ins2.energy) == expected_number_of_states)
    assert(len(ins3.energy) == expected_number_of_states)

    err0 = max(abs(ins0.energy - ins1.energy))
    err1 = max(abs(ins0.energy - ins2.energy))
    err2 = max(abs(ins0.energy - ins3.energy))

    assert(max(err0, err1, err2) < 1e-4)

    E0 = tnac4o.energy_RMF(ins0.J, ins0.states)
    E1 = tnac4o.energy_RMF(ins0.J, ins1.states)
    E2 = tnac4o.energy_RMF(ins0.J, ins2.states)
    E3 = tnac4o.energy_RMF(ins0.J, ins3.states)

    err0 = max(abs(E1 - E0))
    err1 = max(abs(E2 - E0))
    err2 = max(abs(E3 - E0))

    assert(max(err0, err1, err2) < 1e-4)


def test_e06():
    """ Minimal test of ground state search. """
    # expected energy for droplet instance 1 for L=128
    expected_energy = -2309
    expected_degeneracy = 1152

    ins = search_gs_J124(rot=0, C=8, instance=1, D=8, precondition=True)
    assert(abs(expected_energy - ins.energy[0]) < 1e-12)
    assert expected_degeneracy == ins.degeneracy


if __name__ == "__main__":
    test_e01()
    test_e02()
    test_e03()
    test_e05()
    test_e06()
