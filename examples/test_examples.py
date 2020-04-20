from e01_search_gs_droplet_instances import search_gs_droplet
from e02_sample_droplet_instances import gibbs_sampling
from e03_search_spectrum_droplet_instances import search_spectrum_droplet
from e05_minimal_RMF import minimal_RMF
import otn2d
import os


def test_e01():
    """ Minimal test of ground state search. """
    # expected energy for droplet instance 1 for L=128
    expected_energy = -210.933333

    ins = search_gs_droplet(rot=0, D=8, precondition=True)
    assert(abs(expected_energy - ins.energy[0]) < 1e-6)

    ins = search_gs_droplet(rot=3, D=8, precondition=False)
    assert(abs(expected_energy - ins.energy[0]) < 1e-6)
    
    assert(False)


def test_e02():
    """ Minimal test of Gibbs sampling. """
    M = 128
    ins0 = gibbs_sampling(M=M, precondition=False, rot=0)
    assert(M == len(ins0.states))
    ins1 = gibbs_sampling(M=M, precondition=False, rot=1)
    filename_in = os.path.join(os.path.dirname(__file__),
                './../instances/Chimera_droplet_instances/chimera128_spinglass_power/001.txt')

    # test consistency of energies
    J = otn2d.load_Jij(filename_in)
    J = otn2d.Jij_f2p(J)
    J = otn2d.round_Jij(J, 1 / 75)
    st0 = ins0.binary_states()
    st1 = ins1.binary_states()
    E0 = otn2d.energy_Jij(J, st0)
    E1 = otn2d.energy_Jij(J, st1)

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
    J = otn2d.load_Jij(filename_in)
    J = otn2d.Jij_f2p(J)
    J = otn2d.round_Jij(J, 1 / 75)

    E0 = otn2d.energy_Jij(J, st0)
    E1 = otn2d.energy_Jij(J, st1)
    E2 = otn2d.energy_Jij(J, st2)
    E3 = otn2d.energy_Jij(J, st3)

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

    E0 = otn2d.energy_RMF(ins0.J, ins0.states)
    E1 = otn2d.energy_RMF(ins0.J, ins1.states)
    E2 = otn2d.energy_RMF(ins0.J, ins2.states)
    E3 = otn2d.energy_RMF(ins0.J, ins3.states)

    err0 = max(abs(E1 - E0))
    err1 = max(abs(E2 - E0))
    err2 = max(abs(E3 - E0))

    assert(max(err0, err1, err2) < 1e-4)


if __name__ == "__main__":
    test_e01()
    test_e02()
    test_e03()
    test_e05()
