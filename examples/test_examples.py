from e01_search_gs_droplet_instances import search_gs_droplet
from e02_sample_droplet_instances import gibbs_sampling
from e03_search_spectrum_droplet_instances import search_spectrum_droplet
from e05_minimal_RMF import minimal_RMF


def test_e01():
    # expected energy for droplet instance 1 for L=128
    expected_energy = -210.933333

    ins = search_gs_droplet(rot=0, D=8, precondition=True)
    assert(abs(expected_energy-ins.energy[0]) < 1e-6)

    ins = search_gs_droplet(rot=3, D=8, precondition=False)
    assert(abs(expected_energy-ins.energy[0]) < 1e-6)


def test_e02():
    M = 128
    ins = gibbs_sampling(M=M, precondition=False)
    assert(M == len(ins.states))


def test_e03():
    # expected number of states with dE<1 for droplet instance 1 for L=128
    expected_number_of_states = 31
    ins1 = search_spectrum_droplet(D=16, excitations_encoding=1, rot=1, precondition=False, dE=1.)
    ins2 = search_spectrum_droplet(D=16, excitations_encoding=2, rot=2, precondition=False, dE=1.)
    ins3 = search_spectrum_droplet(D=16, excitations_encoding=3, rot=3, precondition=False, dE=1.)

    ins1.decode_low_energy_states(max_dEng=1.)
    ins2.decode_low_energy_states(max_dEng=1.)
    ins3.decode_low_energy_states(max_dEng=1.)

    assert(len(ins1.energy) == expected_number_of_states)
    assert(len(ins2.energy) == expected_number_of_states)
    assert(len(ins3.energy) == expected_number_of_states)

    err1 = max(abs(ins1.energy - ins2.energy))
    err2 = max(abs(ins1.energy - ins3.energy))
    assert(max(err1, err2) < 1e-4)


def test_e05():
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


if __name__ == "__main__":
    test_e01()
    test_e02()
    test_e03()
    test_e05()
