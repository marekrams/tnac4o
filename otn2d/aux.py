r"""
Auxliary functions to load and operate on couplings.
"""

import numpy as np
import scipy.sparse
# import h5py


def load_Jij(file_name):
    r"""
    Loads couplings of the Ising model/problem from a file.

    Args:
        file_name (str): a path to file with coupling written in the format where each line contains i, j, Jij.

    Returns:
        a list of couplings with elements in the format [i, j, Jij].
    """
    J = np.loadtxt(file_name)
    J = [[int(l[0]), int(l[1]), float(l[2])] for l in J]
    return J


def round_Jij(J, dJ):
    r"""
    Round couplings to multiplies of dJ.

    Args:
        J (list): a list of couplings with elements in the format [i, j, Jij].

    Returns:
        a list of [i, j, rJij] couplings, where rJij = round(Jij/dJ)*dJ
    """
    dJ = float(dJ)
    return [[x[0], x[1], round(x[2]/dJ)*dJ] for x in J]


def minus_Jij(J):
    r"""
    Change sign of all Jij couplings.

    Args:
        J (list): a list of couplings with elements in the format [i, j, Jij].

    Returns:
        a list of [i, j, -Jij].
    """
    return [[l[0], l[1], -l[2]] for l in J]


def Jij_f2p(J):
    r"""
    Change 1-base indexig to 0-base indexing in a list of Jij couplings.

    Class otn2d expects 0-based indexing.

    Args:
        J (list): a list of couplings with elements in the format [i, j, Jij].

    Returns:
        a list of [i-1, j-1, Jij].

    """
    return [[l[0]-1, l[1]-1, l[2]] for l in J]


def energy_Jij(J, states):
    r"""
    Calculates energies from bit_strings for Ising model.

    Args:
        J (list): a list of couplings with elements in the format [i, j, Jij]
        states (nparray): used encoding\: 1 (spin up :math:`s_i=+1`), 0 (spin down :math:`s_i=-1`)

    Returns:
        1d numpy array with energies of all states.
    """

    L = len(states[0])

    ii, jj, vv = zip(*J)
    JJ = scipy.sparse.coo_matrix((vv, (ii, jj)), shape=(L, L))
    # makes the coupling matrix upper triangular
    JJ = scipy.sparse.triu(JJ) + scipy.sparse.tril(JJ, -1).T

    st = 2*np.array(states)-1
    Ns, dNs = st.shape[0], 1024
    Eng = np.zeros(Ns, dtype=float)
    for nn in range(0, Ns, dNs):
        ind = np.arange(nn, min(nn+dNs, Ns))
        Eng[ind] = np.sum(np.dot(st[ind], scipy.sparse.triu(JJ, 1).toarray())*st[ind], 1) + np.dot(st[ind], JJ.diagonal())
    return Eng


def energy_RMF(J, states):
    r"""
    Calculates cost function for bit_string for RMF.

    Args:
        J (dict): dictionary encoding the cost function as factored graph on 2d rectangular lattice, see :meth:`otn2d.otn2d` for used conventions.
        states (nparray): configurations

    Returns:
        1d numpy array with energies for all states.
    """
    Engs = np.zeros(len(states))
    for key, val in J['fac'].items():
        if len(key) == 2:
            ny, nx = key
            n = ny*J['Nx']+nx
            Engs += J['fun'][val][states[:, n]]
        elif len(key) == 4:
            ny1, nx1, ny2, nx2 = key
            n1 = ny1*J['Nx']+nx1
            n2 = ny2*J['Nx']+nx2
            Engs += J['fun'][val][states[:, n1], states[:, n2]]
    return Engs
