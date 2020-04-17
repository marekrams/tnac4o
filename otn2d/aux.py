r"""
Auxliary functions to load and operate on couplings.
"""

import numpy as np
import scipy.sparse
# import h5py


def load_Jij(file_name):
    r"""
    Loads couplings of the Ising model from a file.

    Args:
        file_name (str): a path to file with coupling written in the format, :math:`i~~j~~J_{ij}`.

    Returns:
        a list of Jij couplings.
    """
    J = np.loadtxt(file_name)
    J = [[int(l[0]), int(l[1]), float(l[2])] for l in J]
    return J


def round_Jij(J, dJ):
    r"""
    Round couplings to multiplies of dJ.
    """
    dJ = float(dJ)
    return [[x[0], x[1], round(x[2]/dJ)*dJ] for x in J]


def minus_Jij(J):
    r"""
    Change sign of all couplings :math:`J_{ij} \rightarrow -J_{ij}`.
    """
    return [[l[0], l[1], -l[2]] for l in J]


def Jij_f2p(J):
    r"""
    Change 1-base indexig to 0-base indexing in a list of :math:`J_{ij}`.
    """
    return [[l[0]-1, l[1]-1, l[2]] for l in J]


def Jij_p2f(J):
    r"""
    Change 0-base indexig to 1-base indexing in a list of :math:`J_{ij}`.
    """
    return [[l[0]+1, l[1]+1, l[2]] for l in J]


def energy_Jij(J, states):
    r"""
    Calculates energies from bit_strings for Ising model.

    Args:
        J (list): list of couplings
        states (nparray): 1 (spin up :math:`s_i=+1`), 0 (spin down :math:`s_i=-1`)

    Returns:
        Energies for all states.
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
        J (dict): dictionary encoding the cost function as factored graph on 2d rectangular lattice.
        states (nparray): configurations

    Returns:
        Energies for all states.
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


# def load_openGM(fname, Nx, Ny):
#     r"""
#     Loads some factored graphs written in openGM format. Assumes rectangular lattice.

#     Args:
#         file_name (str): a path to file with factor graph in openGM format.
#         ints Nx, Ny: it is assumed that graph if forming an :math:`N_x \times N_y` lattice with
#             nearest-neighbour interactions only.

#     Returns:
#        dictionary with factors and funcitons defining the energy functional.
#     """
#     with h5py.File(fname, 'r') as hf:
#         keys = list(hf.keys())
#         data = hf[keys[0]]
#         H = list(data['header'])
#         #_, _, L, n_factors, _, _, n_functions, _ = H
#         F = np.array(data['factors'], dtype=int)
#         J = np.array(data['function-id-16000/indices'], dtype=int)
#         V = np.array(data['function-id-16000/values'], dtype=float)
#         N = np.array(data['numbers-of-states'], dtype=int)

#     F = list(F[::-1])
#     factors = {}
#     while len(F) > 0:
#         f1 = F.pop()
#         z1 = F.pop()
#         nn = F.pop()
#         n = []
#         for _ in range(nn):
#             tt = F.pop()
#             ny, nx = tt // Nx, tt % Nx
#             n = n + [ny, nx]
#         if len(n) == 4:
#             if abs(n[0]-n[2])+abs(n[1]-n[3]) != 1:
#                 Exception('Not nearest neighbour')
#         if len(n) == 2:
#             if (n[0] >= Ny) or (n[1] >= Nx):
#                 Exception('Wrong size')
#         factors[tuple(n)] = f1
#         if z1 != 0:
#             Exception('Something wrong with the expected convention.')

#     J = list(J[::-1])
#     functions, ii, lower = {}, -1, 0
#     while len(J) > 0:
#         ii += 1
#         nn = J.pop()
#         n = []
#         for _ in range(nn):
#             n.append(J.pop())
#         upper = lower + np.prod(n, dtype=int)
#         functions[ii] = np.reshape(V[lower:upper], n[::-1]).T
#         lower = upper
#     J = {}
#     J['fun'] = functions
#     J['fac'] = factors
#     J['N'] = np.reshape(N, (Ny, Nx))  # size of local block
#     J['Nx'] = Nx
#     J['Ny'] = Ny
#     return J