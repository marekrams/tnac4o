r"""
In this package, the PEPS tensor network is contracted using boundary matrix product state (MPS) approach.
This module implement basic operations on MPS, as neccesary to do the contraction.
"""

import numpy as np
import scipy.linalg as splinalg


def svd(T):
    r"""
    Wraper to SVD.

    Returns:
        U, S, V
    """
    try:
        U, S, V = splinalg.svd(T, full_matrices=False)
    except splinalg.LinAlgError:
        U, S, V = splinalg.svd(T, full_matrices=False, lapack_driver='gesvd')
    maxU, minU = U.max(0), U.min(0)
    maxV, minV = V.max(1), V.min(1)
    ind = (np.abs(minU) > maxU) & (np.abs(minV) > maxV)
    U[:, ind] *= -1
    V[ind] *= -1
    return U, S, V


def qr(T):
    r"""
    Wraper to QR.

    Returns:
        Q, R
    """
    Q, R = splinalg.qr(T, mode='economic')
    sR = np.sign(np.real(np.diag(R)))
    sR[sR == 0] = 1
    Q, R = Q * sR, sR.reshape([-1, 1]) * R
    # maxQ, minQ = Q.max(0), Q.min(0)
    # maxR, minR = R.max(1), R.min(1)
    # ind = (np.abs(minQ) > maxQ) & (np.abs(minR) > maxR)
    # Q[:, ind] *= -1
    # R[ind] *= -1
    return Q, R


def svd_S(T):
    r"""
    Wraper to singular values from SVD.

    Returns:
        S
    """
    try:
        S = splinalg.svd(T, full_matrices=False, compute_uv=False)
    except splinalg.LinAlgError:
        S = splinalg.svd(T, full_matrices=False, lapack_driver='gesvd', compute_uv=False)
    return S


def nfactor(T):
    r"""
    Calculate normalisation of array.

    Returns:
        Largest element of matrix floored to power of 2.
    """
    x = np.max(np.abs(T))
    n = np.abs(np.float64(x)).view(np.int64)
    return 2.**((n >> 52) - 1023)


def dot(phi, psi):
    r"""Calculate dot product of two MPS: <phi|psi>."""
    RL = np.ones((1, 1))
    for n in range(psi.L):
        RL = psi._mps_RL(RL, psi.A[n], phi.A[n])
    return RL.flat[0]


class MPS:
    r"""
    Matrix Product States (MPS)

    Args:
        d (int): dimension of each local tise.
        L (int): number of sites (length of MPS).
        Dmax (int): maximal MPS bond dimension set during initialisation.
        initial {'X', 'Z', 'randR', 'randC'}: initial state.
        canonise {'left', 'right', None}: cannonisation of the initial state.
    """

    def __init__(self, d=2, L=2, Dmax=2, initial='X', canonise='left'):
        r"""
        Initializes MPS.
        """
        self.L = L  # chain length
        if type(d) == int:
            d = [d]
        multid = (L + len(d) - 1) // len(d)
        d = d * multid
        self.d = d[:L]  # dim of each local site
        self.zero = np.finfo(float).eps  # numerical precision
        self.dtype = None  # float64 or complex128
        self.C = np.ones((1, 1))  # C martix
        self.pC = L  # position of C in the chain
        self.normC = 1.  # keep norm of C for numerical stability
        self.reset_mps(initial, Dmax, canonise)

    def show_properties(self):
        r"""
        Shows basic properties of MPS.
        """
        print("L:", self.L)
        print("d:", self.d)
        print("D:", self.D)
        print("dtype:", self.dtype)
        print("R[-1]:", self.R[-1])
        print("F[-1]:", self.F[-1])
        print("Cummulated norm C:", self.normC)

    def reset_mps(self, initial='randR', Dmax=2, canonise='left'):
        r"""
        Initializes MPS.
        """
        self.D = self._Dset(Dmax, self.d)
        self.A = []
        for n in range(self.L):
            self.A.append(self._init_A(self.D[n], self.d[n], self.D[n + 1], initial))
        self.reset_R()
        self.reset_F()
        self.reset_S()
        self.discarded = [0] * (self.L + 1)
        if canonise == 'left':
            self.canonise_left()
        elif canonise == 'right':
            self.canonise_right()
        self.normC = 1.
        if initial == 'randC':
            self.dtype = 'complex128'
        else:
            self.dtype = 'float64'

    def copy(self):
        r"""
        Makes a copy of MPS.
        """
        phi = MPS(d=self.d, L=self.L, Dmax=1, initial='X', canonise=None)
        for n in range(phi.L):
            phi.A[n] = self.A[n].copy()
        phi.C = self.C.copy()
        phi.pC = self.pC
        phi.normC = self.normC
        phi.D = self.D[:]
        phi.R = self.R[:]
        phi.F = self.F[:]
        phi.dtype = self.dtype
        return phi

    def compress_mps(self, Dmax=np.inf, tolS=None, tolV=None, max_sweeps=4, graduate_truncation=True, verbose=False):
        r"""
        Truncate MPS. Initialise with svd and then uses variational compression.

        Args:
            Dmax (int): bond dimension.
            tolS (float): truncate smaller singular values during svd.
            tolV (float): condition for overlap convergence during one sweep.
            max_sweeps (int): maximal number of sweeps of variational compression.
            graduate_truncation: performs 2 more rounds of SVD truncation decreesing bond dimension gradually,
                mixing it with variational sweep.
            verbose: to display convergence statistics.

        Returns:
            overlap between the state before compression and the compressed one.
        """
        self.canonise_right()
        phi = self.copy()
        self.discarded = [0] * (self.L + 1)
        if graduate_truncation:
            self.canonise_left(compress=True, Dmax=Dmax * 4, tol=tolS / 10)
            overlap = self.variational_compress(phi, tol=tolV, max_sweeps=1, verbose=verbose)
            self.canonise_right(compress=True, Dmax=Dmax * 2, tol=tolS / 2)
        self.canonise_left(compress=True, Dmax=Dmax, tol=tolS)
        overlap = self.variational_compress(phi, tol=tolV, max_sweeps=max_sweeps, verbose=verbose)
        return overlap

    def canonise_left(self, compress=False, Dmax=np.inf, tol=None):
        r"""
        Left canonise MPS.

        Args:
            compress (bool): to svd truncate during the process.
            Dmax (int): max bond dimension during truncation.
            tol (float): tolerance of schmidt values kept during truncation.
        """
        self.C = np.ones((1, 1))
        self.pC = 0
        for n in range(self.L):
            self.attach_CA()
            self.orth_left(n)
            if compress:
                self.truncateC(Dmax, tol)
        self.F[-1], self.R[-1] = None, None

    def canonise_right(self, compress=False, Dmax=np.inf, tol=None):
        r"""
        Right canonise MPS.

        Args:
            compress (bool): to svd truncate during the process.
            Dmax (int): max bond dimension during truncation.
            tol (float): tolerance of schmidt values kept during truncation.
        """
        self.C = np.ones((1, 1))
        self.pC = self.L
        for n in range(self.L - 1, -1, -1):
            self.attach_AC()
            self.orth_right(n)
            if compress:
                self.truncateC(Dmax, tol)
        self.F[-1], self.R[-1] = None, None

    def variational_compress(self, phi, tol=None, max_sweeps=1, verbose=False):
        r"""
        Compress MPS variationally.

        Args:
            phi (MPS): target state.
            tol (float): target tolerance on convergence of the overlap during one sweep.
            max_sweeps (int): maximal number of sweeps.
            verbose (bool): display statistics.

        Returns:
            overlap between the state before compression and the compressed one.
        """
        if tol is None:
            tol = self.zero
        overlap = self.setup_RL_mix(phi)
        sweeps, diff = 0, 1.
        while (diff > tol):
            if sweeps >= max_sweeps:
                if verbose:
                    print("Max number of sweeps (%i) has been reached." % (max_sweeps))
                return overlap
            else:
                for n in range(self.L - 1, 0, -1):
                    self.optimise_site(phi, n)
                    self.orth_right(n)
                    self.update_S()
                    self.update_RR_mix(phi, n)
                diff = 0.
                for n in range(self.L):
                    self.optimise_site(phi, n)
                    self.orth_left(n)
                    dS = self.update_S()
                    diff = np.maximum(diff, dS)
                    self.update_RL_mix(phi, n)
                diff_overlap = np.abs(overlap - self.R[-1])
                overlap = self.R[-1]
                sweeps += 1
                if verbose:
                    print("Sweep: %i Overlap: %.16f diff_overlap: %.4e diff_S: %.4e"
                          % (sweeps, overlap, diff_overlap, diff))
        return overlap

    def reset_R(self):
        r"""
        Reset R (no MPO) environment.
        """
        self.R = [np.ones((1, 1)) for _ in range(self.L + 2)]
        self.R[-1] = None

    def reset_F(self):
        r"""
        Reset F (with MPO) environment.
        """
        self.F = [np.ones((1, 1, 1)) for _ in range(self.L + 2)]
        self.F[-1] = None

    def reset_S(self):
        r"""
        Reset schmidt values S.
        """
        self.S = [self._one_S(self.D[n]) for n in range(self.L + 1)]

    def measure_O1(self, OO):
        r"""
        Calculate expectation value: <psi|O_n|psi> for all n.
        O_n is 1-site operator and can be site dependent.
        """
        if type(OO) == np.ndarray:
            OO = [OO]
        multiO = (self.L + len(OO) - 1) // len(OO)
        OO, expO = OO * multiO, []
        normR = self.setup_RR()
        for n in range(self.L):
            expO.append(self.expectation_1site(OO[n], n) / normR)
            self.update_RL(n)
        return expO

    def measure_O2(self, OO):
        r"""
        Calculate expectation value: <psi|O_n|psi> for all n.
        O_n is 2-site operator and can be site dependent.
        """
        if type(OO) == np.ndarray:
            OO = [OO]
        multiO = (self.L + len(OO) - 1) // len(OO)
        OO, expO = OO * multiO, []
        normR = self.setup_RR()
        for n in range(self.L - 1):
            expO.append(self.expectation_2site(OO[n], n) / normR)
            self.update_RL(n)
        return expO

    def measure_correlations(self, OO):
        r"""
        Calculate expectation value: <psi|O_n O_m|psi> for all 2-point correlators.
        O_n is 1-site operator and can be site dependent.
        """
        if type(OO) == np.ndarray:
            OO = [OO]
        multiO = (self.L + len(OO) - 1) // len(OO)
        OO = OO * multiO
        normR = self.setup_RR()
        expOO = np.zeros((self.L, self.L))
        RLO = []
        for n in range(self.L):
            expOO[n, n] = self.expectation_1site(OO[n], n) / normR
            for m in range(n):
                expOO[m, n] = self._mps_expectation_O(RLO[m], self.R[n + 1], self.A[n], self.A[n], OO[n]) / normR
                expOO[n, m] = expOO[m, n]
                RLO[m] = self._mps_RL(RLO[m], self.A[n], self.A[n])
            RLO.append(self._mps_RLO(self.R[n], self.A[n], self.A[n], OO[n]))
            self.update_RL(n)
        return expOO

    def apply_mpo(self, M, Hconj=False):
        r"""
        Apply MPO to MPS: psi =H psi.  If Hconj: psi = H^dag psi.
        """
        for n in range(self.L):
            if M.support[n]:
                self.A[n], self.D[n], self.d[n], self.D[n + 1] = self._mps_HA(self.A[n], M.W[n], Hconj)

    def apply_diagonalO(self, diagO, n):
        r"""
        Apply diagonal operator to MPS on site n.
        """
        for ii in range(len(diagO)):
            self.A[n][:, ii, :] *= diagO[ii]

    def attach_AC(self):
        r"""
        A[n]*C[n+1] -> A[n].
        """
        n = self.pC - 1
        self.A[n] = self._mps_AC(self.A[n], self.C)

    def attach_CA(self):
        r"""
        C[n]*A[n] -> A[n].
        """
        n = self.pC
        self.A[n] = self._mps_CA(self.C, self.A[n])

    def update_RR(self, n):
        r"""
        Include site n to its right environment R.
        """
        newR = self._mps_RR(self.R[n + 1], self.A[n], self.A[n])
        if n == 0:
            self.R[self.L + 1] = newR.flat[0]
        else:
            self.R[n] = newR

    def setup_RR(self):
        r"""
        Prepare right environment. Returns <psi|psi>.
        """
        for n in range(self.L - 1, -1, -1):
            self.update_RR(n)
        return self.R[-1]

    def update_RL(self, n):
        r"""
        Include site n to its left environment.
        """
        newR = self._mps_RL(self.R[n], self.A[n], self.A[n])
        if n == self.L - 1:
            self.R[self.L + 1] = newR.flat[0]
        else:
            self.R[n + 1] = newR

    def setup_RL(self):
        r"""
        Prepare left environment.  Returns <psi|psi>.
        """
        for n in range(self.L):
            self.update_RL(n)
        return self.R[-1]

    def update_RR_mix(self, phi, n):
        r"""
        Include site n to its right mixed environment R.
        """
        newR = self._mps_RR(self.R[n + 1], phi.A[n], self.A[n])
        if n == 0:
            self.R[self.L + 1] = newR.flat[0]
        else:
            self.R[n] = newR

    def setup_RR_mix(self, phi):
        r"""
        Prepare right environment. Returns <psi|phi>.
        """
        for n in range(self.L - 1, -1, -1):
            self.update_RR_mix(phi, n)
        return self.R[-1]

    def update_RL_mix(self, phi, n):
        r"""
        Include site n to its left mixed environment R.
        """
        newR = self._mps_RL(self.R[n], phi.A[n], self.A[n])
        if n == self.L - 1:
            self.R[self.L + 1] = newR.flat[0]
        else:
            self.R[n + 1] = newR

    def setup_RL_mix(self, phi):
        r"""
        Prepare left environment.  Returns <psi|phi>.
        """
        for n in range(self.L):
            self.update_RL_mix(phi, n)
        return self.R[-1]

    def bond_env_mix(self, phi, n):
        r"""
        Env of then n-th bond in <psi|phi>.
        """
        return self._mps_bond_env(self.R[n], phi.A[n], self.A[n], self.R[n + 1])

    def update_FR(self, M, n):
        r"""
        Update FL environment.
        """
        newF = self._mps_FR(self.F[n + 1], M.W[n], self.A[n], self.A[n])
        if n == 0:
            self.F[self.L + 1] = newF.flat[0]
        else:
            self.F[n] = newF

    def setup_FR(self, M):
        r"""
        Prepare FR environment. Returns <psi|H|psi>.
        """
        for n in range(self.L - 1, -1, -1):
            self.update_FR(M, n)
        return self.F[-1]

    def update_FL(self, M, n):
        r"""
        update FL environment.
        """
        newF = self._mps_FL(self.F[n], M.W[n], self.A[n], self.A[n])
        if n == self.L - 1:
            self.F[self.L + 1] = newF.flat[0]
        else:
            self.F[n + 1] = newF

    def setup_FL(self, M):
        r"""
        Prepare FL environment. Returns <psi|H|psi>.
        """
        for n in range(self.L):
            self.update_FL(M, n)
        return self.F[-1]

    def update_FR_mix(self, M, phi, n):
        r"""
        Update mixed FR environment.
        """
        newF = self._mps_FR(self.F[n + 1], M.W[n], phi.A[n], self.A[n])
        if n == 0:
            self.F[self.L + 1] = newF.flat[0]
        else:
            self.F[n] = newF

    def setup_FR_mix(self, M, phi):
        r"""
        Prepare FR environment. Returns <psi|H|phi>.
        """
        for n in range(self.L - 1, -1, -1):
            self.update_FR_mix(M, phi, n)
        return self.F[-1]

    def update_FL_mix(self, M, phi, n):
        r"""
        Update FL environment.
        """
        newF = self._mps_FL(self.F[n], M.W[n], phi.A[n], self.A[n])
        if n == self.L - 1:
            self.F[self.L + 1] = newF.flat[0]
        else:
            self.F[n + 1] = newF

    def setup_FL_mix(self, M, phi):
        r"""
        Prepare FL environment. Returns <psi|H|phi>.
        """
        for n in range(self.L):
            self.update_FL_mix(M, phi, n)
        return self.F[-1]

    def orth_left(self, n):
        r"""
        Left orthogonalization of n-th site; overwrites C.
        """
        self.A[n], self.C, nC, Dr = self._mps_decompose_AC(self.A[n])
        self.normC *= nC
        self.D[n + 1] = Dr
        self.pC = n + 1

    def orth_right(self, n):
        r"""
        Right orthogonalization of n-th site; overwrites C.
        """
        self.C, self.A[n], nC, Dl = self._mps_decompose_CA(self.A[n])
        self.normC *= nC
        self.D[n] = Dl
        self.pC = n

    def update_S(self):
        r"""
        Updates schmidt values; returns the differencs from the old ones.
        """
        S = svd_S(self.C)
        if self.S[self.pC].size != S.size:
            self.S[self.pC] = self._one_S(S.size)
        dS = (self.S[self.pC] - S)
        dS = np.sqrt(np.sum(dS**2))
        self.S[self.pC] = S
        return dS

    def truncateC(self, Dmax, tol=None):
        r"""
        Truncate C[n] with SVD.

        A[m] should be left canonical for m<=n and right canonical for m>n.

        Args:
            Dmax (int): max bond dimension during truncation.
            tol (float): tolerance of schmidt values kept during truncation.

        Returns:
            error = sqrt(sum discarded S^2).
        """
        if (self.pC > 0) and (self.pC < self.L):
            if tol is None:
                tol = self.zero
            projL, self.C, projR, newD, discarded = self._mps_truncateC(self.C, Dmax, tol)
            self.A[self.pC - 1] = self._mps_AC(self.A[self.pC - 1], projL)
            self.A[self.pC] = self._mps_CA(projR, self.A[self.pC])
            self.D[self.pC] = newD
            self.discarded[self.pC] = max(self.discarded[self.pC], discarded)
        else:
            discarded = 0.
        return discarded

    def expectation_mix(self, phi, n):
        r"""
        Calculate expectation value <psi|H|phi> given proper environments and MPO H[n].
        """
        return self._mps_expectation(self.R[n], self.R[n + 1], phi.A[n], self.A[n])  # * (self.normC * phi.normC)

    def expectation_1mpo_mix(self, W, phi, n):
        r"""
        Calculate expectation value <psi|H|phi> given proper environments and MPO H[n].
        """
        return self._mps_expectation_mpo(self.F[n], self.F[n + 1], W, phi.A[n], self.A[n]) * (self.normC * phi.normC)

    def expectation_list_1mpo_mix(self, W, phi, n):
        r"""
        Calculate expectation value <psi|H|phi> given proper environments and MPO H[n].
        """
        return self._mps_expectation_list_mpo(self.F[n], self.F[n + 1], W, phi.A[n], self.A[n]) * (self.normC * phi.normC)

    def expectation_1site(self, O, n):
        r"""
        Calculate expectation value of 1 site operator O.
        """
        return self._mps_expectation_O(self.R[n], self.R[n + 1], self.A[n], self.A[n], O)

    def expectation_2site(self, O, n):
        r"""
        Calculate expectation value of 2 site operator.
        """
        return self._mps_expectation_O2(self.R[n], self.R[n + 2], self.A[n], self.A[n], self.A[n + 1], self.A[n + 1], O)

    def optimise_site(self, phi, n):
        r"""
        Optimise site n for variational compression.
        """
        self.A[n] = self._mps_RAR(self.R[n], phi.A[n], self.R[n + 1])

    def _one_S(self, D):
        r"""Set S to have one non-zero value."""
        S = np.zeros(D)
        S[0] = 1.
        return S

    def _init_A(self, Dl, d, Dr, initial, state=0):
        r"""Initialise MPS tensor. X is maximally mixed state."""
        if initial == 'randR':
            return np.array(2 * np.random.rand(Dl, d, Dr) - 1, order='C')
        elif initial == 'randC':
            return np.array((2 * np.random.rand(Dl, d, Dr) - 1) + 1j * (2 * np.random.rand(Dl, d, Dr) - 1), order='C')
        elif initial == 'X':
            A = np.zeros((Dl, d, Dr))
            A[0, :, 0] = 1. / np.sqrt(d)
            return np.array(A, order='C')
        else:  # == 'Z'
            A = np.zeros((Dl, d, Dr))
            A[0, state, 0] = 1
            return np.array(A, order='C')

    def _Dset(self, Dmax, d):
        r"""Sets bond dimension to match maximal one and physical dimensions."""
        L = len(d)
        D = [1] * (L + 1)
        for n in range(L):
            D[n + 1] = min(D[n] * d[n], Dmax)
        D[-1] = 1
        for n in range(L - 1, -1, -1):
            D[n] = min(D[n + 1] * d[n], Dmax, D[n])
        return D

    def _mps_RL(self, RL, A, Ac):
        """Update left environment."""
        T = np.tensordot(RL, A, axes=(1, 0))
        return np.tensordot(Ac, T, axes=([0, 1], [0, 1]))

    def _mps_RR(self, RR, A, Ac):
        r"""Update right environment."""
        T = np.tensordot(A, RR, axes=(2, 0))
        return np.tensordot(T, Ac, axes=([1, 2], [1, 2]))

    def _mps_RLO(self, RL, A, Ac, O):
        r"""Update left environment with 1-site operator."""
        T1 = np.tensordot(RL, A, axes=(1, 0))
        T2 = np.tensordot(T1, O, axes=(1, 1))
        return np.tensordot(Ac, T2, axes=([0, 1], [0, 2]))

    def _mps_FL(self, FL, Hn, A, Ac):
        r"""Update left environment with MPO"""
        T1 = np.transpose(Ac, (2, 1, 0))
        T2 = np.tensordot(T1, FL, axes=(2, 0))
        T3 = np.tensordot(T2, Hn, axes=([1, 2], [1, 0]))
        return np.tensordot(T3, A, axes=([1, 3], [0, 1]))

    def _mps_FR(self, FR, Hn, A, Ac):
        r"""Update right environment with MPO."""
        T1 = np.transpose(A, (2, 1, 0))
        T2 = np.tensordot(FR, T1, axes=(2, 0))
        T3 = np.tensordot(Hn, T2, axes=([2, 3], [1, 2]))
        return np.tensordot(Ac, T3, axes=([1, 2], [1, 2]))

    def _mps_H1(self, FL, FR, Hn, A):
        r"""Effective action of MPO on 1 MPS martix."""
        sA = A.shape
        Dl, d, Dr = FL.shape[2], Hn.shape[3], FR.shape[2]
        T1 = np.transpose(np.reshape(A, [Dl, d, Dr]), (2, 1, 0))
        T2 = np.tensordot(FR, T1, axes=(2, 0))
        T3 = np.tensordot(Hn, T2, axes=([2, 3], [1, 2]))
        return np.reshape(np.tensordot(FL, T3, axes=([1, 2], [0, 3])), sA)

    def _mps_expectation(self, RL, RR, A, Ac):
        r"""Calculate expectation value of MPO."""
        T1 = np.tensordot(RL, A, axes=(1, 0))
        T2 = np.tensordot(T1, RR, axes=(2, 0))
        return np.tensordot(T2, Ac, axes=((0, 1, 2), (0, 1, 2)))

    def _mps_expectation_mpo(self, FL, FR, Hn, A, Ac):
        r"""Calculate expectation value of MPO."""
        T1 = np.transpose(A, (2, 1, 0))
        T2 = np.tensordot(FR, T1, axes=(2, 0))
        T3 = np.tensordot(Hn, T2, axes=([2, 3], [1, 2]))
        T4 = np.tensordot(Ac, T3, axes=([1, 2], [1, 2]))
        return np.tensordot(FL, T4, axes=([0, 1, 2], [0, 1, 2]))

    def _mps_expectation_list_mpo(self, FL, FR, Hn, A, Ac):
        r"""Calculate expectation value of MPO."""
        T1 = np.transpose(A, (2, 1, 0))
        T2 = np.tensordot(FR, T1, axes=(2, 0))
        T3 = np.tensordot(Hn, T2, axes=([3, 4], [1, 2]))
        T4 = np.tensordot(T3, Ac, axes=([2, 3], [1, 2]))
        return np.tensordot(T4, FL, axes=([1, 2, 3], [1, 2, 0]))

    def _mps_expectation_O(self, RL, RR, A, Ac, O):
        r"""Calculate expectation value of 1 site operator."""
        T1 = np.tensordot(RL, A, axes=(1, 0))
        T2 = np.tensordot(T1, RR, axes=(2, 0))
        T3 = np.tensordot(T2, O, axes=(1, 1))
        return np.tensordot(T3, Ac, axes=([0, 2, 1], [0, 1, 2]))

    def _mps_expectation_O2(self, RL, RR, A1, A1c, A2, A2c, OO):
        r"""Calculate expectation value of 1 site operator."""
        AA = self._mps_AA(A1, A2)
        AAc = self._mps_AA(A1c, A2c)
        d1, d2 = A1.shape[1], A2.shape[1]
        OO = np.reshape(OO, [d1 * d2, d1 * d2])
        T1 = np.tensordot(RL, AA, axes=(1, 0))
        T2 = np.tensordot(T1, RR, axes=(2, 0))
        T3 = np.tensordot(T2, OO, axes=(1, 1))
        return np.tensordot(T3, AAc, axes=([0, 2, 1], [0, 1, 2]))

    def _mps_AA(self, A1, A2):
        r"""Collects two MPS sites into one."""
        Dl, d1, _ = A1.shape
        _, d2, Dr = A2.shape
        return np.reshape(np.tensordot(A1, A2, axes=(2, 0)), [Dl, d1 * d2, Dr])

    def _mps_AC(self, A, C):
        r"""A, C -> AC."""
        return np.tensordot(A, C, axes=(2, 0))

    def _mps_CA(self, C, A):
        r"""C, A -> CA."""
        return np.tensordot(C, A, axes=(1, 0))

    def _mps_RAR(self, RL, A, RR):
        r"""RL, A, RR -> RL*A*RR."""
        T1 = np.tensordot(RL, A, axes=(1, 0))
        return np.tensordot(T1, RR, axes=(2, 0))

    def _mps_HA(self, A, Hn, Hconj):
        r"""Applies MPO to MPS at site n. If Hconj then apply Hn^dag."""
        if Hconj:
            T = np.tensordot(A, Hn, axes=([1, 1]))
            T1 = np.transpose(T, (0, 2, 4, 1, 3))
        else:
            T = np.tensordot(Hn, A, axes=([3, 1]))
            T1 = np.transpose(T, (0, 3, 1, 2, 4))
        a, a1, d, b, b1 = T1.shape
        Dl, Dr = a * a1, b * b1
        return np.reshape(T1, [Dl, d, Dr]), Dl, d, Dr

    def _mps_bond_env(self, RL, A, Ac, RR):
        r"""Update left environment."""
        T1 = np.tensordot(RL, A, axes=(1, 0))
        T2 = np.tensordot(T1, RR, axes=(2, 0))
        return np.tensordot(T2, Ac, axes=([0, 2], [0, 2]))

    # decomposition of objects
    def _mps_decompose_AC(self, A):
        r"""Splits QR A -> AC."""
        Dl, d, Dr = A.shape
        Q, C = qr(np.reshape(A, [Dl * d, Dr]))
        nC = nfactor(C)
        # nC = max(abs(C.min()), abs(C.max()))
        if C.shape == (1, 1):   # if number then makes C = 1
            Q *= np.sign(C.flat[0])
            C = np.ones((1, 1))
        else:
            C = C / nC
        Dr = C.shape[0]
        Q = np.reshape(Q, [Dl, d, Dr])
        return Q, C, nC, Dr

    def _mps_decompose_CA(self, A):
        r"""Splits QR A -> CA."""
        Dl, d, Dr = A.shape
        Q, C = qr(np.reshape(A, [Dl, d * Dr]).T)
        nC = nfactor(C)
        # nC = max(abs(C.min()), abs(C.max()))
        if C.shape == (1, 1):
            Q *= np.sign(C.flat[0])
            C = np.ones((1, 1))
        else:
            C = (C.T) / nC
        Dl = C.shape[1]
        Q = np.reshape(Q.T, [Dl, d, Dr])
        return C, Q, nC, Dl

    def _mps_truncateC(self, C, Dmax, tol):
        r"""Truncates C using svd."""
        U, S, V = svd(C)
        tol = max(np.finfo(float).eps, tol)
        keep = min(sum(S > (S[0] * tol)), Dmax)
        projR = V[:keep, :]
        projL = U[:, :keep]
        discarded = np.sqrt(sum(S[keep:]**2)) / S[0]
        S = np.diag(S[:keep])
        return projL, S, projR, keep, discarded

#
# ==============================================================================
#


class MPO:
    r"""
    Matrix Product Operator (MPO).

    Args:
        d (int): dimensions of local sites (ingoing).
        dout (int): dimensions of local sites (outgoing). If 'None', use 'd'.
        L (int): number of sites.
    """
    def __init__(self, d=2, dout=None, L=2):
        self.L = L  # length
        self.W = [1] * L  # mpo tensors
        self.support = [0] * L  # which are nontrivial
        if type(d) == int:
            d = [d]
        multid = (L + len(d) - 1) // len(d)
        self.din = d * multid            # local dim of |psi>
        if dout is None:
            self.dout = self.din
        else:
            if type(dout) == int:
                dout = [dout]
            multid = (L + len(dout) - 1) // len(dout)
            self.dout = dout * multid    # local dim of <psi|
        for n in range(self.L):
            self.reset_site(n)

    def set_from_block(self, M, n):
        r"""
        Sets MPO tensor from block matrix.
        """
        self.support[n] = 1
        self.W[n] = self._block_matrix_to_mpo(M, self.dout[n], self.din[n])

    def reset_site(self, n):
        r"""
        Sets MPO tensor to identity.
        """
        self.support[n] = 0
        self.W[n] = self._mpo_identity(self.dout[n], self.din[n])

    def set_direct(self, W, n):
        r"""
        Sets MPO tensor directly (it have to follow convention of the legs ordering).
        """
        self.support[n] = 1
        self.W[n] = W
        self.dout[n], self.din[n] = self._mpo_get_d(W)

    def _block_matrix_to_mpo(self, M, dout, din):
        r"""Reshapes block matrix into MPO tensor."""
        sout, sin = M.shape
        H = np.transpose(np.reshape(M, [sout // dout, dout, sin // din, din]), (0, 1, 2, 3))
        return H

    def _mpo_identity(self, dout, din):
        r"""Identity MPO tensor."""
        II = np.zeros((dout, din))
        np.fill_diagonal(II, 1)
        return np.transpose(np.reshape(II, [1, dout, 1, din]), (0, 1, 2, 3))

    def _mpo_get_d(self, W):
        r"""Dim of physical legs of MPO tensor."""
        din = W.shape[3]
        dout = W.shape[1]
        return dout, din
