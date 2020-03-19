"""
The main module of the package.
It puts together the heuristics to solve
Ising-type optimization problems defined on a quasi-2d lattice (including e.g., chimera graph),
or a Random Markov Field on 2d rectangular lattice.
"""

import otn2d.mps as mps
import numpy as np
import itertools
import logging
import scipy.sparse as sparse
import scipy.linalg as splinalg
import time
import matplotlib.pyplot as plt


def load_Jij(file_name):
    """
    Loads couplings of the Ising model from a file.

    Args:
        file_name (str): a path to file with coupling written in the format, i j Jij.

    Returns:
        a list of Jij couplings.
    """
    J = np.loadtxt(file_name)
    J = [[int(l[0]), int(l[1]), float(l[2])] for l in J]
    return J


def minus_Jij(J):
    """ 
    Change sign of all couplings Jij -> -Jij.
    """
    return [[l[0], l[1], -l[2]] for l in J]


def Jij_f2p(J):
    """
    Change 1-base indexig to 0-base indexing in a list of Jij.
    """
    return [[l[0]-1, l[1]-1, l[2]] for l in J]


def Jij_p2f(J):
    """
    Change 0-base indexig to 1-base indexing in a list of Jij.
    """
    return [[l[0]+1, l[1]+1, l[2]] for l in J]


def energy_Jij(J, states):
    """
    Calculates energies from bit_strings for Ising model.

    Args: 
        J (list): list of couplings
        states (nparray): 1 (spin up), 0 (spin down)

    Returns:
        Energies for all states.
    """
    
    L = len(states[0])

    ii, jj, vv = zip(*J)
    JJ = sparse.coo_matrix((vv, (ii, jj)), shape=(L, L))
    JJ = sparse.triu(JJ) + sparse.tril(JJ, -1).T   # makes the coupling matrix upper triangular

    st = 2*np.array(states)-1
    Ns, dNs = st.shape[0], 1024
    Eng = np.zeros(Ns, dtype=float)
    for nn in range(0, Ns, dNs):
        ind = np.arange(nn, min(nn+dNs, Ns))
        Eng[ind] = np.sum(np.dot(st[ind], sparse.triu(JJ, 1).toarray())*st[ind], 1) + np.dot(st[ind], JJ.diagonal())
    return Eng


def load(file_name):
    """Loads solution of an instance from a file."""
    d = np.load(file_name, allow_pickle=True)
    Nx = d.item().get('Nx')
    Ny = d.item().get('Ny')
    Nc = d.item().get('Nc')
    beta = d.item().get('beta')
    mode = d.item().get('mode')
    ins = otn2d(mode=mode, Nx=Nx, Ny=Ny, Nc=Nc, beta=beta)
    ins.energy = d.item().get('energy')
    ins.probability = d.item().get('probability')
    ins.degeneracy = d.item().get('degeneracy')
    ins.states = d.item().get('states')
    ins.discarded_probability = d.item().get('discarded_probability')
    ins.negative_probability = d.item().get('negative_probability')
    ins.ind0 = d.item().get('ind')
    ins.adj = np.zeros((0,0))
    try:
        ins.excitations_encoding = d.item().get('excitations_encoding')
        ins.d = d.item().get('d')
        ins.invd = d.item().get('invd')
        ins.el = d.item().get('el')
        ins.free_d = d.item().get('free_d')
        if ins.excitations_encoding > 1:
            if ins.mode == 'Ising':
                ins.adj = d.item().get('adj')
            else:
                ins.adj = []
            ins._reset_adj(J=ins.adj, Nx=ins.Nx, Ny=ins.Ny, ind=ins.ind0)
    except TypeError:
        pass
    return ins


class otn2d:
    """
    Contains instance to be solved.

    Args:
        mode (str):
            ``'Ising'`` assumes Ising-type representation of the problem
            with the cost function E(s) = sum_{i<j} Jij si sj + sum_i Jii si.
            The couplings Jij form a 2d rectangular (Nx x Ny) lattice of elementary cells with Nc spins in each cell. 
            Spin index i = Nx*Nc*k+Nc*l+m, with k=0:Ny-1, l=0:Nx-1, m=0:Nc-1 (zero-based indexing is used).

            Spins which are not active (with all Jij equal 0), are automatically recognized.  They not taken into account during the search.

            ``'RMF'`` assumes a Random Markov Field type model on a 2d rectangular (Nx x Ny) lattice
            with cost function E = sum_{<i,j>} E(s_i, s_j) + sum_i E(s_i) and nearest-neighbour interactions only.

        ints Nx, Ny, Nc : defining lattice.
        beta (float): sets the inverse temperature used during the search. 
            It is the most relevant parameter, with larger beta allowing to better zoom in on low energy states, 
            but making tensor network contraction numerically less stable.
        J (others): couplings.
            For mode ``'Ising'``, it should be a list of [i, j, Jij].
            For mode ``'RMF'``, it should be a 

    Returns:
        Obtained results are stored as instance attributes (see below).

    Attributes:
        energy: energy(ies) of states obtained from searching or sampling.
        probability: log10 of the corresponding calculated probabilities.
        degeneracy: degeneracy of the ground state.
        states: states of clusters from searching or sampling. 
            In the mode ``'Ising'`` use method `binary_states` to get the bit strings.
        discarded_probability: log10 of the largest probability discarded during the search.
        negative_probability: a potential red flag. 
            Takes values in [-1,0]. 
            A negative value means that some conditional probabilities calculated from tensor network contraction were negative. 
            This indicates that the contraction was not fully numerically stable. 
            The worst case is shown. 
            The value shows the ratio of negative and positive conditional probabilities for one cluster and partial configuration.
        logger: logger
        excitations_encoding: if the low-energy spectrum was searched, this is the index of the merging approach, which was used.
        el: tree representing the hierarchy of droplets, as obtained during merging.
        d: dictionary of droplets' shapes.
    """

    def __init__(self, mode='Ising', Nx=4, Ny=4, Nc=8, beta=1, J=None):
        self.mode = mode
        self.beta = beta
        self.Nx_model = Nx # original shape
        self.Ny_model = Ny
        self.Nx = Nx  # perhaps can be rotated
        self.Ny = Ny
        if self.mode == 'Ising':
            self.Nc = Nc
            if self.Nc <= 8:
                self.indtype = np.int8
            else:
                raise('Single cluster is too large.')
        elif self.mode == 'RMF':
            self.Nc = 1
            self.indtype = np.int8
        self.L = Nx*Ny*Nc
        self.order = np.arange(self.Nx*self.Ny)  # order of clusters
        self.order_i = np.arange(self.Nx*self.Ny)  # inverse order of clusters
        self.logger = logging.getLogger('otn2d')
        self.energy = np.zeros(0)
        self.probability = np.zeros(0)
        self.rotation = 0
        self.degeneracy = 0
        self.states = np.zeros((0, Nx*Ny), dtype=self.indtype)
        if J is not None:
            self._import_J(J)
            self._divide_couplings()

    def save(self, file_name):
        """
        Saves solution of the instance to a file.

        Args:
            file_name (str): where to save
        """
        d = {}
        d['mode'] = self.mode
        d['rotation'] = self.rotation
        d['energy'] = self.energy
        d['probability'] = self.probability
        d['degeneracy'] = self.degeneracy
        d['states'] = self.states
        d['discarded_probability'] = self.discarded_probability
        d['negative_probability'] = self.negative_probability
        d['Nx'] = self.Nx_model
        d['Ny'] = self.Ny_model
        d['Nc'] = self.Nc
        d['beta'] = self.beta
        d['ind'] = self.ind0
        try:
            d['excitations_encoding'] = self.excitations_encoding
            d['d'] = self.d
            d['invd'] = self.invd
            d['el'] = self.el
            d['free_d'] = self.free_d
            if self.excitations_encoding > 1:
                if self.mode == 'Ising':
                    d['adj'] = sparse.csr_matrix(self.adj)
        except AttributeError:
            pass
        np.save(file_name, d)

    # def plot(self, name='', ind=0, show=True):
    #     if ind < len(self.states):
    #         plt.matshow(self.states[ind, :].reshape(self.Ny, self.Nx).T)
    #         if show:
    #             plt.show()
    #         plt.savefig(name+"_st="+str(ind)+".png")
    #     return 0

    def show_properties(self):
        """
        Displays basic properties of the lattice.
        """
        print("L:     ", self.L)
        print("Ny:    ", self.Ny)
        print("Nx:    ", self.Nx)
        print("Beta:  ", self.beta)

    def show_solution(self, state=False):
        """
        Shows the solution found and some info from search and contraction.
        """
        if len(self.energy) > 0:
            print("Energy            : %4.6f" % self.energy[0])
            print("Degeneracy        : %2d" % self.degeneracy)
            print("log2(Probability) : %0.2e" % self.probability[0])
            print("Discarder log2(P) : %0.2e" % self.discarded_probability)
            print("Min P (err)       : %0.2e" % self.negative_probability)
            print("# of states       : %1d" % len(self.energy))
            print("Rotation/direction: %1d" % self.rotation)
            if state:
                print(self.states[0])
        else:
            print('No solution to show')

    def binary_states(self, number=-1):
        """
        Returns states in binary form.

        1 - spin up (si=+1); 0 - spin down (si=-1); 2 - inactive spin

        Args:
            number (int): Maximal number of states to be returned. -1 returns all states.

        Returns:
            nparray
        """
        ns = self.states.shape[0]
        if number < 0:
            ns += number + 1
        else:
            ns = min(number, ns)
        if self.mode == 'Ising':
            decoded_states = np.zeros((ns, self.L), dtype=np.int8) + 2
            kk = -1
            for ny in range(self.Ny_model):
                for nx in range(self.Nx_model):
                    kk += 1
                    decode = self._cluster_configurations(len(self.ind0[ny][nx]))
                    decoded_states[:, self.ind0[ny][nx]] = decode[self.states[:ns, kk]]
            return decoded_states
        elif self.mode == 'RMF':
            return self.states[:ns]
    
    def rotate_graph(self, rot=1):
        """
        Rotate 2d graph by 90 degrees.

        It is used to contract peps and search from other directions.
        Rotations are cumulative.
        """
        for _ in range(rot):
            self.rotation +=1
            order_full = np.arange(self.L)
            order = np.arange(self.Nx * self.Ny)
            order_i = np.arange(self.Nx * self.Ny)
            for nx in range(self.Nx):
                for ny in range(self.Ny):
                    ii = ny*self.Nc*self.Nx + nx*self.Nc + np.arange(self.Nc)
                    jj = (self.Nx - nx - 1)*self.Nc*self.Ny + ny*self.Nc + np.arange(self.Nc)
                    order_full[ii] = jj
                    ii = ny*self.Nx + nx
                    jj = (self.Nx - nx - 1)*self.Ny + ny
                    order[ii], order_i[jj] = jj, ii
            self.Nx, self.Ny = self.Ny, self.Nx
            self.J = self.J[order_full, :][:, order_full]
            self.J = sparse.triu(self.J) + sparse.tril(self.J, -1).T
            self.order = order_i[self.order]
        self.order_i[self.order] = np.arange(self.Nx * self.Ny)
        self.rotation = np.mod(self.rotation, 4)
        self._divide_couplings()

    def precondition(self, mode='balancing', steps=2, beta_cond=[], Dmax_cond=[], max_scale=1024, tolS=1e-16, tolV=1e-10, max_sweeps=4):
        """
        Apply the preconditioning procedure.
        
        Args:
            mode (str): Type of heuristics used. For now, only 'balancing' trick is implemented.
            steps (int): number of smaller betas used (if they are not provided explicitly).
            beta_cond (list of floats): beta's used to search for preconditioning.
            Dmax_cond (list of ints): corresponding maximal bond dimensions used in boundary MPS.
            max_scale (float): bound on local rescaling used in one step.
            tolS (float): truncate smaller singular values during svd in boundary MPS.
            tolV (float): condition for overlap convergence during one sweep in boundary MPS.
            max_sweeps (int): maximal number of sweeps of variational compression.             
        """
        if mode is 'balancing':
            if not beta_cond:
                beta_cond = [self.beta * 2.**(nn-steps) for nn in range(steps)]
            if not Dmax_cond:
                Dmax_cond = [8] * len(beta_cond)  # default D for conditioning is 8
            main_beta = self.beta
            for nn in range(len(beta_cond)):
                self.beta = beta_cond[nn]
                self.logger.info('Preconditioning with beta = %.2f', self.beta)
                keep_time = time.time()
                self._update_conditioning(direction='lr', Dmax=Dmax_cond[nn], tolS=tolS, tolV=tolV, max_sweeps=max_sweeps, max_scale=max_scale)
                self._update_conditioning(direction='ud', Dmax=Dmax_cond[nn], tolS=tolS, tolV=tolV, max_sweeps=max_sweeps, max_scale=max_scale)
                self.logger.info('Elapsed: %.2f seconds', time.time() - keep_time)
            self.beta = main_beta

    def search_ground_state(self, M=2**10, relative_P_cutoff=1e-6,
                            min_dEng=1e-12,
                            Dmax=32, tolS=1e-16,
                            tolV=1e-10, max_sweeps=20):
        """
        Searches for the most probable state (ground state) on a quasi-2d graph.

        Merge matching configurations during branch-and-bound search going line (ny=0:Ny-1) by line. 
        It keeps track of GS degeneracy, distinguishing different energies with precision min_dEng.
        Probabilities kept as log10. Results are stored as instance attributes.

        Args:
            M (int): maximal number of branches (partial configurations) that are kept during the search.
            relative_P_cutoff (float): do not keep branches with a probability smaller by that factor comparing with most probable one.
            min_dEng (float): precision below which two states (perhaps partial) are considered to have the same energy.
            Dmax (int): maximal bond dimensions used in boundary MPS.
            tolS (float): truncate smaller singular values during svd in boundary MPS.
            tolV (float): condition for overlap convergence during one sweep in boundary MPS.
            max_sweeps (int): maximal number of sweeps of variational compression.    

        Returns: 
            The lowest energy found.
        """

        keep_total_time, keep_time = time.time(), time.time()
        # Prepare environments for layers from bottom
        self.logger.info('Searching ground state with beta = %.2f', self.beta)

        self.logger.info('Preprocesing ... ')
        self._setup_rhoT(Dmax=Dmax, tolS=tolS, tolV=tolV, max_sweeps=max_sweeps)
        self.logger.info('Elapsed: %.2f seconds', time.time() - keep_time)

        # Initilise
        # virtual indices from partial configurations
        vind = np.zeros((1, self.Nx+1), dtype=self.indtype)
        # (partial) spin configurations
        states = np.zeros((1, self.Nx*self.Ny), dtype=self.indtype)
        # energies / probabilities / deg of partial configurations
        Eng, prob, deg = np.zeros(1), np.zeros(1), np.ones(1, dtype=int)
        # largest discarded probability, smallest calculated prob
        pd_max, globalmin = -np.inf, 0.

        # Start searching
        self.logger.info('Searching ... ')
        for ny in range(self.Ny):  # consider row ny
            keep_time = time.time()
            self.logger.info('Row %d / %d', ny+1, self.Ny)

            # setup right environments to calculate prob
            RRl = self._setup_RR(vind, ny)
            RLl = {(): np.ones(1)}  # and left

            for nx in range(self.Nx):    # consider site nx in the row
                # number of: states in the cluster, considered states
                block_states, cons_states = self.N[ny][nx], prob.size

                W = self._peps_tensor(ny, nx)  # PEPS tensor
                newprob = np.zeros((cons_states,  block_states))
                minprob = np.zeros(cons_states)
                for kk in range(cons_states):
                    tind = tuple(vind[kk])
                    AA = W[:, tind[nx], :, :, tind[nx+1]]
                    newprob[kk], minprob[kk] = self._calculate_Pn(AA, RLl[tind[:nx]], self.rhoT[ny+1].A[nx], RRl[self.Nx-nx-1][tind[nx+2:]] )

                newprob = np.log2(newprob)
                newprob += prob[:, np.newaxis]  # use conditional probability to calculate probability of partial configuration
                newprob = np.reshape(newprob, cons_states*block_states)

                minprob, maxprob = np.min(minprob), np.max(newprob)

                cutoff = maxprob + np.log2(relative_P_cutoff)
                order = np.arange(newprob.size)

                # cutoff on which probabilities are kept
                keep = max((newprob[order] > cutoff).sum(), 1)
                if keep < order.size:
                    or2 = newprob[order].argpartition(-keep-1)
                    pd_max = max(pd_max, newprob[order[or2[-keep-1]]])
                    order = order[or2[-keep:]]
                    newprob = newprob[order]  # keep largest probabilities

                keep_more = 4
                if newprob.size > keep_more*M:  # looks for max_state largest probabilities
                    or2 = newprob.argpartition(-keep_more*M-1)
                    pd_max = max(pd_max, newprob[or2[-keep_more*M-1]])
                    order = order[or2[-keep_more*M::]]
                    newprob = newprob[or2[-keep_more*M::]]

                prob = newprob  # keep largest probabilities
                # inds = which previous states
                # indc = and state at the considered cluster (site)
                inds, indc = order // block_states, np.mod(order, block_states)
                states = states[inds]
                states[:, ny*self.Nx+nx] = indc[:]
                vind = vind[inds]
                # update corresponding virtual indices
                vind[:, nx] = self._ind_bond_down(indc, ny, nx)
                vind[:, nx+1] = self._ind_bond_right(indc, ny, nx)
                Eng = Eng[inds]
                Eng += self._update_Eng(states, ny, nx)
                deg = deg[inds]

                # merge configurations where vind is the same
                seen = {}
                seenind = {}

                for kk in range(order.size):
                    tind = tuple(vind[kk])
                    if tind in seen:
                        if Eng[kk] + min_dEng < Eng[seen[tind]]:
                            del seenind[seen[tind]]
                            seen[tind] = kk
                            seenind[kk] = [kk]
                        elif abs(Eng[kk] - Eng[seen[tind]]) < min_dEng:
                            seenind[seen[tind]].append(kk)
                    else:
                        seen[tind] = kk
                        seenind[kk] = [kk]

                uni = np.fromiter(seen.values(), dtype=np.int)
                vind = vind[uni]
                states = states[uni]
                Eng = Eng[uni]
                degn, probn = np.zeros(uni.size, dtype=int), np.zeros(uni.size)
                for kk in range(uni.size):
                    ind = np.array(seenind[uni[kk]], dtype=int)
                    probn[kk] = np.median(prob[ind])
                    degn[kk] = np.sum(deg[ind])
                prob, deg = probn, degn

                # looks for max_state largest probabilities
                if prob.size > M:
                    order = prob.argpartition(-M-1)
                    pd_max = max(pd_max, prob[order[-M-1]])
                    order = order[-M::]
                    vind = vind[order]
                    states = states[order]
                    prob = prob[order]
                    Eng = Eng[order]
                    deg = deg[order]

                RLnew = {}  # update left environment
                for one_ind in vind:
                    tind = tuple(one_ind[:(nx+1)])
                    if tind not in RLnew:
                        tempR = np.dot(RLl[tind[:-1]], self.rhoT[ny+1].A[nx][:, tind[-1], :])
                        tempR *= (1/mps.nfactor(tempR))
                        RLnew[tind] = tempR
                RLl = RLnew
                # self.logger.info('Prob: min/ min kept/ max: %0.2e / %0.2e / %0.2e ', minprob, np.min(prob), maxprob)
                globalmin = min(globalmin, minprob)

            self.logger.info('Elapsed: %.2f seconds', time.time() - keep_time)
            vind[:, 1:] = vind[:, :-1]  # reset vind before going to next layer
            # shifting the one corresponding to "central" bond to begining
            vind[:, 0] = 0

        self.logger.info('Elapsed search total: %.2f seconds', time.time() - keep_total_time)
        self.energy = Eng
        self.degeneracy = deg[0]
        self.states = states[:, self.order]
        self.probability = prob
        self.discarded_probability = pd_max
        self.negative_probability = min(globalmin, 0)
        return Eng

    def gibbs_sampling(self, M=2**10,
                       Dmax=32, tolS=1e-15,
                       tolV=1e-10, max_sweeps=20):
        """
        Samples from the Boltzman distribution on a quasi-2d graph.

        Probabilities kept as log10. Results are stored as instance attributes.

        Args:
            M (int): number of configurations.
            Dmax (int): maximal bond dimensions used in boundary MPS.
            tolS (float): truncate smaller singular values during svd in boundary MPS.
            tolV (float): condition for overlap convergence during one sweep in boundary MPS.
            max_sweeps (int): maximal number of sweeps of variational compression.    

        Returns: 
            Sampled energies.
        """

        keep_total_time, keep_time = time.time(), time.time()
        self.logger.info('Preprocesing ... ')
        # prepare environments for layers from bottom
        self._setup_rhoT(Dmax=Dmax, tolS=tolS, tolV=tolV, max_sweeps=max_sweeps)
        self.logger.info('Elapsed: %.2f seconds', time.time() - keep_time)

        # Initilise
        vind = np.zeros((M, self.Nx+1), dtype=int)  # virtual indices from partial configurations
        states = np.zeros((M, self.Nx*self.Ny), dtype=int)  # partial spin configurations
        Eng = np.zeros(M)
        globalmin = 1.

        self.logger.info('Sampling ... ')

        for ny in range(self.Ny):  # consider row ny
            keep_time = time.time()
            self.logger.info('Row %d / %d', ny+1, self.Ny)

            RRl = self._setup_RR(vind, ny)  # setup right environments to calculate prob
            RLl = {(): np.ones(1)}  # and left

            for nx in range(self.Nx):  # consider site nx in the row
                block_states = self.N[ny][nx]  # number of states in the cluster (site)
                W = self._peps_tensor(ny, nx)
                newprob = np.zeros((M,  block_states))  # to update probabilities
                minprob = np.zeros(M)
                seen = {}  # some diagrams repeat itself
                for kk in range(M):
                    tind = tuple(vind[kk])
                    if tind in seen:
                        newprob[kk] = newprob[seen[tind]]
                        minprob[kk] = minprob[seen[tind]]
                    else:   # calculate conditional probability
                        seen[tind] = kk
                        AA = W[:, tind[nx], :, :, tind[nx+1]]
                        newprob[kk], minprob[kk] = self._calculate_Pn(AA, RLl[tind[:nx]], self.rhoT[ny+1].A[nx], RRl[self.Nx-nx-1][tind[nx+2:]])

                minprob = np.min(minprob)  # np.min(newprob)#
                maxprob = np.max(newprob)  # np.min(newprob)#

                newprob = newprob.cumsum(axis=1)
                rr = np.random.rand(M)
                indc = np.zeros(M, dtype=int)

                for kk in range(M):
                    indc[kk] = np.searchsorted(newprob[kk], rr[kk])

                states[:, ny*self.Nx+nx] = indc[:]
                # update corresponding virtual indices
                vind[:, nx] = self._ind_bond_down(indc, ny, nx)
                vind[:, nx+1] = self._ind_bond_right(indc, ny, nx)
                Eng += self._update_Eng(states, ny, nx)
                RLnew = {}   # update left environment
                for one_ind in vind:
                    tind = tuple(one_ind[:(nx+1)])
                    if tind not in RLnew:
                        tempR = np.dot(RLl[tind[:-1]], self.rhoT[ny+1].A[nx][:, tind[-1], :])
                        tempR *= (1/mps.nfactor(tempR))
                        RLnew[tind] = tempR
                RLl = RLnew
                # self.logger.info('Prob: min/ max:  %0.2e / %0.2e ', minprob, maxprob)
                globalmin = min(globalmin, minprob)

            self.logger.info('Elapsed: %.2f seconds', time.time() - keep_time)
            vind[:, 1:] = vind[:, :-1]  # reset vind before going to next layer
            vind[:, 0] = 0  # by shifting the one corresponding to "central" bond to begining

        self.logger.info('Elapsed total: %.2f seconds', time.time()-keep_total_time)
        self.energy = Eng
        self.degeneracy = 0
        self.states = states[:, self.order]
        self.probability = np.zeros(1)
        self.discarded_probability = 0
        self.negative_probability = min(globalmin, 0)
        return Eng

    def search_low_energy_spectrum(self, excitations_encoding=1,
                                   M=2**10, relative_P_cutoff=1e-6,
                                   max_dEng=0., lim_hd=0,
                                   min_dEng=1e-12,
                                   Dmax=32, tolS=1e-16,
                                   tolV=1e-10, max_sweeps=20):
        """
        Searches for low-energy spectrum on a quasi-2d graph.

        Merge matching configurations during branch-and-bound search going line (ny=0:Ny-1) by line.
        Information about excited states (droplets) is collected during merging, which allows reconstructing the low-energy spectrum.
        It keeps track of GS degeneracy, distinguishing different energies with precision min_dEng.
        Probabilities kept as log10. Results are stored as instance attributes.

        Args:
            excitations_encoding (int): Approach used to define independent/elementary droplets

                ``1`` Independence determined based on the order of snake spanning 2d lattice line by line.
                It gives a one-to-one correspondence between the low-energy spectrum
                and the stored excitation structure
                (assuming, that the search itself was successful).

                ``2`` Independent and elementary droplets determined based on adjacency matrix (i.e., graph of interactions).
                Droplets which are not single-connected are discarded during merging, 
                leaving only the elementary, single-connected ones.
                A one-to-one correspondence between the low-energy spectrum and the stored excitation structure
                might be lost when merging configurations with many layers of the excitation hierarchy.

                ``3`` As in ``2`` but excitations are compressed to one layer of hierarchy.
                It is useful only for problems with a single basin of attraction and low-energy excitations of small sizes but retains a one-to-one correspondence between the low-energy spectrum and the stored excitation structure.
            M (int): maximal number of branches (partial configurations) that are kept during the search.
            relative_P_cutoff (float): do not keep branches with a probability smaller by that factor comparing with most probable one.
            max_dEng (float): maximal excitation energy being targeted.
            lim_hd (int): Lower limit of Hamming distance between states (while merging). Outputs fewer states.
            min_dEng (float): precision below which two states (perhaps partial) are considered to have the same energy.
            Dmax (int): maximal bond dimensions used in boundary MPS.
            tolS (float): truncate smaller singular values during svd in boundary MPS.
            tolV (float): condition for overlap convergence during one sweep in boundary MPS.
            max_sweeps (int): maximal number of sweeps of variational compression.    

        Returns:
            The lowest energy found.
        """
        self.excitations_encoding = excitations_encoding
        if excitations_encoding == 1:
            Eng = self._search_low_energy_spectrum_v1(M=M, relative_P_cutoff=relative_P_cutoff, max_dEng=max_dEng, lim_hd=lim_hd, min_dEng=min_dEng, Dmax=Dmax, tolS=tolS, tolV=tolV, max_sweeps=max_sweeps)
        elif excitations_encoding == 2:
            Eng = self._search_low_energy_spectrum_v2(M=M, relative_P_cutoff=relative_P_cutoff, max_dEng=max_dEng, lim_hd=lim_hd, min_dEng=min_dEng, Dmax=Dmax, tolS=tolS, tolV=tolV, max_sweeps=max_sweeps)
        elif excitations_encoding == 3:
            Eng = self._search_low_energy_spectrum_v3(M=M, relative_P_cutoff=relative_P_cutoff, max_dEng=max_dEng, lim_hd=lim_hd, min_dEng=min_dEng, Dmax=Dmax, tolS=tolS, tolV=tolV, max_sweeps=max_sweeps)
        else:
            raise('Available droplets handling strategies are excitations_encoding = 1,2,3.')
        return Eng

    def _search_low_energy_spectrum_v1(self, M=2**10, relative_P_cutoff=1e-6, max_dEng=0., lim_hd=0, min_dEng=1e-12, Dmax=32, tolS=1e-16, tolV=1e-10, max_sweeps=20):
        """
        Searching for most probable states on quasi-2d graph.
        Merge and keeps track of excitation.
        Independence determined based on order of snake spanning 2d lattice.
        """

        keep_total_time, keep_time = time.time(), time.time()
        #  prepare environments for layers from bottom
        self.logger.info('Preprocesing ... ')
        self._setup_rhoT(Dmax=Dmax, tolS=tolS, tolV=tolV, max_sweeps=max_sweeps)
        self.logger.info('Elapsed: %.2f seconds', time.time() - keep_time)

        #  Initilise
        vind = np.zeros((1, self.Nx+1), dtype=self.indtype)  # virtual indices from partial configurations
        states = np.zeros((1, self.Nx*self.Ny), dtype=self.indtype)  # (partial) spin configurations
        Eng, prob, deg = np.zeros(1), np.zeros(1), np.ones(1, dtype=int)  # energies / probabilities / deg of partial configurations
        pd_max, globalmin = -np.inf, 1.  # largest discarded probability, smallest calculated prob
        self._exc_initialise()

        #  Start searching
        self.logger.info('Searching ... ')
        for ny in range(self.Ny):  # consider row ny
            keep_time = time.time()
            self.logger.info('Layer %d / %d', ny+1, self.Ny)

            RRl = self._setup_RR(vind, ny)  # setup right environments to calculate prob
            RLl = {(): np.ones(1)}  # and left

            for nx in range(self.Nx):  # consider site nx in the row
                # number of: states in the cluster, considered states
                block_states, cons_states = self.N[ny][nx], prob.size
                W = self._peps_tensor(ny, nx)  # PEPS tensor
                newprob, minprob = np.zeros((cons_states,  block_states)), np.zeros(cons_states)
                for kk in range(cons_states):
                    tind = tuple(vind[kk])
                    AA = W[:, tind[nx], :, :, tind[nx+1]]
                    newprob[kk], minprob[kk] = self._calculate_Pn(AA, RLl[tind[:nx]], self.rhoT[ny+1].A[nx], RRl[self.Nx-nx-1][tind[nx+2:]])

                newprob = np.log2(newprob)

                newprob += prob[:, np.newaxis]  # use conditional probability to calculate probability of partial configuration
                newprob = np.reshape(newprob, cons_states*block_states)
                minprob, maxprob = np.min(minprob), np.max(newprob)
                cutoff = maxprob + np.log2(relative_P_cutoff)

                order = np.arange(newprob.size)

                # cutoff on which probabilities are kept
                keep = max((newprob[order] > cutoff).sum(), 1)
                if keep < order.size:
                    or2 = newprob[order].argpartition(-keep-1)
                    pd_max = max(pd_max, newprob[order[or2[-keep-1]]])
                    order = order[or2[-keep:]]
                    newprob = newprob[order]  # keep largest probabilities

                keep_more = 4
                if newprob.size > keep_more*M:  # looks for max_state largest probabilities
                    or2 = newprob.argpartition(-keep_more*M-1)
                    pd_max = max(pd_max, newprob[or2[-keep_more*M-1]])
                    order = order[or2[-keep_more*M::]]
                    newprob = newprob[or2[-keep_more*M::]]

                prob = newprob  # keep largest probabilities
                # inds = which previous states
                # indc = and state at the considered cluster (site)
                inds, indc = order // block_states, np.mod(order, block_states)
                states = states[inds]
                states[:, ny*self.Nx+nx] = indc
                vind = vind[inds]
                # update corresponding virtual indices
                vind[:, nx] = self._ind_bond_down(indc, ny, nx)
                vind[:, nx+1] = self._ind_bond_right(indc, ny, nx)
                Eng = Eng[inds]
                Eng += self._update_Eng(states, ny, nx)
                deg = deg[inds]

                # merge configurations where vind is the same
                seen = {}  # identify similar configurations
                for kk in range(order.size):
                    tind = tuple(vind[kk])
                    if tind in seen:
                        seen[tind].append((kk, Eng[kk], prob[kk]))
                    else:
                        seen[tind] = [(kk, Eng[kk], prob[kk])]

                lseen = len(seen)
                uni = np.zeros(lseen, dtype=int)  # index of "unique" (main) configurations
                probn = np.zeros(lseen)  # their probabilities
                degn = np.zeros(lseen, dtype=int)  # their degeneracy
                el = []  # new list collecting all excitations (excitations list) of all branches

                # ind  = indexing new main branches
                for ind, confs in enumerate(seen.values()):  # merge configuration
                    mEng = np.inf  # minimal Energy
                    for ii, En, pr in confs:  # for given vind find min energy
                        if En+min_dEng < mEng:
                            ui = ii
                            mEng = En
                            cprob = [pr]
                            cdeg = deg[ui]  # degeneracy of smallest energy
                        elif np.abs(En - mEng) < min_dEng:
                            cprob.append(pr)
                            cdeg += deg[ii]
                    probn[ind] = np.median(cprob)
                    uni[ind] = ui
                    degn[ind] = cdeg
                    # new branch excitation list <- copy from old
                    bel = self.el[inds[ui]][:]
                    # add other merged branches as new excitations
                    for ii, En, pr in confs:  # merge other excitations
                        conf_dEng = En - mEng
                        if (conf_dEng <= max_dEng) and (ii != ui):
                            # where the states differ
                            dstate = np.bitwise_xor(states[ui], states[ii])
                            dpos = dstate.nonzero()[0]
                            dstate = dstate[dpos]
                            hamming = self._exc_hd(dstate)
                            if hamming >= lim_hd:
                                dfirst, dlast = dpos[0], self.Nx*ny+nx
                                dP = pr-probn[ind]
                                di = self._exc_add_to_d(dpos, dstate)  # dict index
                                sel = []  # sub-excitations list
                                # add sub-excitations
                                for sne in self.el[inds[ii]]:
                                    # (sne.last >= dfirst) ; sne[0][0] + conf_dEng == total exc eng
                                    if (sne[0][3] >= dfirst) and (sne[0][0] + conf_dEng <= max_dEng):
                                        sel.append(self._exc_cut_energy(sne, max_dEng - (sne[0][0] + conf_dEng)))
                                ne = ((conf_dEng, di, dfirst, dlast, dP), tuple(sel))
                                bel.append(ne)
                    el.append(bel)  # creates new list of excitations

                prob = probn
                deg = degn
                vind = vind[uni]
                states = states[uni]
                Eng = Eng[uni]
                self.el = el

                RLnew = {}  # update left environment
                for one_ind in vind:
                    tind = tuple(one_ind[:(nx+1)])
                    if tind not in RLnew:
                        tempR = np.dot(RLl[tind[:-1]], self.rhoT[ny+1].A[nx][:, tind[-1], :])
                        tempR *= (1/mps.nfactor(tempR))
                        RLnew[tind] = tempR
                RLl = RLnew
                # self.logger.info('Prob: min/ min kept/ max: %0.2e / %0.2e / %0.2e ', minprob, np.min(prob), maxprob)
                self._exc_clear_d()
                globalmin = min(globalmin, minprob)
            self.logger.info('Elapsed: %.2f seconds', time.time() - keep_time)
            vind[:, 1:] = vind[:, :-1]  # reset vind before going to next layer
            vind[:, 0] = 0  # by shifting the one corresponding to "central" bond to begining

        self.logger.info('Elapsed total: %.2f seconds', time.time() - keep_total_time)
        self.energy = Eng
        self.degeneracy = deg[0]
        self.states = states[:, self.order]
        self.probability = prob
        self.discarded_probability = pd_max
        self.negative_probability = min(globalmin, 0)
        self.el = self.el[0]
        # rotates d to original ordering befor rotation.
        for key, value in self.d.items():
            dpos, dstate = value
            dpos = self.order_i[dpos]
            order = dpos.argsort()
            dpos = dpos[order]
            dstate = dstate[order]
            self.d[key] = (dpos, dstate)
        return Eng

    def add_noise(self, amplitude=1e-7):
        """
        Adds a small random noise to the couplings.
        
        It should be used to remove accidental degeneracies while searching low-energy states in 'excitations_encoding' 2 or 3.

        Args:
            amplitude (float): the amplitude of the random noise.
        """
        self.logger.info('Adding noise to the coupling with ampliture %.2e', amplitude)
        if self.mode == 'Ising':
            nzr = self.J.nonzero()
            kk = ((np.random.rand(len(nzr[0]))*2-1)*amplitude)
            for i, j, k in zip(nzr[0], nzr[1], kk):
                self.J[i, j] += k
            self._divide_couplings()

        elif self.mode == 'RMF':
            func_new = {}
            for key, value in self.func.items():
                func_new[key] = value.copy()
                if len(value.shape) == 1:
                    func_new[key] += ((np.random.rand(value.shape[0])*2-1)*amplitude)

            self.func_clean = self.func
            self.func = func_new

    # def noise_clean(self):
    #     try:
    #         self.func = self.func_clean
    #     except NameError:
    #         pass

    def _search_low_energy_spectrum_v2(self, M=2**10, relative_P_cutoff=1e-6, Dmax=32, tolS=1e-16, tolV=1e-10, max_dEng=0., min_dEng=1e-12, max_sweeps=20, lim_hd=0):
        """
        Searching for most probable states on quasi-2d graph.   
        Merge and keeps track of excitation. 
        Independence determined based on adjecency graph.
        Might miss some states from higher levels of droplets hierarchy.
        """

        keep_total_time, keep_time = time.time(), time.time()
        #  prepare environments for layers from bottom
        self.logger.info('Preprocesing ... ')
        self._setup_rhoT(Dmax=Dmax, tolS=tolS, tolV=tolV, max_sweeps=max_sweeps)
        self.logger.info('Elapsed: %.2f seconds', time.time() - keep_time)

        #  Initilise
        vind = np.zeros((1, self.Nx+1), dtype=self.indtype)  # virtual indices from partial configurations
        states = np.zeros((1, self.Nx*self.Ny), dtype=self.indtype)  # (partial) spin configurations
        Eng, prob, deg = np.zeros(1), np.zeros(1), np.ones(1, dtype=int)  # energies / probabilities / deg of partial configurations
        pd_max, globalmin = -np.inf, 1.  # largest discarded probability, smallest calculated prob
        self._exc_initialise()  # initialise structure to keep excitations
        self._reset_adj(J=self.J, Nx=self.Nx, Ny=self.Ny, ind=self.ind)

        #  Start searching
        self.logger.info('Searching ... ')
        for ny in range(self.Ny):  # consider row ny
            keep_time = time.time()
            self.logger.info('Layer %d / %d', ny+1, self.Ny)

            RRl = self._setup_RR(vind, ny)  # setup right environments to calculate prob
            RLl = {(): np.ones(1)}  # and left

            for nx in range(self.Nx):  # consider site nx in the row
                # number of: states in the cluster, considered states
                block_states, cons_states = self.N[ny][nx], prob.size
                W = self._peps_tensor(ny, nx)  # PEPS tensor
                newprob, minprob = np.zeros((cons_states,  block_states)), np.zeros(cons_states)
                for kk in range(cons_states):
                    tind = tuple(vind[kk])
                    AA = W[:, tind[nx], :, :, tind[nx+1]]
                    newprob[kk], minprob[kk] = self._calculate_Pn(AA, RLl[tind[:nx]], self.rhoT[ny+1].A[nx], RRl[self.Nx-nx-1][tind[nx+2:]])

                newprob = np.log2(newprob)

                newprob += prob[:, np.newaxis]  # use conditional probability to calculate probability of partial configuration
                newprob = np.reshape(newprob, cons_states*block_states)
                minprob, maxprob = np.min(minprob), np.max(newprob)
                cutoff = maxprob + np.log2(relative_P_cutoff)

                order = np.arange(newprob.size)

                # cutoff on which probabilities are kept
                keep = max((newprob[order] > cutoff).sum(), 1)
                if keep < order.size:
                    or2 = newprob[order].argpartition(-keep-1)
                    pd_max = max(pd_max, newprob[order[or2[-keep-1]]])
                    order = order[or2[-keep:]]
                    newprob = newprob[order]   ## keep largest probabilities

                keep_more = 4
                if newprob.size > keep_more*M:  ## looks for max_state largest probabilities
                    or2 = newprob.argpartition(-keep_more*M-1)
                    pd_max = max(pd_max, newprob[or2[-keep_more*M-1]])
                    order = order[or2[-keep_more*M::]]
                    newprob = newprob[or2[-keep_more*M::]]

                prob = newprob  # keep largest probabilities
                # inds = which previous states
                # indc = and state at the considered cluster (site)
                inds, indc = order // block_states, np.mod(order, block_states)
                states = states[inds]
                states[:, ny*self.Nx+nx] = indc
                vind = vind[inds]
                # update corresponding virtual indices
                vind[:, nx] = self._ind_bond_down(indc, ny, nx)
                vind[:, nx+1] = self._ind_bond_right(indc, ny, nx)
                Eng = Eng[inds]
                Eng += self._update_Eng(states, ny, nx)
                deg = deg[inds]

                # merge configurations where vind is the same
                seen = {}  # identify similar configurations
                for kk in range(order.size):
                    tind = tuple(vind[kk])
                    if tind in seen:
                        seen[tind].append((kk, Eng[kk], prob[kk]))
                    else:
                        seen[tind] = [(kk, Eng[kk], prob[kk])]

                lseen = len(seen)
                uni = np.zeros(lseen, dtype=int)  # index of "unique" (main) configurations
                probn = np.zeros(lseen)  # their probabilities
                degn = np.zeros(lseen, dtype=int)  # their degeneracy
                el = []  # new list collecting all excitations (excitations list) of all branches

                # ind  = indexing new main branches
                for ind, confs in enumerate(seen.values()):  # merge configuration
                    mEng = np.inf  # minimal Energy
                    for ii, En, pr in confs:  # for given vind find min energy
                        if En+min_dEng < mEng:
                            ui = ii
                            mEng = En
                            cprob = [pr]
                            cdeg = deg[ui]  # degeneracy of smallest energy
                        elif np.abs(En - mEng) < min_dEng:
                            cprob.append(pr)
                            cdeg += deg[ii]
                    probn[ind] = np.median(cprob)
                    uni[ind] = ui
                    degn[ind] = cdeg
                    # new branch excitation list <- copy from old
                    bel = self.el[inds[ui]][:]
                    # add other merged branches as new excitations (possible with subexcitations)
                    for ii, En, pr in confs:
                        conf_dEng = En - mEng
                        if (conf_dEng <= max_dEng) and (ii != ui):  # if not the main branch and has small exc energy
                            dstate = np.bitwise_xor(states[ui], states[ii])
                            dpos = dstate.nonzero()[0]
                            dstate = dstate[dpos]
                            hamming = self._exc_hd(dstate)
                            # if not elementary then discard it
                            if (hamming >= lim_hd) and self._exc_elementary((dpos, dstate)):
                                di = self._exc_add_to_d(dpos, dstate)  # dict index
                                sel = []  # sub-excitations list
                                for sne in self.el[inds[ii]]:  # starts adding subexcitations
                                    if (sne[0][0] + conf_dEng <= max_dEng) and self._exc_overlap(di, sne[0][1]):
                                        # sel.append(sne) ## this line do not cut high energy excitations higher in the tree
                                        sel.append(self._exc_cut_energy(sne, max_dEng - (sne[0][0] + conf_dEng)))
                                        # add subexc if its total energy is small enough and it depends on the new one
                                ne = ((conf_dEng, di), tuple(sel))   # base to form new excitation [dict index, dE, subexc]
                                bel.append(ne)
                    el.append(bel)  # finish merging given branch

                prob = probn
                deg = degn
                vind = vind[uni]
                states = states[uni]
                Eng = Eng[uni]
                self.el = el

                RLnew = {}  # update left environment
                for one_ind in vind:
                    tind = tuple(one_ind[:(nx+1)])
                    if tind not in RLnew:
                        tempR = np.dot(RLl[tind[:-1]], self.rhoT[ny+1].A[nx][:, tind[-1], :])
                        tempR *= (1/mps.nfactor(tempR))
                        RLnew[tind] = tempR
                RLl = RLnew
                # self.logger.info('Prob: min/ min kept/ max: %0.2e / %0.2e / %0.2e ', minprob, np.min(prob), maxprob)
                self._exc_clear_d()
                globalmin = min(globalmin, minprob)  # collects information on smallest encountered probability (negative indicate error)
            self.logger.info('Elapsed: %.2f seconds', time.time() - keep_time)
            vind[:, 1:] = vind[:, :-1]  # reset vind before going to next layer
            vind[:, 0] = 0  # by shifting the one corresponding to "central" bond to begining

        self.logger.info('Elapsed total: %.2f seconds', time.time() - keep_total_time)
        self.energy = Eng
        self.degeneracy = deg[0]
        self.states = states[:, self.order]
        self.probability = prob
        self.discarded_probability = pd_max
        self.negative_probability = min(globalmin, 0)
        self.el = self.el[0]
        # rotates d and adjecency settings to the original ordering before rotation.
        for key, value in self.d.items():
            dpos, dstate = value
            dpos = self.order_i[dpos]
            order = dpos.argsort()
            dpos = dpos[order]
            dstate = dstate[order]
            self.d[key] = (dpos, dstate)
        self._reset_adj(J=self.J0, Nx=self.Nx_model, Ny=self.Ny_model, ind=self.ind0)
        return Eng

    def _search_low_energy_spectrum_v3(self, M=2*10, relative_P_cutoff=1e-6, Dmax = 32, tolS = 2.**(-52), tolV = 1e-12, max_dEng = 0, min_dEng = 1e-12, max_sweeps = 20, lim_hd = 0):
        """
        Searching for most probable states on quasi-2d graph.
        Merge and keeps track of excitation. 
        Independence determined based on adjecency graph.
        here forms only 1 layer of hierarchy of excitations. 
        """

        keep_total_time, keep_time = time.time(), time.time()

        # prepare environments for layers from bottom
        self.logger.info('Preprocesing ... ')
        self._setup_rhoT(Dmax = Dmax, tolS = tolS, tolV = tolV, max_sweeps = max_sweeps)
        self.logger.info('Elapsed: %.2f seconds', time.time() - keep_time)

        #Initilise
        vind       = np.zeros((1, self.Nx+1), dtype = self.indtype)  # virtual indices from partial configurations
        states     = np.zeros((1, self.Nx*self.Ny), dtype = self.indtype) # (partial) spin configurations
        Eng, prob, deg = np.zeros(1), np.zeros(1), np.ones(1, dtype = int)  # energies / probabilities / deg of partial configurations
        pd_max, globalmin = -np.inf, 1.   # largest discarded probability, smallest calculated prob
        self._exc_initialise() ## initialise structure to keep excitations
        self._reset_adj(J=self.J, Nx=self.Nx, Ny=self.Ny, ind=self.ind)

        #Start searching the tree
        self.logger.info('Searching ... ')
        for ny in range(self.Ny):  ## consider row ny
            keep_time = time.time()
            self.logger.info('Layer %d / %d', ny+1, self.Ny)

            RRl = self._setup_RR(vind, ny) ## setup right environments to calculate prob
            RLl = { () : np.ones(1) }     ## and left

            for nx in range(self.Nx):    ## consider site nx in the row
                ## number of: states in the cluster, considered states
                block_states, cons_states = self.N[ny][nx], prob.size
                W = self._peps_tensor(ny, nx) # PEPS tensor
                newprob, minprob = np.zeros((cons_states,  block_states)), np.zeros(cons_states)
                for kk in range(cons_states):
                    tind = tuple(vind[kk])
                    AA = W[:, tind[nx], :, :, tind[nx+1]]
                    newprob[kk], minprob[kk] = self._calculate_Pn(AA, RLl[tind[:nx]], self.rhoT[ny+1].A[nx], RRl[self.Nx-nx-1][tind[nx+2:]] )

                newprob = np.log2(newprob)

                newprob += prob[:, np.newaxis]  # use conditional probability to calculate probability of partial configuration
                newprob = np.reshape(newprob, cons_states*block_states)
                minprob, maxprob = np.min(minprob), np.max(newprob)
                cutoff = maxprob + np.log2(relative_P_cutoff)

                order   = np.arange(newprob.size)

                keep = max((newprob[order] > cutoff).sum(), 1)
                if keep < order.size:
                    or2     =  newprob[order].argpartition(-keep-1)
                    pd_max  =  max(pd_max, newprob[order[or2[-keep-1]]])
                    order   =  order[or2[-keep:]]
                    newprob =  newprob[order]   ## keep largest probabilities

                keep_more = 4
                if newprob.size > keep_more*M:  ## looks for max_state largest probabilities
                    or2     =  newprob.argpartition(-keep_more*M-1)
                    pd_max  =  max(pd_max, newprob[or2[-keep_more*M-1]])
                    order   =  order[or2[-keep_more*M::]]
                    newprob = newprob[or2[-keep_more*M::]]

                prob = newprob ## keep largest probabilities
                ## inds = which previous states
                ## indc = and state at the considered cluster (site)
                inds, indc = order // block_states, np.mod(order, block_states)
                states = states[inds]
                states[:, ny*self.Nx+nx] = indc
                vind = vind[inds]
                # update corresponding virtual indices
                vind[:, nx] = self._ind_bond_down(indc, ny, nx)
                vind[:, nx+1] = self._ind_bond_right(indc, ny, nx)
                Eng = Eng[inds]
                Eng += self._update_Eng(states, ny, nx)
                deg = deg[inds]

                # search for configurations where vind is the same:
                seen = {}  # identify similar configurations
                for kk in range(order.size):
                    tind = tuple(vind[kk])
                    if tind in seen:
                        seen[tind].append([kk, Eng[kk], prob[kk]])
                    else:
                        seen[tind] = [(kk, Eng[kk], prob[kk])]

                lseen = len(seen)
                uni = np.zeros(lseen, dtype = int)   ## index of "unique" (main) configurations
                probn = np.zeros(lseen)                ## their probabilities
                degn = np.zeros(lseen, dtype = int)   ## their degeneracy
                el = []  # new list collecting all excitations (excitations list) of all branches

                # ind  = indexing new main branches
                for ind, confs in enumerate(seen.values()):  # merge configuration
                    mEng = np.inf  # minimal Energy
                    for ii, En, pr in confs:  # for given vind find min energy
                        if En+min_dEng < mEng:
                            ui = ii
                            mEng = En
                            cprob = [pr]
                            cdeg = deg[ui]  # degeneracy of smallest energy
                        elif np.abs(En - mEng) < min_dEng:
                            cprob.append(pr)
                            cdeg += deg[ii]
                    probn[ind] = np.median(cprob)
                    uni[ind] = ui
                    degn[ind] = cdeg
                    # new branch excitation list <- copy from old
                    bel  = self.el[inds[ui]][:] 
                    new_bel = []
                    # add other merged branches as new excitations (possible with subexcitations)
                    for ii, En, pr in confs:
                        conf_dEng = En - mEng
                        if (conf_dEng <= max_dEng) and (ii != ui):  # if has small exc energy and is not the main branch 
                            dstate = np.bitwise_xor(states[ui], states[ii])  # find where the states differ (define excitation)
                            dpos = dstate.nonzero()[0]
                            dstate = dstate[dpos]
                            
                            nsel = []  ## list of subexcitations of dstate overlaping with it
                            for sne in self.el[inds[ii]]:  # starts adding subexcitations
                                if (sne[0][0] + conf_dEng <= max_dEng) and (self._exc_overlap((dpos, dstate), sne[0][1])):
                                    nsel.append(sne)
                            sEng, sflip = self._exc_unpack_v2(nsel, max_dEng - conf_dEng, one_layer=True)
                            for nn in range(len(sEng)):
                                substate = (dpos, dstate)
                                for sdi in sflip[nn]:
                                    substate = self._exc_merge(substate, sdi)
                                if self._exc_hd(substate[1]) >= lim_hd and self._exc_elementary(substate): # and excs.overlap2(dstate, sne[0][1]):
                                    sdi = self._exc_add_to_d(*substate)
                                    new_bel.append( ((sEng[nn] + conf_dEng, sdi), ()) )
                    new_bel = sorted(new_bel, key=lambda x: x[0][0]) # sorted by energy
                    distinct_new_bel = []
                    for x in new_bel:
                        to_add = True
                        for y in distinct_new_bel:
                            hd = self._exc_hd_comp(x[0][1], y[0][1])
                            if hd < lim_hd:
                                to_add=False
                                break
                        if to_add:
                            distinct_new_bel.append(x)
                    bel.extend(distinct_new_bel)                    
                    el.append(bel)  #finish merging given branch

                prob = probn
                deg = degn
                vind = vind[uni]
                states = states[uni]
                Eng = Eng[uni]
                self.el = el

                RLnew = {}   ## update left environment
                for one_ind in vind:
                    tind = tuple(one_ind[:(nx+1)])
                    if tind not in RLnew:
                        tempR  = np.dot(RLl[tind[:-1]], self.rhoT[ny+1].A[nx][:, tind[-1], :] )
                        tempR *= (1/mps.nfactor(tempR))
                        RLnew[tind] = tempR
                RLl = RLnew
                # self.logger.info('Prob: min/ min kept/ max: %0.2e / %0.2e / %0.2e ', minprob, np.min(prob), maxprob)
                globalmin = min(globalmin, minprob)  ## collects information on smallest encountered probability (negative indicate error)
            self._exc_clear_d()
            self.logger.info('Elapsed: %.2f seconds', time.time() - keep_time)
            vind[:, 1:] = vind[:, :-1]  ## reset vind before going to next layer
            vind[:, 0]  = 0             ## by shifting the one corresponding to "central" bond to begining
        
        # at the end greadily removes similar droplets
        bel = sorted(self.el[0], key=lambda x: x[0][0]) # sorted by energy
        distinct_bel = []
        for x in bel:
            to_add = True
            for y in distinct_bel:
                hd = self._exc_hd_comp(x[0][1], y[0][1])
                if hd < lim_hd:
                    to_add=False
                    break
            if to_add:
                distinct_bel.append(x)
        self.el[0] = distinct_bel  #finish merging given branch
        self._exc_clear_d()

        self.logger.info('Elapsed total: %.2f seconds', time.time() - keep_total_time)
        self.energy = Eng
        self.degeneracy = deg[0]
        self.states = states[:, self.order]
        self.probability = prob
        self.discarded_probability = pd_max
        self.negative_probability = min(globalmin, 0)
        self.el = self.el[0]
        # rotates d and adjecency settings to the original ordering before rotation.
        for key, value in self.d.items():
            dpos, dstate = value
            dpos = self.order_i[dpos]
            order = dpos.argsort()
            dpos = dpos[order]
            dstate = dstate[order]
            self.d[key] = (dpos, dstate)
        self._reset_adj(J=self.J0, Nx=self.Nx_model, Ny=self.Ny_model, ind=self.ind0)
        return Eng

    def decode_low_energy_states(self, max_dEng=0., max_states=1024):
        """
        Decode excitation structure found into actuall low-energy states.

        It can be used after method `search_low_energy_spectrum`.
        
        Args:
            max_dEng (float): bound on excitation energy.
            max_states (int): bound on a number of generated states.
                States with smallest energies are selected.
        """
        
        Eng, flip = self._exc_unpack(max_dEng=max_dEng, max_states=max_states)
        gs = self.states[0]

        order = Eng.argsort()
        Eng = Eng[order]

        max_states = min(max_states, len(Eng))
        states = np.zeros((max_states, self.Nx*self.Ny), dtype = self.indtype)
        for ii in range(max_states):
            newst = gs.copy()
            for kk in flip[order[ii]]:
                dpos, dstate = self.d[kk]
                newst[dpos] = np.bitwise_xor(newst[dpos], dstate)
            states[ii] = newst

        self.energy = Eng + self.energy[0]
        self.states = states
        return Eng[0]

    def _import_J(self, J):
        """
        Import list of couplings into class.
        """
        if self.mode == 'Ising':
            ii, jj, vv = zip(*J)
            JJ = sparse.coo_matrix((vv, (ii, jj)), shape=(self.L, self.L))
            # makes matrix J upper triangular
            self.J = sparse.triu(JJ) + sparse.tril(JJ, -1).T
            # makes sure that J is float
            self.J = self.J.astype(dtype=float, copy=False)
            self.active = 0 # number of active spins
            self.ind0 = [[0]*self.Nx for _ in range(self.Ny)]
            self.J0 = self.J.copy()
            for ny in range(self.Ny_model):
                for nx in range(self.Nx_model):
                    ind = self.Nc * (self.Nx_model * ny + nx) + np.arange(self.Nc)
                    Jsum = np.sum(np.abs(self.J[ind,:].toarray()), axis = 1) + np.sum(np.abs(self.J[:,ind].toarray()), axis = 0)
                    self.ind0[ny][nx] = ind[np.nonzero(Jsum > 1e-12)]
                    self.active += len(self.ind0[ny][nx])

    def _divide_couplings(self):
        """
        Preselect couplings contributing to peps tensor.
        """
        self.lu = np.ones((self.Ny, self.Nx), dtype=int)
        self.lr = np.ones((self.Ny, self.Nx), dtype=int)
        self.ll = np.ones((self.Ny, self.Nx), dtype=int)
        self.ld = np.ones((self.Ny, self.Nx), dtype=int)

        self.N = np.zeros((self.Ny, self.Nx), dtype=int) #  number of states in local block

        if self.mode == 'Ising':
            # indices corresponding to active spins in the block
            self.ind = [[0]*self.Nx for _ in range(self.Ny)]
            self.sN = np.zeros((self.Ny, self.Nx), dtype=int) # number of spins in local block


            for ny in range(self.Ny):
                for nx in range(self.Nx):
                    ind  =  self.Nc * (self.Nx * ny + nx) + np.arange(self.Nc)
                    Jsum =  np.sum(np.abs(self.J[ind,:].toarray()), axis = 1) + np.sum(np.abs(self.J[:,ind].toarray()), axis = 0)
                    self.ind[ny][nx] = ind[np.nonzero(Jsum > 1e-12)]
                    self.N[ny][nx] = 2**len(self.ind[ny][nx])
                    self.sN[ny][nx] = len(self.ind[ny][nx])

            self.Jin = [ [0]*self.Nx for _ in range(self.Ny) ]
            self.Jl  = [ [np.zeros((self.sN[ny][nx], 0)) for nx in range(self.Nx) ] for ny in range(self.Ny) ]
            self.Ju  = [ [np.zeros((self.sN[ny][nx], 0)) for nx in range(self.Nx) ] for ny in range(self.Ny) ]
            self.id  = [ [np.zeros(0, dtype=int)]*self.Nx for _ in range(self.Ny) ]
            self.ir  = [ [np.zeros(0, dtype=int)]*self.Nx for _ in range(self.Ny) ]
            self.sl = np.zeros((self.Ny, self.Nx), dtype=int)
            self.sd = np.zeros((self.Ny, self.Nx), dtype=int)
            self.sr = np.zeros((self.Ny, self.Nx), dtype=int)
            self.su = np.zeros((self.Ny, self.Nx), dtype=int)

            for ny in range(self.Ny):
                for nx in range(self.Nx):
                    ind = self.ind[ny][nx]
                    self.Jin[ny][nx] = self.J[ind, :][:, ind].toarray()
                    if nx > 0:
                        JJ  = self.J[self.ind[ny][nx-1]][:,ind].toarray()  ## connection to left cluster
                        iright  = np.nonzero(np.sum(np.abs(JJ), axis=1))[0]  ## looks for non-zero raws
                        self.Jl[ny][nx]   = JJ[iright].T
                        self.ir[ny][nx-1] = iright
                        self.sr[ny][nx-1] = len(iright)
                        self.sl[ny][nx]   = len(iright)
                        self.lr[ny][nx-1] = 2**len(iright)

                    if ny > 0:
                        JJ  = self.J[self.ind[ny-1][nx]][:,ind].toarray()  ## connection to upper cluster
                        idown  = np.nonzero(np.sum(np.abs(JJ), axis=1))[0]  ## looks for non-zero raws
                        self.Ju[ny][nx]   = JJ[idown].T
                        self.id[ny-1][nx] = idown
                        self.sd[ny-1][nx] = len(idown)
                        self.su[ny][nx]   = len(idown)
                        self.ld[ny-1][nx] = 2**len(idown)
        self._reset_X()

    # def _divide_couplings(self):
    #     """Selects couplings contributing to peps tensor"""
    #     # indices corresponding to active spins in the block
    #     self.active = 0 # number of active spins
    #     self.ll = np.ones((self.Ny, self.Nx), dtype=int)
    #     self.ld = np.ones((self.Ny, self.Nx), dtype=int)
    #     self.lr = np.ones((self.Ny, self.Nx), dtype=int)
    #     self.lu = np.ones((self.Ny, self.Nx), dtype=int)
    #     for ny in range(self.Ny):
    #         for nx in range(self.Nx):
    #             if ((ny, nx-1, ny, nx) in self.fact) or ((ny, nx, ny, nx-1) in self.fact):
    #                 self.ll[ny, nx]   = self.N[ny][nx-1]
    #             if ((ny, nx, ny, nx+1) in self.fact) or ((ny, nx+1, ny, nx) in self.fact):
    #                 self.lr[ny, nx]   = self.N[ny][nx+1]
    #             if ((ny-1, nx, ny, nx) in self.fact) or ((ny, nx, ny-1, nx) in self.fact):
    #                 self.lu[ny, nx]   = self.N[ny-1][nx]
    #             if ((ny, nx, ny+1, nx) in self.fact) or ((ny+1, nx, ny, nx) in self.fact):
    #                 self.ld[ny, nx]   = self.N[ny+1][nx]             
    #     self._reset_X()

    #  Auxliary functions for local states numbering 

    def _cluster_configurations(self, N):
        """
        All spin configurations in a block.
        """
        st = np.array(list(itertools.product([1, 0], repeat=N)), dtype = np.int8) #all configurations in block
        st = st[:,::-1]                                     #first spin changing fastest
        return st

    def _ind_bond_down(self, st_number, ny, nx):
        """
        Virtual indices selected by considered spin configurations.
        """
        if self.mode == 'Ising':
            st = 1-self._cluster_configurations(self.sN[ny][nx])
            todec = 2** np.arange(self.sd[ny][nx])
            return np.dot(st[st_number, :][:, self.id[ny][nx]], todec)
        elif self.mode == 'RMF':
            return np.mod(st_number, self.ld[ny, nx])  # update corresponding virtual indices

    def _ind_bond_right(self, st_number, ny, nx):
        """
        Virtual indices selected by considered spin configurations.
        """
        if self.mode == 'Ising':
            st = 1-self._cluster_configurations(self.sN[ny][nx])
            todec = 2** np.arange(self.sr[ny][nx])
            return np.dot(st[st_number, :][:, self.ir[ny][nx]], todec)
        elif self.mode == 'RMF':
            return np.mod(st_number, self.lr[ny, nx])  # update corresponding virtual indices

    def _ind_split(self, N, select):
        """
        Return indices corresponding to delta for some of the spins.
        """
        ind_delta = np.zeros(1, dtype = int)
        for ii in range(len(select)):
           ind_delta = np.hstack(( ind_delta, ind_delta+2**select[ii] ))

        ind_rest = np.zeros(1, dtype = int)
        for ii in range(N):
            if ii not in select:
                ind_rest = np.hstack(( ind_rest, ind_rest+2**ii ))

        return ind_delta, ind_rest

    def _update_Eng(self, states, ny, nx):
        st  = 2*self._cluster_configurations(self.sN[ny][nx])-1
        Es  = 1.*np.sum(np.dot(st, np.triu(self.Jin[ny][nx], 1)) * st, 1) + np.dot(st, self.Jin[ny][nx].diagonal())  ## inner cluster energy

        pos = ny*self.Nx+nx        # position of cluster
        dEng = Es[states[:, pos]]  # add energy of cluster for every configuration

        if nx > 0:  ## if there is a cluster to the left
            posl  = ny*self.Nx+(nx-1)
            ext  = 2*self._cluster_configurations( self.sl[ny][nx] ).T -1    #indices corresponding to left leg
            Esl   = np.dot(np.dot(st, self.Jl[ny][nx]), ext)
            indr  = self._ind_bond_right(states[:, posl], ny, nx-1)
            dEng  += Esl[states[:, pos], indr]

        if ny > 0:  ## if there is a cluster to the left
            posu  = (ny-1)*self.Nx+nx
            ext  = 2*self._cluster_configurations( self.su[ny][nx] ).T -1    #indices corresponding to up leg
            Esl   = np.dot(np.dot(st, self.Ju[ny][nx]), ext)
            indu  = self._ind_bond_down(states[:, posu], ny-1, nx)
            dEng  += Esl[states[:, pos], indu]

        return dEng

    # Auxliary functions for PEPS contractions

    def _peps_tensor(self, ny, nx):
        """
        Create peps tensors for exp(- beta H).
        """
        if self.mode == 'Ising':
            N = self.sN[ny][nx]  # number of spins in block
            L1, L2, L3, L4 = self.sl[ny][nx], self.sd[ny][nx], self.sr[ny][nx], self.su[ny][nx] # number of outer legs
            #cluster self energy
            st = 2*self._cluster_configurations(N)-1
            Es = np.sum(np.dot(st, np.triu(self.Jin[ny][nx], 1)) * st, 1) + np.dot(st, self.Jin[ny][nx].diagonal())
            minEs = np.min(Es)
            Es = self.beta*(minEs-Es)    #subtract Esmax for conditioning

            ext1    = 2*self._cluster_configurations(L1).T -1    #indices corresponding to leg 1 (left)
            Ese1    = np.dot(np.dot(st, self.Jl[ny][nx]), ext1)
            minEse1 = np.min(Ese1) #0
            Ese1    = self.beta*(minEse1-Ese1)

            ext4    = 2* self._cluster_configurations(L4).T -1    #indices corresponding to leg 4 (up)
            Ese4    = np.dot(np.dot(st, self.Ju[ny][nx]), ext4)
            minEse4 = np.min(Ese4) #0
            Ese4    = self.beta*(minEse4-Ese4)

            #collect exp(-beta (Es + Eleft + Eup))
            Es_full  = np.reshape(np.tile(Es[:, np.newaxis], (1, 2**(L1+L4))), (2**N, 2**L1, 2**L4))
            Es_full += np.reshape(np.tile(np.reshape(Ese1, (2**(N+L1), 1)), (1, 2**L4)), (2**N, 2**L1, 2**L4))
            Es_full += np.reshape(np.tile(Ese4, (1, 2**L1) ), (2**N, 2**L1, 2**L4) )
            Es_full  = np.exp(Es_full)

            for ii in range(2**L4):
                Es_full[:, :, ii] *= self.Xu[ny][nx][ii]  #conditioning

            for ii in range(2**L1):
                Es_full[:, ii, :] *= self.Xl[ny][nx][ii]  #conditioning

            #add delta L3 (right) + conditioning
            ind_delta, ind_rest = self._ind_split(N, self.ir[ny][nx])
            A = np.zeros((2**N, 2**L1, 2**L3, 2**L4))
            for ii in range(2**L3):
                A[ind_rest+ind_delta[ii], :, ii, :] = Es_full[ind_rest+ind_delta[ii], :, :]*self.Xr[ny][nx][ii]

            #add delta L2 (down) + conditioning
            ind_delta, ind_rest = self._ind_split(N, self.id[ny][nx])
            Es_full = np.zeros((2**N, 2**L1, 2**L2, 2**L3, 2**L4))
            for ii in range(2**L2):
                Es_full[ind_rest+ind_delta[ii], :, ii, :, :] = A[ind_rest+ind_delta[ii], :, :, :]*self.Xd[ny][nx][ii]

        return Es_full

    def _setup_rhoT(self, Dmax=64, tolS=1e-16, tolV=1e-10, max_sweeps=20):
        """
        Creates environment for layers of peps; from top.
        """
        self.rhoT = [1]*(self.Ny+1)
        self.rhoT_overlap   = [1]*(self.Ny+1)
        self.rhoT_discarded = [0]*(self.Ny+1)

        self.rhoT[-1] = mps.MPS(d=1, L=self.Nx, Dmax=1, initial='X')
        for ny in range(self.Ny-1, -1, -1):
            At = mps.MPO(L=self.Nx)
            for nx in range(self.Nx):  ## trace physical to form mpo
                W = np.sum(self._peps_tensor(ny, nx), axis = 0)
                At.set_direct(W, nx)
            self.rhoT[ny] = self.rhoT[ny+1].copy()
            self.rhoT[ny].apply_mpo(At, Hconj = True)
            self.rhoT_overlap[ny] = self.rhoT[ny].compress_mps(Dmax = Dmax, tolS = tolS, tolV = tolV, max_sweeps = max_sweeps, verbose = False)
            self.rhoT_discarded[ny] = max(self.rhoT[ny].discarded)
            #self.rhoT[ny].normC = 1.0

    def _setup_rhoB(self, Dmax = 64, tolS = 2.**(-52), tolV = 1e-12, max_sweeps = 20):
        """
        Creates environment for layers of peps; from bottom.
        """
        self.rhoB = [1]*(self.Ny+1)
        self.rhoB_overlap   = [1]*(self.Ny+1)
        self.rhoB_discarded = [0]*(self.Ny+1)

        self.rhoB[0] = mps.MPS(d=1, L=self.Nx, Dmax=1, initial='X')
        for ny in range(self.Ny):
            At = mps.MPO(L=self.Nx)
            for nx in range(self.Nx):  ## trace physical to form mpo
                W = np.sum(self._peps_tensor(ny, nx), axis = 0)
                At.set_direct(W, nx)
            self.rhoB[ny+1] = self.rhoB[ny].copy()
            self.rhoB[ny+1].apply_mpo(At, Hconj = False)
            self.rhoB_overlap[ny+1] = self.rhoB[ny+1].compress_mps(Dmax = Dmax, tolS = tolS, tolV = tolV, max_sweeps = max_sweeps, verbose = False)
            self.rhoB_discarded[ny+1] = max(self.rhoB[ny+1].discarded)
            #self.rhoB[ny+1].normC = 1.0

    def _setup_rhoL(self, Dmax = 64, tolS = 2.**(-52), tolV = 1e-12, max_sweeps = 20):
        """
        Creates environment for layers of peps; from left.
        """
        self.rhoL = [1]*(self.Nx+1)
        self.rhoL_overlap   = [1]*(self.Nx+1)
        self.rhoL_discarded = [0]*(self.Nx+1)

        self.rhoL[0] = mps.MPS(d=1, L=self.Ny, Dmax=1, initial='X')
        for nx in range(self.Nx):
            At = mps.MPO(L=self.Ny)
            for ny in range(self.Ny):  ## trace physical to form mpo
                W = np.sum(self._peps_tensor(ny, nx), axis = 0)
                W = np.transpose(W, (3, 0, 1, 2))
                At.set_direct(W, ny)
            self.rhoL[nx+1] = self.rhoL[nx].copy()
            self.rhoL[nx+1].apply_mpo(At, Hconj = True)
            self.rhoL_overlap[nx+1] = self.rhoL[nx+1].compress_mps(Dmax = Dmax, tolS = tolS, tolV = tolV, max_sweeps = max_sweeps, verbose = False)
            self.rhoL_discarded[nx+1] = max(self.rhoL[nx+1].discarded)
            #self.rhoL[nx+1].normC = 1.0

    def _setup_rhoR(self, Dmax = 64, tolS = 2.**(-52), tolV = 1e-12, max_sweeps = 20):
        """
        Creates environment for layers of peps; from right.
        """
        self.rhoR = [1]*(self.Nx+1)
        self.rhoR_overlap   = [1]*(self.Nx+1)
        self.rhoR_discarded = [0]*(self.Nx+1)

        self.rhoR[-1] = mps.MPS(d=1, L=self.Ny, Dmax=1, initial='X')
        for nx in range(self.Nx-1, -1, -1):
            At = mps.MPO(L=self.Ny)
            for ny in range(self.Ny):  ## trace physical to form mpo
                W = np.sum(self._peps_tensor(ny, nx), axis = 0)
                W = np.transpose(W, (3, 0, 1, 2))
                At.set_direct(W, ny)
            self.rhoR[nx] = self.rhoR[nx+1].copy()
            self.rhoR[nx].apply_mpo(At, Hconj = False)
            self.rhoR_overlap[nx] = self.rhoR[nx].compress_mps(Dmax = Dmax, tolS = tolS, tolV = tolV, max_sweeps = max_sweeps, verbose = False)
            self.rhoR_discarded[nx] = max(self.rhoR[nx].discarded)
            #self.rhoR[nx].normC = 1.0

    def _setup_RR(self, vind, ny):
        """
        Creates environment to contract layer from right.
        """
        RRl = [{ () : np.ones((1, 1)) }]
        for nx in range(self.Nx-1, 0, -1):
            W = np.sum(self._peps_tensor(ny, nx), axis = 0)
            RRnew = {}
            for ind in vind:
                tind = tuple(ind[nx+1:])
                if tind not in RRnew:
                    T = np.tensordot(self.rhoT[ny+1].A[nx], RRl[-1][tind[1:]], axes=(2, 0))
                    tempR = np.tensordot(T, W[:, :, :, tind[0]], axes=([1, 2], [1, 2]))
                    tempR *= (1/mps.nfactor(tempR))
                    RRnew[tind] = tempR
            RRl.append(RRnew)
        return RRl

    def _calculate_Pn(self, A, RL, AT, RR):
        T1 = np.tensordot(RL, AT, axes=(0, 0))
        T2 = np.tensordot(T1, RR, axes= (1, 0))
        Pn = np.tensordot(A, T2, axes = ((1, 2), (0, 1)))
        mPn = Pn.min()  # error handling: negative probabilities 
        if mPn < 0.:
            ind = (Pn<np.abs(mPn))
            Pn[ind] = np.abs(mPn)
            mPn *= np.sum(ind)
        no = np.sum(Pn)
        if no > 0.:
            Pn  *= 1./no
            mPn *= 1./no
        else:   # if all zeros
            Pn += 1./len(Pn)
            mPn = -1
        return Pn, mPn

    # functons for setting conditioning

    def _reset_X(self):
        """
        Initialised diagonal matrices to condition peps.
        """
        self.Xu = np.ones((self.Ny, self.Nx, np.max(self.ld)))
        self.Xd = np.ones((self.Ny, self.Nx, np.max(self.ld)))
        self.Xl = np.ones((self.Ny, self.Nx, np.max(self.lr)))
        self.Xr = np.ones((self.Ny, self.Nx, np.max(self.lr)))
        self.overlaps_ud = np.empty(shape=[0, self.Ny-1])
        self.overlaps_lr = np.empty(shape=[0, self.Nx-1])
        # here they are associated with legs of peps tensor A[ny][nx]
        # corresponding Xu, Xd; and Xl, Xr should combine to identity

    def _update_conditioning(self, direction='ud', Dmax=8, tolS=1e-16, tolV=1e-10, max_sweeps=4, max_scale=1024):
        """
        A sweep searching for conditioning using balancing heuristics.
        """
        max_scale = mps.nfactor(np.sqrt(max_scale))
        if direction is 'ud':
            self._setup_rhoT(Dmax, tolS, tolV, max_sweeps)  # is left canonical
            self._setup_rhoB(Dmax, tolS, tolV, max_sweeps)  # is left canonical
            overlaps = np.ones((2, self.Ny-1))
            for ny in range(1, self.Ny):
                for nx in range(self.Nx):
                    self.rhoB[ny].update_RL_mix(self.rhoT[ny], nx)
                    self.rhoB[ny].R[nx+1] *= (1/np.linalg.norm(self.rhoB[ny].R[nx+1]))

                for nx in range(self.Nx-1, -1, -1):
                    env = self.rhoB[ny].bond_env_mix(self.rhoT[ny], nx)
                    _ , scale = splinalg.matrix_balance(env, permute = False, separate = True)
                    scale = np.minimum(np.maximum(scale[0], 1/max_scale), max_scale)
                    
                    o1 = self.rhoB[ny].expectation_mix(self.rhoT[ny], nx)
                    o1B = np.linalg.norm(self.rhoB[ny].A[nx])
                    o1T = np.linalg.norm(self.rhoT[ny].A[nx])
                    o1 *= 1/(o1B*o1T)

                    self.rhoB[ny].apply_diagonalO(scale, nx)
                    self.rhoT[ny].apply_diagonalO(1/scale, nx)

                    o2B = np.linalg.norm(self.rhoB[ny].A[nx])
                    o2T = np.linalg.norm(self.rhoT[ny].A[nx])
                    o2 = self.rhoB[ny].expectation_mix(self.rhoT[ny], nx)
                    o2 *= 1/(o2B*o2T)

                    if o1 < overlaps[0, ny-1]:
                        overlaps[0, ny-1] = o1
                        overlaps[1, ny-1] = max(o1, o2)

                    if o2 > o1:
                        self.Xd[ny-1, nx, :self.ld[ny-1, nx]] *= scale
                        self.Xu[ny, nx, :self.ld[ny-1, nx]] *= 1/scale
                    else:
                        self.rhoB[ny].apply_diagonalO(1/scale, nx)
                        self.rhoT[ny].apply_diagonalO(scale, nx)

                    if nx > 0:
                        self.rhoB[ny].orth_right(nx)
                        self.rhoB[ny].attach_AC()
                        self.rhoT[ny].orth_right(nx)
                        self.rhoT[ny].attach_AC()
                        self.rhoB[ny].update_RR_mix(self.rhoT[ny], nx)
                        self.rhoB[ny].R[nx] *= (1/np.linalg.norm(self.rhoB[ny].R[nx]))

                for nx in range(self.Nx):
                    env = self.rhoB[ny].bond_env_mix(self.rhoT[ny], nx)
                    _ , scale = splinalg.matrix_balance(env, permute = False, separate = True)
                    scale = np.minimum(np.maximum(scale[0], 1/max_scale), max_scale)
                    
                    o1 = self.rhoB[ny].expectation_mix(self.rhoT[ny], nx)
                    o1B = np.linalg.norm(self.rhoB[ny].A[nx])
                    o1T = np.linalg.norm(self.rhoT[ny].A[nx])
                    o1 *= 1/(o1B*o1T)
                    
                    self.rhoB[ny].apply_diagonalO(scale, nx)
                    self.rhoT[ny].apply_diagonalO(1/scale, nx)
                    
                    o2B = np.linalg.norm(self.rhoB[ny].A[nx])
                    o2T = np.linalg.norm(self.rhoT[ny].A[nx])
                    o2 = self.rhoB[ny].expectation_mix(self.rhoT[ny], nx)
                    o2 *= 1/(o2B*o2T)

                    if o1 < overlaps[0, ny-1]:
                        overlaps[0, ny-1] = o1
                        overlaps[1, ny-1] = max(o1, o2)

                    if o2 > o1:
                        self.Xd[ny-1, nx, :self.ld[ny-1, nx]] *= scale
                        self.Xu[ny, nx, :self.ld[ny-1, nx]] *= 1/scale
                    else:
                        self.rhoB[ny].apply_diagonalO(1/scale, nx)
                        self.rhoT[ny].apply_diagonalO(scale, nx)

                    if nx < self.Nx-1:
                        self.rhoB[ny].orth_left(nx)
                        self.rhoB[ny].attach_CA()
                        self.rhoT[ny].orth_left(nx)
                        self.rhoT[ny].attach_CA()
                        self.rhoB[ny].update_RL_mix(self.rhoT[ny], nx)
                        self.rhoB[ny].R[nx+1] *= (1/np.linalg.norm(self.rhoB[ny].R[nx+1]))

            self.overlaps_ud = np.vstack([self.overlaps_ud, overlaps])
            self.rhoB = []

        elif direction is 'lr':
            self._setup_rhoL (Dmax, tolS, tolV, max_sweeps)  # is left canonical
            self._setup_rhoR (Dmax, tolS, tolV, max_sweeps)  # is left canonical
            overlaps = np.ones((2, self.Nx-1))
            for nx in range(1, self.Nx):
                for ny in range(self.Ny):
                    self.rhoL[nx].update_RL_mix(self.rhoR[nx], ny)
                    self.rhoL[nx].R[ny+1] *= (1/np.linalg.norm(self.rhoR[nx].R[ny+1]))

                for ny in range(self.Ny-1, -1, -1):
                    env = self.rhoL[nx].bond_env_mix(self.rhoR[nx], ny)
                    _ , scale = splinalg.matrix_balance(env, permute = False, separate = True)
                    scale = np.minimum(np.maximum(scale[0], 1/max_scale), max_scale)
                    
                    o1 = self.rhoL[nx].expectation_mix(self.rhoR[nx], ny)
                    o1L = np.linalg.norm(self.rhoL[nx].A[ny])
                    o1R = np.linalg.norm(self.rhoR[nx].A[ny])
                    o1 *= 1/(o1L*o1R)

                    self.rhoL[nx].apply_diagonalO(scale, ny)
                    self.rhoR[nx].apply_diagonalO(1/scale, ny)
                    o2L = np.linalg.norm(self.rhoL[nx].A[ny])
                    o2R = np.linalg.norm(self.rhoR[nx].A[ny])
                    o2 = self.rhoL[nx].expectation_mix(self.rhoR[nx], ny)
                    o2 *= 1/(o2L*o2R)

                    if o1 < overlaps[0, nx-1]:
                        overlaps[0, nx-1] = o1
                        overlaps[1, nx-1] = max(o1, o2)

                    if o2 > o1:
                        self.Xr[ny, nx-1, :self.lr[ny, nx-1]] *= scale
                        self.Xl[ny, nx, :self.lr[ny, nx-1]] *= 1/scale
                    else:
                        self.rhoL[nx].apply_diagonalO(1/scale, ny)
                        self.rhoR[nx].apply_diagonalO(scale, ny)
                    
                    if ny > 0:
                        self.rhoL[nx].orth_right(ny)
                        self.rhoL[nx].attach_AC()
                        self.rhoR[nx].orth_right(ny)
                        self.rhoR[nx].attach_AC()
                        self.rhoL[nx].update_RR_mix(self.rhoR[nx], ny)
                        self.rhoL[nx].R[ny] *= (1/np.linalg.norm(self.rhoR[nx].R[ny]))

                for ny in range(self.Ny):
                    env = self.rhoL[nx].bond_env_mix(self.rhoR[nx], ny)
                    _ , scale = splinalg.matrix_balance(env, permute = False, separate = True)
                    scale = np.minimum(np.maximum(scale[0], 1/max_scale), max_scale)
                    
                    o1 = self.rhoL[nx].expectation_mix(self.rhoR[nx], ny)
                    o1L = np.linalg.norm(self.rhoL[nx].A[ny])
                    o1R = np.linalg.norm(self.rhoR[nx].A[ny])
                    o1 *= 1/(o1L*o1R)
                    
                    self.rhoL[nx].apply_diagonalO(scale, ny)
                    self.rhoR[nx].apply_diagonalO(1/scale, ny)
                    o2L = np.linalg.norm(self.rhoL[nx].A[ny])
                    o2R = np.linalg.norm(self.rhoR[nx].A[ny])
                    o2 = self.rhoL[nx].expectation_mix(self.rhoR[nx], ny)
                    o2 *= 1/(o2L*o2R)

                    if o1 < overlaps[0, nx-1]:
                        overlaps[0, nx-1] = o1
                        overlaps[1, nx-1] = max(o1, o2)

                    if o2 > o1:
                        self.Xr[ny, nx-1, :self.lr[ny, nx-1]] *= scale
                        self.Xl[ny, nx, :self.lr[ny, nx-1]] *= 1/scale
                    else:
                        self.rhoL[nx].apply_diagonalO(1/scale, ny)
                        self.rhoR[nx].apply_diagonalO(scale, ny)

                    if ny < self.Ny-1:
                        self.rhoL[nx].orth_left(ny)
                        self.rhoL[nx].attach_CA()
                        self.rhoR[nx].orth_left(ny)
                        self.rhoR[nx].attach_CA()
                        self.rhoL[nx].update_RL_mix(self.rhoR[nx], ny)
                        self.rhoL[nx].R[ny+1] *= (1/np.linalg.norm(self.rhoR[nx].R[ny+1]))

            self.overlaps_lr = np.vstack([self.overlaps_lr, overlaps])
            self.rhoL, self.rhoR = [], []
    
    ##############
    # functions handling excitations
    ##############

    def _exc_initialise(self):
        """
        Assume nearest neighbour interaction on Nx x Ny lattice.
        """
        self.d = {}  # dict keeping excitations
        self.invd = {}  # dict giving partial inverse for better searching
        self.el = [[]] # list of excitations for all branches
        self.free_d = 0  # first free index in dict
        
    def _reset_adj(self, J, Nx, Ny, ind):   
        if self.mode == 'Ising':
            # adjecency matrix
            self.adj = (sparse.triu(J, 1) != 0) + (sparse.triu(J, 1).T != 0) 
            self.adj = self.adj.toarray()  # faster to keep it as a full matrix
            self.xor2ind = []
            for ny in range(Ny):
                for nx in range(Nx):
                    Nc = len(ind[ny][nx])
                    decode = self._cluster_configurations(Nc)
                    decode = 1-decode #[:,::-1]   ## has to be consistent with cluster_configurations -- but replace 0 <--> 1
                    decode = decode.astype(dtype = bool, copy = False)
                    xor2ind_cluster = []
                    for ii in range(2**Nc):
                        xor2ind_cluster.append(ind[ny][nx][decode[ii]])
                    self.xor2ind.append(xor2ind_cluster)
        elif self.mode == 'RMF':
            # adjecency matrix
            self.adj_Nx = Nx
            self.adj_Ny = Ny
            
    def exc_show_properties(self):
        """
        Displays some info on the tree storing excitation structure.
        """
        print("Excitation encoding  :", self.excitations_encoding)
        print("Size of dictionary   :", len(self.d))
        print("Exc in first layer   :", len(self.el))

    def _exc_add_to_d(self, dpos, dstate):
        """
        If shape is in the dictionary, return its key; else add it to the dictionary and returns new key.
        """
        droplet = (dpos, dstate)
        sh = self._exc_get_sh(droplet) # semi-hash
        newkey = self.free_d  # use if neccesary
        if sh in self.invd:
            suspects = self.invd[sh]
            for ss in suspects:
                if np.array_equal(droplet[0], self.d[ss][0]) and np.array_equal(droplet[1], self.d[ss][1]):                
                    return ss  # exhitation already exists in d
            # it does not exists in d
            self.invd[sh].append(newkey)  # adds index to invd
        else:
            self.invd[sh] = [newkey]
        self.d[newkey] = droplet
        self.free_d += 1
        return newkey

    def _exc_cut_energy(self, exc, maxdE):
        """
        Recursively removes sub-excitations with too large energy.
        """
        nse = [] # new subexcitations list
        for se in exc[1]:
            if (se[0][0] <= maxdE):
                nse.append(self._exc_cut_energy(se, maxdE - se[0][0]))
        return (exc[0], tuple(nse))

    def _exc_xor2ind(self, exc):
        """
        Translates excitation (format as kept in dictionary) to mask for adjacency matrix
        """
        return np.hstack([ self.xor2ind[nn][ds] for nn, ds in zip(*exc)])

    def _exc_elementary(self, exc):
        """
        Check if excitation is elementary (single-connected).
        """
        #exc = (dpos, dstate)
        if self.mode == 'Ising':
            exci = self._exc_xor2ind(exc)
            gr, rest = exci[:1], exci[1:]  # starting point
            while (gr.size > 0) and (rest.size > 0):
                ind  = np.any(self.adj[gr, :][:, rest], axis=0)
                #ind  = np.any(self.adj[gr, :][:, rest].toarray(), axis=0)  is adj is sparse
                gr   = rest[ind]
                rest = rest[~ind]
            return (rest.size == 0)   ## if rest.size == 0 then it is connected, else if is not connected    
        elif self.mode == 'RMF':
            gr, rest = exc[0][:1], exc[0][1:]  # starting point
            while (gr.size > 0) and (rest.size > 0):
                gr_nx = np.mod(gr, self.adj_Nx)
                gr_ny = gr // self.adj_Nx
                rt_nx = np.mod(rest, self.adj_Nx)
                rt_ny = rest // self.adj_Nx
                lg, lr = gr.size, rest.size
                distance = np.abs(gr_nx.reshape(lg, 1) - rt_nx.reshape(1, lr)) + \
                        np.abs(gr_ny.reshape(lg, 1) - rt_ny.reshape(1, lr))
                ind  = np.any(distance==1, axis=0)
                gr   = rest[ind]
                rest = rest[~ind]
            return (rest.size == 0) # if rest.size == 0 then it is connected, else if is not connected

    def _exc_overlap(self, ie1, ie2):
        """
        Check if two shapes in the dictionary are connected.

        For mode Ising according to adjacency matrix.
        For mode RFM on 2d  nearest-neighbours Nx x Ny lattice.
        """

        exc1 = self.d[ie1] if isinstance(ie1, int) else ie1
        exc2 = self.d[ie2] if isinstance(ie2, int) else ie2

        if self.mode == 'Ising':
            # connected
            exc1 = self._exc_xor2ind(exc1)
            exc2 = self._exc_xor2ind(exc2)
            return np.any(self.adj[exc1, :][:, exc2])
        elif self.mode == 'RMF':    
            ## overlap if nearest-neighbours
            nx1 = np.mod(exc1[0], self.adj_Nx)
            ny1 = exc1[0] // self.adj_Nx
            nx2 = np.mod(exc2[0], self.adj_Nx)
            ny2 = exc2[0] // self.adj_Nx
            l1, l2 = nx1.size, nx2.size
            distance = np.abs(nx1.reshape(l1, 1) - nx2.reshape(1, l2)) + \
                    np.abs(ny1.reshape(l1, 1) - ny2.reshape(1, l2))
            return np.any(distance <= 1)

    def _exc_hd(self, dstate):
        """
        Hamming distance of a droplet.
        """
        if self.mode == 'Ising':
            return len(dstate)
        elif self.mode == 'RMF':
            return sum( bin(st).count('1') for st in dstate)
    
    def _exc_hd_comp(self, ie1, ie2):
        """
        Hamming distance of the overlap of two droplets.
        """

        exc1 = self.d[ie1] if isinstance(ie1, int) else ie1
        exc2 = self.d[ie2] if isinstance(ie2, int) else ie2

        l1, l2 = len(exc1[0]), len(exc2[0])
        n1, n2, hd = 0, 0, 0
        
        if self.mode == 'Ising':    
            while (n1 < l1) and (n2 < l2):
                if (exc1[0][n1] == exc2[0][n2]):
                    hd += bin(np.bitwise_xor(exc1[1][n1], exc2[1][n2])).count('1')
                    n1 += 1
                    n2 += 1
                elif exc1[0][n1] < exc2[0][n2]:
                    hd += bin(exc1[1][n1]).count('1')
                    n1 += 1
                else:  #exc1[0][n1] > exc2[0][n2]
                    hd += bin(exc2[1][n2]).count('1')
                    n2 += 1
            for ii in range(n1, l1):
                hd += bin(exc1[1][ii]).count('1')
            for ii in range(n2, l2):
                hd += bin(exc2[1][ii]).count('1')
        elif self.mode == 'RMF':
            while (n1 < l1) and (n2 < l2):
                if (exc1[0][n1] == exc2[0][n2]):
                    if (exc1[1][n1] != exc2[1][n2]):
                        hd += 1
                    n1 += 1
                    n2 += 1
                elif exc1[0][n1] < exc2[0][n2]:
                    n1 += 1
                    hd += 1
                else:  #exc1[0][n1] > exc2[0][n2]
                    n2 += 1
                    hd += 1
            if (n1 < l1):
                hd += l1-n1
            elif (n2 < l2):
                hd += l2-n2
        return hd

    def _exc_merge(self, ie1, ie2):
        if isinstance(ie1, int):
            dpos1, dstate1 = self.d[ie1]
        else:
            dpos1, dstate1 = ie1
        if isinstance(ie2, int):
            dpos2, dstate2 = self.d[ie2]
        else:
            dpos2, dstate2 = ie2
        
        l1 = len(dpos1)
        l2 = len(dpos2)
        dpos = np.zeros(l1+l2, dtype=np.int)
        dstate = np.zeros(l1+l2, dtype=np.int)
        n1 = 0
        n2 = 0
        n = 0
        while (n1 < l1) and (n2 < l2):
            if (dpos1[n1] == dpos2[n2]):
                if (dstate1[n1] != dstate2[n2]):
                    dpos[n] = dpos1[n1]
                    dstate[n] = np.bitwise_xor(dstate1[n1], dstate2[n2])
                    n += 1
                n1 += 1
                n2 += 1
            elif dpos1[n1] < dpos2[n2]:
                dpos[n] = dpos1[n1]
                dstate[n] = dstate1[n1]
                n1 += 1
                n += 1
            else:  #dpos1[n1] > dpos2[n2]
                dpos[n] = dpos2[n2]
                dstate[n] = dstate2[n2]
                n2 += 1
                n += 1
        
        if (n1 < l1):
            dpos[n:n+l1-n1] = dpos1[n1:]
            dstate[n:n+l1-n1] = dstate1[n1:]
            n = n+l1-n1
        elif (n2 < l2):
            dpos[n:n+l2-n2] = dpos2[n2:]
            dstate[n:n+l2-n2] = dstate2[n2:]
            n = n+l2-n2
        dpos = dpos[:n]
        dstate = dstate[:n]
        return dpos, dstate


    def _exc_clear_d(self):
        """
        Clear dictionary keeping existing shapes of excitations.
        """
        uq = []
        for bel in self.el:
            uq.append(self._exc_get_unique_keys(bel))
        uq = list(set(itertools.chain(*uq)))

        nd = {}  # new dictionary with neccesary elements only
        ninvd = {}
        for k in uq:
            nd[k] = self.d[k]
            sh = self._exc_get_sh(self.d[k])
            if sh in ninvd:
                ninvd[sh].append(k)
            else:
                ninvd[sh] = [k]
        self.d = nd
        self.invd = ninvd

    def _exc_get_sh(self, exc):
        """
        Semi-hash to faster identify shapes in dictionary.
        """
        fnz = (exc[0][0], exc[1][0], exc[0][-1], exc[1][-1])
        return fnz

    def _exc_get_unique_keys(self, l):
        """
        From list/tuple of excitations recursively gets unique keys of excitation shapes.
        """
        uq = []  # unique keys
        for exc in l:
            uq.append([exc[0][1]])
            uq.append(self._exc_get_unique_keys(exc[1]))
        return list(set(itertools.chain(*uq)))

    def _exc_unpack(self, max_dEng = 0., max_states=np.inf):
        if self.excitations_encoding == 1:
            return self._exc_unpack_v1(self.el, max_dEng=max_dEng, max_states=max_states)
        elif self.excitations_encoding == 2:
            return self._exc_unpack_v2(self.el, max_dEng=max_dEng, max_states=max_states)
        elif self.excitations_encoding == 3:
            return self._exc_unpack_v2(self.el, max_dEng=max_dEng, max_states=max_states, one_layer=True)

    def _exc_unpack_v1(self, el, max_dEng=0., max_states=np.inf):
        """
        Unpack excitations list l;=.
        Encoding of excitations as in search_low_energy_spectrum_v1.
        Return all possible excitation energy states.
        Up to max_dEng and max_states.
        """
        Eng   = [0.0]
        #Pn   = [1.0]
        flip  = [[]]
        #ne = ((conf_dEng, di, first, last, dP),  tuple(sel))
        excs = [[ ((0, 0, -1, self.Nx_model*self.Ny_model-1, 1), tuple(el)) ]]  ## excitations corresponding to the branch with Eng = [0.0] -- i.e. at the last site

        for nn in range(self.Nx_model*self.Ny_model-1, -1, -1):
            kk = 0
            while kk < len(Eng):
                for ee in excs[kk][-1][1]:
                    if (ee[0][3] == nn) and (Eng[kk] + ee[0][0]) <= max_dEng :  ## expand ee[0][3] = ee.last; ee[0][0] = dEng
                        Eng.append(Eng[kk] + ee[0][0])
                        #Pn.append(Pn[kk] * ee[0][4]) # ee[0][4] = dP
                        newflip = flip[kk][:]
                        newflip.append(ee[0][1])
                        flip.append( newflip )  ## ee[0][1] -> shape of exc from dict
                        excs.append( excs[kk][:] )
                        excs[-1].append(ee)
                    elif  ee[0][3] > nn:
                        break
                kk += 1

            if len(Eng) > max_states: # if to many states
                # throw away high-energy ones
                order = np.array(Eng).argpartition(max_states)[:max_states]
                Eng = [Eng[ii] for ii in order]
                flip = [flip[ii] for ii in order]
                excs = [excs[ii] for ii in order]

            for kk in range(len(Eng)):
                while excs[kk][-1][0][2] >= nn:
                    excs[kk].pop()
        return np.array(Eng), flip, #, np.array(Pn), 

    def _exc_unpack_v2(self, l, max_dEng=0., max_states=np.inf, one_layer=False):
        """
        Unpack excitations list l.
        Encoding of excitations as in search_low_energy_spectrum_v2.
        (and _v3 if one_layer=True)
        Return all possible excitation energy states.
        Up to max_dEng and max_states.
        """
        Eng   = [0.0]
        excs  = [l[:]]
        flip =  [[]]
        notend = True
        while notend:
            notend = False
            kk = 0
            while kk < len(Eng):
                if excs[kk]:
                    exc = excs[kk].pop()
                    if (Eng[kk] + exc[0][0]) <= max_dEng:
                        Eng.append( Eng[kk] + exc[0][0] )
                        newflip = flip[kk][:]
                        newflip.append(exc[0][1])
                        flip.append( newflip )  ## ee[0][1] -> shape of exc from dict
                        newexc = []
                        for xx in excs[kk]:
                            if not self._exc_overlap(xx[0][1], exc[0][1]): ## in l1 keep only interactions which are independent
                                newexc.append(xx)
                        excs.append(newexc)
                        if not one_layer:
                            newexc.extend(list(exc[1]))
                        if (not notend) or (newexc) or (excs[kk]):
                            notend = True
                kk += 1
            if len(Eng) > max_states: # if to many states
                # throw away high-energy ones
                order = np.array(Eng).argpartition(max_states)[:max_states]
                Eng = [Eng[ii] for ii in order]
                flip = [flip[ii] for ii in order]
                excs = [excs[ii] for ii in order]
        return np.array(Eng), flip

    def _exc_excitations_to_list(self, l):
        """
        """
        exc = []
        for ee in l:
            subexc = self._exc_excitations_to_list(ee[1])
            nee = [ee[0], subexc]
            exc.append(nee)
        return exc

    def exc_export_shapes(self):
        return self._exc_export_shapes(self.el)

    def _exc_export_shapes(self, el, ind=-1, d={}):
        if self.mode == 'RMF':
            for exc in el:
                ind += 1
                Eng = exc[0][0]
                kk = self.d[exc[0][1]][0]
                nx = np.mod(kk, self.adj_Nx)
                ny = kk // self.adj_Nx
                d[ind] = [Eng, list([x1, y1] for x1, y1 in zip(nx, ny))]
                if exc[1]:
                    d = self._exc_export_shapes(exc[1], ind, d)
            return d

    def exc_print(self):
        self._exc_print(el=self.el, layer=1)

    def _exc_print(self, el, layer=1):
        """
        Displays tree of the excitation structure.
        """
        for exc in el:
            Eng = exc[0][0]
            kk = self.d[exc[0][1]]
            print((3*layer-3)*' '+"|- %0.4f "%(Eng)+' : '+' '.join(map(str, kk[0]))+' | '+' '.join(map(str, kk[1])))
            # ' '.join(["%2d " % v for v in exc[0][1:]])
            self._exc_print(exc[1], layer+1)

    def exc_print_f(self, f):
        self._exc_print_f(f, el=self.el, layer=1)

    def _exc_print_f(self, f, el, layer= 1):
        """
        Print tree of the excitation structure to a file.
        """
        for exc in el:
            Eng = exc[0][0]
            kk = self.d[exc[0][1]]
            print("%1d "%(layer)+' '+"|- %0.4f "%(Eng)+' : '+' '.join(map(str, kk[0]))+' | '+' '.join(map(str, kk[1])), file=f)
            self._print_exc_f(f, exc[1], layer+1)
