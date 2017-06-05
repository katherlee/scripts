from __future__ import division, print_function
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import sys


def gen_annihilators(nflavors):
    '''Generate annihilation operators'''
    nhilb = 2**nflavors
    anni = [None] * nflavors
    mask = 1
    sign = np.ones(nhilb, dtype=int)
    for i in range(nflavors):
        nonzero = np.array(np.arange(nhilb) & mask, dtype=bool)
        anni[i] = sp.dia_matrix(
            (nonzero * sign, mask), shape=(nhilb, nhilb), dtype=int).todense()
        sign[nonzero] *= -1
        mask <<= 1

    return anni


class HubbardHamiltonian:
    def __init__(self, t_matrix, mu, U, beta=10, ntau=100, niw=100):
        self.build_ham(t_matrix, mu, U)
        ham = self.ham_total()
        e, V = la.eigh(self.ham_total())
        self.eigval = e - e.min()
        self.eigvec = np.asmatrix(V)

        self.alist_diag = list(map(self.transform_to_diag, self.alist))
        self.clist_diag = list(map(self.transform_to_diag, self.clist))
        self.nlist_diag = [c.dot(a) for c, a in zip(self.clist_diag, self.alist_diag)]

        self.beta = beta
        self.ntau = ntau
        self.niw = niw
        self.Z = np.exp(-beta * self.eigval).sum()

        self.tau_grid = np.arange(ntau + 1) * beta / ntau
        self.iw_grid = (np.arange(niw) * 2 + 1) * np.pi / beta
        self.gtau_vals = np.asarray(list(map(self.get_gtau, self.tau_grid)))
        self.giw_vals = np.asarray(list(map(self.get_giw, self.iw_grid)))

    def build_ham(self, t_matrix, mu, U):
        t_matrix = np.asmatrix(t_matrix)
        assert la.norm(t_matrix - t_matrix.H) < 1.0e-16
        nsites = t_matrix.shape[0]
        nflavors = nsites * 2
        alist = gen_annihilators(nflavors)
        clist = [a.H for a in alist]
        nlist = [c.dot(a) for c, a in zip(clist, alist)]

        ham_kin = 0
        for i in range(nsites):
            for j in range(nsites):
                ham_kin += t_matrix[i, j] * (
                    clist[i] * alist[j] + clist[i + nsites] * alist[j +
                                                                    nsites])

        ham_int = 0
        for i in range(nsites):
            ham_int += U * nlist[i] * nlist[i + nsites]

        ham_mu = -mu * sum(nlist)

        self.t = t_matrix
        self.U = U
        self.mu = mu
        self.nflavors = nflavors
        self.alist = alist
        self.clist = clist
        self.nlist = nlist
        self.ham_kin = ham_kin
        self.ham_int = ham_int
        self.ham_mu = ham_mu

    def ham_total(self):
        return self.ham_kin + self.ham_int + self.ham_mu

    def ham_canonical(self):
        return self.ham_kin + self.ham_int

    def transform_to_diag(self, matrix):
        matrix = np.asmatrix(matrix)
        return self.eigvec.H * matrix * self.eigvec

    # G_ij(t) = -<T c_i(t) c^+_j(0)>
    def get_gtau(self, tau):
        sign = 1
        if tau < 0:
            tau += beta
            sign = -1
        evolv1 = np.exp(-tau * self.eigval)
        evolv2 = np.exp(-(self.beta - tau) * self.eigval)
        return -sign * np.einsum('a, iab, b, jba -> ij', evolv2,
                                 np.asarray(self.alist_diag), evolv1,
                                 np.asarray(self.clist_diag)) / self.Z

    # G(iw) = \int_0^\beta G(t) exp(iwt) dt
    def get_giw(self, w):
        boltz = np.exp(-self.beta * self.eigval)
        pref = (boltz.reshape(1, -1) + boltz.reshape(-1, 1)) / (
            1j * w + self.eigval.reshape(-1, 1) - self.eigval.reshape(1, -1))
        return np.einsum('ab, iab, jba -> ij', pref,
                         np.asarray(self.alist_diag),
                         np.asarray(self.clist_diag)) / self.Z

    # \rho_ij = <c^+_i c_j> = G_ji(-0)
    def get_density_mat(self):
        return np.einsum('a, iab, jba -> ij',
                         np.exp(-self.beta * self.eigval),
                         np.asarray(self.clist_diag),
                         np.asarray(self.alist_diag)) / self.Z

    def get_density_corr(self):
        return np.einsum('a, iab, jba -> ij',
                         np.exp(-self.beta * self.eigval),
                         np.asarray(self.nlist_diag),
                         np.asarray(self.nlist_diag)) / self.Z

    def get_hifreq_moments(self):
        c1 = np.eye(self.nflavors)
        dens = np.diag(self.get_density_mat())
        c2 = la.block_diag(self.t, self.t) - self.mu * np.eye(self.nflavors) - self.U * np.diag(
            np.roll(dens, int(self.nflavors / 2)))
        c3 = c2.dot(c2) + self.U ** 2 * (self.get_density_corr() - np.outer(dens, dens))
        return [c1, c2, c3]
