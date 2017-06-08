from __future__ import division, print_function
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import h5py as h5
from h5archive import gf


def write_tau_gf(fname, path, data, beta, stat='fermi'):
    f = h5.File(fname)
    qtty = gf.new_quantity(f, path)
    ntau, n1, n2 = data.shape
    gf.new_mesh_group(qtty, 3)
    gf.new_tau_mesh(qtty, 1, ntau=ntau, beta=beta, stat=stat)
    gf.new_index_mesh(qtty, 2, n1)
    gf.new_index_mesh(qtty, 3, n2)
    gf.new_data(qtty, data)
    f.close()


def write_iw_gf(fname, path, data, beta, tails=None, stat='fermi'):
    f = h5.File(fname)
    qtty = gf.new_quantity(f, path)
    niw, n1, n2 = data.shape
    gf.new_mesh_group(qtty, 3)
    gf.new_matsubara_mesh(qtty, 1, niw=niw, beta=beta, full=False, stat=stat)
    gf.new_index_mesh(qtty, 2, n1)
    gf.new_index_mesh(qtty, 3, n2)
    gf.new_data(qtty, data)
    if tails:
        gf.new_inftail(qtty, tails)
    f.close()


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
    def __init__(self,
                 t_matrix,
                 mu,
                 U,
                 beta=10,
                 ntau=100,
                 niw=100,
                 shift=False):
        self.build_ham(t_matrix, mu, U, shift)
        ham = self.ham_total()
        e, V = la.eigh(self.ham_total())
        self.eigval = e - e.min()
        self.eigvec = np.asmatrix(V)

        self.alist_diag = list(map(self.transform_to_diag, self.alist))
        self.clist_diag = list(map(self.transform_to_diag, self.clist))
        self.nlist_diag = [
            c.dot(a) for c, a in zip(self.clist_diag, self.alist_diag)
        ]

        self.beta = beta
        self.ntau = ntau
        self.niw = niw
        self.Z = np.exp(-beta * self.eigval).sum()

        self.tau_grid = np.arange(ntau + 1) * beta / ntau
        self.iwf_grid = (np.arange(niw) * 2 + 1) * np.pi / beta
        self.iwb_grid = (np.arange(niw) * 2) * np.pi / beta
        self.gtau_vals = np.asarray(list(map(self.get_gtau, self.tau_grid)))
        self.giw_vals = np.asarray(list(map(self.get_giw, self.iwf_grid)))

        self.chi_tau_vals = np.asarray(
            list(map(self.get_chi_tau, self.tau_grid)))
        self.chi_iw_vals = np.asarray(
            list(map(self.get_chi_iw, self.iwb_grid)))

        self.W_tau_vals = np.asarray(
            list(map(self.get_W_from_chi, self.chi_tau_vals)))
        self.W_iw_vals = np.asarray(
            list(map(self.get_W_from_chi, self.chi_iw_vals)))

    def build_ham(self, t_matrix, mu, U, shift):
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
                ham_kin -= t_matrix[i, j] * (
                    clist[i] * alist[j] + clist[i + nsites] * alist[j +
                                                                    nsites])

        ham_int = 0
        for i in range(nsites):
            ham_int += U * nlist[i] * nlist[i + nsites]

        ham_mu = -mu * sum(nlist)
        if shift:
            ham_mu -= U / 2 * sum(nlist)

        self.t = t_matrix
        self.U = U
        self.mu = mu
        self.nflavors = nflavors
        self.nsites = nsites
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

    def get_g_moments(self):
        c1 = np.eye(self.nflavors)
        dens = np.diag(self.get_density_mat())
        nsites = self.nflavors // 2
        c2 = la.block_diag(self.t, self.t) - self.mu * np.eye(
            self.nflavors) + self.U * np.diag(np.roll(dens, nsites))
        # c3 = c2.dot(c2) + self.U**2 * np.roll(
        #     np.roll(
        #         self.get_density_corr() - np.outer(dens, dens), nsites,
        #         axis=0),
        #     nsites,
        #     axis=1)
        return [c1, c2]

    def get_chi_tau(self, tau):
        evolv1 = np.exp(-tau * self.eigval)
        evolv2 = np.exp(-(self.beta - tau) * self.eigval)
        dens = np.diag(self.get_density_mat())
        return np.einsum('a, iab, b, jba -> ij', evolv2,
                         np.asarray(self.nlist_diag), evolv1,
                         np.asarray(self.nlist_diag)) / self.Z - np.outer(
                             dens, dens)

    def get_chi_iw(self, w):
        boltz = np.exp(-self.beta * self.eigval)
        pref = (boltz[None, :] - boltz[:, None]) / (
            1j * w + self.eigval[:, None] - self.eigval[None, :])
        if w == 0:
            tmp = np.tile(self.eigval, (self.eigval.size, 1)).astype(complex)
            tmp[self.eigval[:, None] !=
                self.eigval[None, :]] = pref[self.eigval[:, None] !=
                                             self.eigval[None, :]]
            pref = tmp
        return np.einsum('ab, iab, jba -> ij', pref,
                         np.asarray(self.nlist_diag),
                         np.asarray(self.nlist_diag)) / self.Z

    def get_hubbard_umatrix(self, U):
        zero = np.zeros((self.nsites, self.nsites))
        eye = np.eye(self.nsites)
        return np.bmat([[zero, eye * U / 2], [eye * U / 2, zero]])

    def get_W_from_chi(self, chival):
        umat = self.get_hubbard_umatrix(self.U)
        return np.einsum('ab, bc, cd -> ad', umat, chival, umat)
