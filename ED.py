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

def W_to_tensor(W):
    npoints, n1, n2 = W.shape
    assert n1 == n2
    W_tens = np.zeros((npoints, n1*n1, n2*n2), dtype=W.dtype)
    for i in range(n1):
        for j in range(n2):
            W_tens[:, i*n1+i, j*n2+j] = W[:, i, j]
    return W_tens

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

def umat_dd_to_full(umat):
    tens = np.zeros((*umat.shape, *umat.shape))
    it = np.nditer(umat, flags=['multi_index'])
    while not it.finished:
        i, j = it.multi_index
        tens[i,i,j,j] = it[0] / 2
        tens[i,j,i,j] = -it[0] / 2
        it.iternext()
    return tens

class HubbardHamiltonian:
    def __init__(self,
                 t_matrix,
                 mu,
                 U,
                 beta=10,
                 ntau=100,
                 niw=100,
                 shift=False):
        self._build_ham(t_matrix, mu, U, shift)
        ham = self.ham_total()
        e, V = la.eigh(self.ham_total())
        self.eigmin = e.min()
        self.eigval = e - self.eigmin
        self.eigvec = V

        self._alist_diag = self.transform_to_diag(self.alist)
        self._clist_diag = self.transform_to_diag(self.clist)
        self._nlist_diag = np.einsum('iab,ibc->iac', self._clist_diag, self._alist_diag)

        self.beta = beta
        self.ntau = ntau
        self.niw = niw
        self.Z = np.exp(-beta * self.eigval).sum()

        self.tau_grid = np.arange(ntau + 1) * beta / ntau
        self.iwf_grid = (np.arange(niw) * 2 + 1) * np.pi / beta
        self.iwb_grid = (np.arange(niw) * 2) * np.pi / beta
        self.gtau_vals = np.array([self.get_gtau(t) for t in self.tau_grid])
        self.giw_vals = np.array([self.get_giw(w) for w in self.iwf_grid])

        self.chi_tau_vals = np.array([self.get_chi_tau(t) for t in self.tau_grid])
        self.chi_iw_vals = np.array([self.get_chi_iw(w) for w in self.iwb_grid])
        
        self.chi4_tau_vals = np.array([self.get_chi4_tau(t) for t in self.tau_grid])
        self.chi4_iw_vals = np.array([self.get_chi4_iw(w) for w in self.iwb_grid])

        self.w_tau_vals = self.get_W_from_chi4(self.chi4_tau_vals)
        self.w_iw_vals = self.get_W_from_chi4(self.chi4_iw_vals)

        self.W_tau_vals = self.get_W_from_chi(self.chi_tau_vals)
        self.W_iw_vals = self.get_W_from_chi(self.chi_iw_vals)

    def _build_ham(self, t_matrix, mu, U, shift):
        t_matrix = np.asmatrix(t_matrix)
        assert la.norm(t_matrix - t_matrix.H) < 1.0e-16
        nsites = t_matrix.shape[0]
        nflavors = nsites * 2
        alist = gen_annihilators(nflavors)
        clist = [a.H for a in alist]
        nlist = [c.dot(a) for c, a in zip(clist, alist)]

        self.alist = np.array(alist)
        self.clist = np.array(clist)
        self.nlist = np.array(nlist)

        ham_kin = -np.einsum('ij,iab,jbc -> ac', 
                             la.block_diag(t_matrix, t_matrix), 
                             self.clist, self.alist)

        ham_int = U * np.einsum('iab,ibc -> ac', 
                                self.nlist[:nsites,...], 
                                self.nlist[nsites:,...])

        if shift:
            mu += U/2
        ham_mu = -mu * np.sum(self.nlist, axis=0)

        self.t = t_matrix
        self.U = U
        self.mu = mu
        self.nflavors = nflavors
        self.nsites = nsites
        self.ham_kin = ham_kin
        self.ham_int = ham_int
        self.ham_mu = ham_mu

    def ham_total(self):
        return self.ham_kin + self.ham_int + self.ham_mu

    def ham_canonical(self):
        return self.ham_kin + self.ham_int

    def transform_to_diag(self, matrix):
        return np.einsum('ab,...bc,cd->...ad', self.eigvec.conj().T, matrix, self.eigvec)
        #return self.eigvec.H * matrix * self.eigvec
    
    def get_energy(self):
        return np.sum(np.exp(-self.beta*self.eigval) * self.eigval) / self.Z + self.eigmin

    def evolve(self, tau):
        return np.exp(-tau * self.eigval)

    # G_ij(t) = -<T c_i(t) c^+_j(0)>
    def get_gtau(self, tau):
        sign = 1
        if tau < 0:
            tau += self.beta
            sign = -1
        return -sign * np.einsum('a, iab, b, jba -> ij', self.evolve(self.beta - tau),
                                 self._alist_diag, self.evolve(tau),
                                 self._clist_diag) / self.Z

    # G(iw) = \int_0^\beta G(t) exp(iwt) dt
    def get_giw(self, w):
        boltz = np.exp(-self.beta * self.eigval)
        pref = (boltz[:,None] + boltz[None,:]) / (
            1j * w + self.eigval[:,None] - self.eigval[None,:])
        return np.einsum('ab, iab, jba -> ij', pref,
                         np.asarray(self._alist_diag),
                         np.asarray(self._clist_diag)) / self.Z

    # \rho_ij = <c^+_i c_j> = G_ji(-0)
    def get_density_mat(self):
        return np.einsum('a, iab, jba -> ij',
                         self.evolve(self.beta),
                         self._clist_diag,
                         self._alist_diag) / self.Z

    def get_density_corr(self):
        return np.einsum('a, iab, jba -> ij',
                         self.evolve(self.beta),
                         self._nlist_diag,
                         self._nlist_diag) / self.Z

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

    # \chi_ijkl(t) = <c^+_i(t) c_j(t) c^+_k(0) c_l(0)> - \rho_ij\rho_kl
    def get_chi4_tau(self, tau):
        densmat = self.get_density_mat()
        cdagc = np.einsum('Iab,Jbc->IJac', self._clist_diag, self._alist_diag)
        return np.einsum('a,IJac,c,KLca->IJKL',
            self.evolve(self.beta - tau),
            cdagc,
            self.evolve(tau),
            cdagc
        ) / self.Z - densmat[:,:,None,None] * densmat[None,None,:,:]

    def get_chi4_iw(self, w):
        densmat = self.get_density_mat()
        cdagc = np.einsum('Iab,Jbc->IJac', self._clist_diag, self._alist_diag)
        boltz = np.exp(-self.beta * self.eigval)
        with np.errstate(divide='ignore', invalid='ignore'):
            pref = (boltz[None, :] - boltz[:, None]) / (
                1j * w + self.eigval[:, None] - self.eigval[None, :])
            if w == 0:
                tmp = self.beta * np.tile(boltz, (self.eigval.size,
                                                  1)).astype(np.complex128)
                pref[self.eigval[:, None] == self.eigval[None, :]] = tmp[
                    self.eigval[:, None] == self.eigval[None, :]]
        res = np.einsum('ac, IJac, KLca -> IJKL', pref,
            cdagc,
            cdagc
        ) / self.Z
        if w == 0:
            res -= densmat[:,:,None,None] * densmat[None,None,:,:] * self.beta
        return res

    def get_chi_tau(self, tau):
        evolv1 = np.exp(-tau * self.eigval)
        evolv2 = np.exp(-(self.beta - tau) * self.eigval)
        dens = np.diag(self.get_density_mat())
        return np.einsum('a, iab, b, jba -> ij', evolv2,
                         self._nlist_diag, evolv1,
                         self._nlist_diag) / self.Z - np.outer(
                             dens, dens)

    def get_chi_iw(self, w):
        boltz = np.exp(-self.beta * self.eigval)
        with np.errstate(divide='ignore', invalid='ignore'):
            pref = (boltz[None, :] - boltz[:, None]) / (
                1j * w + self.eigval[:, None] - self.eigval[None, :])
            if w == 0:
                tmp = self.beta * np.tile(boltz, (self.eigval.size,
                                                  1)).astype(np.complex128)
                pref[self.eigval[:, None] == self.eigval[None, :]] = tmp[
                    self.eigval[:, None] == self.eigval[None, :]]
        dens = np.diag(self.get_density_mat())
        res = np.einsum('ab, iab, jba -> ij', pref,
                        np.asarray(self._nlist_diag),
                        np.asarray(self._nlist_diag)) / self.Z
        if w == 0:
            res -= np.outer(dens, dens) * self.beta
        return res

    def get_hubbard_umatrix(self, U):
        zero = np.zeros((self.nsites, self.nsites))
        eye = np.eye(self.nsites)
        return np.bmat([[zero, eye * U], [eye * U, zero]])

    def get_W_from_chi(self, chival):
        umat = self.get_hubbard_umatrix(self.U)
        return np.einsum('ab,...bc, cd ->...ad', umat, chival, umat)

    def get_W_from_chi4(self, chi4val):
        utens = umat_dd_to_full(self.get_hubbard_umatrix(self.U))
        return -np.einsum('ijKL,...KLIJ,IJkl->...ijkl', utens, chi4val, utens)

    def get_W_moments(self):
        umat = self.get_hubbard_umatrix(self.U)
        c1 = np.zeros((self.nflavors, self.nflavors))
        rho = self.get_density_mat()
        tmat = la.block_diag(self.t, self.t)
        tca = np.asarray(tmat) * rho
        c2 = tca + tca.T - np.diag(np.sum(tca, axis=0) + np.sum(tca, axis=1))
        c2 = np.einsum('ab, bc, cd -> ad', umat, c2, umat)
        return [c1, c2]

    def write_g(self, fname):
        write_tau_gf(
            fname,
            '/G_tau',
            self.gtau_vals[:, :self.nsites, :self.nsites],
            self.beta,
            stat='fermi')
        write_iw_gf(
            fname,
            '/G',
            self.giw_vals[:, :self.nsites, :self.nsites],
            self.beta,
            stat='fermi',
            tails=self.get_g_moments())

    def write_W(self, fname):
        tails = self.get_W_moments()
        write_tau_gf(
            fname,
            '/W_tau_samespin',
            W_to_tensor(self.W_tau_vals[:, :self.nsites, :self.nsites]),
            self.beta,
            stat='bose')
        write_iw_gf(
            fname,
            '/W_samespin',
            W_to_tensor(self.W_iw_vals[:, :self.nsites, :self.nsites]),
            self.beta,
            stat='bose',
            tails=[c[:self.nsites, :self.nsites] for c in tails])
        write_tau_gf(
            fname,
            '/W_tau_diffspin',
            W_to_tensor(self.W_tau_vals[:, :self.nsites, self.nsites:]),
            self.beta,
            stat='bose')
        write_iw_gf(
            fname,
            '/W_diffspin',
            W_to_tensor(self.W_iw_vals[:, :self.nsites, self.nsites:]),
            self.beta,
            stat='bose',
            tails=[c[:self.nsites, self.nsites:] for c in tails])
