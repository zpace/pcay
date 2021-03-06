import numpy as np
from scipy.linalg import pinv2
from utils import SqFromSqCacher

from scipy.optimize import curve_fit, OptimizeWarning
from scipy import linalg as spla
from scipy.sparse import diags

from functools import lru_cache

eps = np.finfo(float).eps

def prep_broadcastable(K, spatial_axes):
    '''
    how to prepare an array for broadcasting
    '''
    n_spatial_axes = len(spatial_axes)
    spatial_shape = [s for s in K.shape if s in spatial_axes]
    operable_shape = [s for s in K.shape if s not in spatial_axes]
    broadcast_spatial_axes = list(range(-n_spatial_axes, 0))

    return spatial_shape, operable_shape, broadcast_spatial_axes

def spla_solve_posdef(K, b):
    '''
    solve positive-definite matrix
    '''
    return spla.solve(K, b, assume_a='pos', check_finite=False)

spla_solve_posdef_vectorized = np.vectorize(
        spla_solve_posdef, signature='(n,n),(n,p)->(n,n)', otypes=[np.ndarray,])

def broadcasted_sinvert(K, spatial_axes):
    '''
    compute inverse of all q-by-q sub-arrays of K, obeying broadcasting rules
    '''
    spatial_shape, operable_shape, broadcast_spatial_axes = prep_broadcastable(
        K, spatial_axes)
    # move spatial axes to last few
    K = np.moveaxis(K, spatial_axes, broadcast_spatial_axes)

    # make multidimensional identity matrix with ones along diag of axes (0, 1)
    multidim_eye = np.zeros(operable_shape + spatial_shape)
    np.einsum('ii...->i...', multidim_eye)[:] = 1.

    P = spla_solve_posdef_vectorized(K, multidim_eye).astype(K.dtype)

    P = np.moveaxis(broadcast_spatial_axes, spatial_axes)

    return P

def spla_chol_invert(K, eye):
    '''
    invert a positive-definite matrix using cholesky decomposition
    '''
    Ltup = spla.cho_factor(K, lower=True)
    K_inv = spla.cho_solve(Ltup, eye, check_finite=False)
    return K_inv

spla_chol_invert_vectorized = np.vectorize(
    spla_chol_invert, signature='(n,n),(n,p)->(n,n)', otypes=[np.ndarray])

def broadcasted_cholinvert(K, spatial_axes):
    '''
    compute inverse of all q-by-q sub-arrays of K, obeying broadcasting uses,
        using Cholesky decomposition
    '''
    spatial_shape, operable_shape, broadcast_spatial_axes = prep_broadcastable(
        K, spatial_axes)
    # move spatial axes to last few
    K = np.moveaxis(K, spatial_axes, broadcast_spatial_axes)

    # make multidimensional identity matrix with ones along diag of axes (0, 1)
    multidim_eye = np.zeros(operable_shape + spatial_shape)
    np.einsum('ii...->i...', multidim_eye)[:] = 1.

    P = spla_chol_invert_vectorized(K, multidim_eye).astype(K.dtype)

    P = np.moveaxis(broadcast_spatial_axes, spatial_axes)

    return P

class PCAProjectionSolver(object):
    '''
    projects data down onto PCs
    '''
    def __init__(self, e, K_inst_cacher, K_th, regul=.1):
        self.e = e
        self.q, self.nl = e.shape
        self.K_inst_cacher = K_inst_cacher
        self.K_th = K_th

        self.eTe = e.T @ e
        self.inv_eTe = spla_chol_invert(
            self.eTe + regul * np.diag(np.diag(self.eTe)),
            np.eye(self.nl))
        self.H = self.inv_eTe @ e.T

        self.K_PC_th = self.H.T @ self.K_th @ self.H

        self.regul = regul * np.ones(self.nl)

    def solve_single(self, f, var, mask, a, lam_i0, nodata):
        if nodata or (mask.mean() > .3):
            success = False
            return np.zeros(self.q), .0001 * np.eye(self.q), success

        K_PC_inst = self.K_inst_cacher.covwindows.all_K_PCs[lam_i0]

        fr = 0.5
        offdiag = (fr * var[1:] + (1. - fr) * var[:-1])
        K_meas = diags([offdiag, var + self.regul, offdiag], [-1, 0, 1])

        K_PC_meas = self.H.T @ K_meas @ self.H
        K_PC = K_PC_inst + K_PC_meas + self.K_PC_th

        try:
            A = f @ self.H
            P_PC = spla_chol_invert(K_PC, np.eye(self.q))
        except (spla.LinAlgError, ValueError):
            success = False
            A, P_PC = np.zeros(self.q), 1.0e-4 * np.eye(self.q)
        else:
            success = True

        return A, P_PC, success


def gen_Kinst(nl, lims=(-.01, .03), nsamp=1000, rms=.01):
    import sklearn.covariance as sklcov
    samples = np.random.randn(nsamp)[None, :] * np.linspace(*lims, nl)[:, None]
    samples_noise = rms * np.random.randn(nsamp, nl)

    cov_reg = sklcov.ShrunkCovariance(shrinkage=.05, store_precision=True)
    cov_reg.fit(samples.T + samples_noise)

    #cov = np.cov(samples + samples_noise.T)

    return cov_reg

def tuple_insert(tup, pos, ele):
    tup = tup[:pos] + (ele, ) + tup[pos:]
    return tup

def tuple_delete(tup, pos):
    tup = tuple(e for i, e in enumerate(tup) if i != pos)

class HighDimDataSet(object):
    '''
    artificial, high-dimensionality data set to do some PCA tests on
    '''
    def __init__(self, M, E_full, K_inst, q, x):
        self.M = M
        self.E_full = E_full  # full PC basis
        self.E = E_full[:q, :]  # reduced PC basis
        self.q = q  # size of reduced PC basis
        self.n = self.E.shape[1]  # number of channels per measurement
        self.x = x  # locations of channels
        self.K_inst = K_inst

    @classmethod
    def Randomize(cls, x, q, Kinst_kwargs={}):
        from utils import random_orthogonal_basis

        nl = len(x)

        Mf = lambda slope, xint, x: slope * (x - xint)
        M0 = np.minimum(Mf(2., 3700., 10.**x), Mf(-.1, 15000., 10.**x))
        M = M0 / M0.mean()

        E_full = random_orthogonal_basis((nl, nl))
        K_inst = gen_Kinst(nl, **Kinst_kwargs)

        return cls(M, E_full, K_inst, q, x)

    def gen(self, snr, x_axis=0, ivar_precision=.05, structure_shape=(1, )):
        '''
        generate data from full PC basis, and noisify according to snr
        '''

        if x_axis < 0:
            raise ValueError('x axis index must be positive')

        # since in this case we're using all PCs to construct fake data
        q = self.n
        self.x_axis = x_axis

        # if SNR is a single number, just return a single spectrum
        if not hasattr(snr, '__len__'):
            snr = snr * np.ones_like(self.x)
            fulldata_shape = (self.n, )
            coeffs_shape = (q, )
        # if SNR is given as a map (i.e., has an incompatible shape to self.x),
        # then add a dimension where specified in x_axis to make shapes compatible
        elif self.n not in snr.shape:
            # define higher-dimensional data structure shape
            # that delimits separate measurements
            structure_shape = snr.shape
            snr = np.expand_dims(snr, x_axis)
            snr = np.repeat(snr, self.n, axis=x_axis)
            fulldata_shape = snr.shape
            coeffs_shape = tuple_insert(structure_shape, x_axis, q)
        else:
            structure_shape = tuple_delete(snr.shape, x_axis)
            fulldata_shape = snr.shape
            coeffs_shape = tuple_insert(structure_shape, x_axis, q)

        self.snr = snr

        self.A0 = np.random.randn(*coeffs_shape)
        # generate centered data, and then add mean
        self.obs0_ctrd = np.moveaxis(
            (np.moveaxis(self.A0, x_axis, -1) @ self.E_full.T), -1, x_axis)
        self.obs0 = np.moveaxis(
            np.moveaxis(self.obs0_ctrd, x_axis, -1) + self.M, -1, x_axis)
        obs_noise = self.obs0 * np.random.randn(*fulldata_shape) / snr
        spectrophotometric_noise = np.moveaxis(
            np.random.multivariate_normal(
                np.zeros(self.n), self.K_inst.covariance_,
                         structure_shape),
                -1, x_axis)

        self.obs = self.obs0 + obs_noise + spectrophotometric_noise
        self.ivar0 = (snr / self.obs)**2.
        self.ivar = (self.ivar0 * (1. + ivar_precision * \
                     np.random.randn(*self.ivar0.shape))).clip(min=0.)

def run_pca(S, q=None):
    R = np.cov(S, rowvar=False)
    # calculate evecs & evalse of covariance matrix
    # (use 'eigh' rather than 'eig' since R is symmetric for performance
    evals_, evecs_ = np.linalg.eigh(R)
    # sort eigenvalues and eigenvectors in decreasing order
    idx = np.argsort(evals_)[::-1]
    evals_, evecs_ = evals_[idx], evecs_[:, idx].T
    # and select first `q`
    evals, evecs = evals_[:q], evecs_[:q]

    return evals_, evals, evecs_, evecs

def quick_data_to_PC(specs, e, regul=.1):
    q, nl = e.shape

    eTe = e.T @ e
    inv_eTe = spla_chol_invert(
        eTe + regul * np.diag(np.diag(eTe)), np.eye(nl))
    H = inv_eTe @ e.T

    # carry out the transformation on the data using eigenvectors
    A = specs @ H

    return A
