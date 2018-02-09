import numpy as np

eps = np.finfo(float).eps

def SVD_downproject(M, F, rcond=1e-10):
    # do lstsq down-projection using SVD

    u, s, v = np.linalg.svd(M, full_matrices=False)
    s_max = s.max(axis=-1, keepdims=True)
    s_min = rcond * s_max
    inv_s = np.zeros_like(s)
    inv_s[s >= s_min] = 1. / s[s >= s_min]

    x = np.einsum('...ji,...j->...i', v,
                  inv_s * np.einsum('...ji,...j->...i', u, F.conj()))

    return np.conj(x, x)

def broadcasted_pinvert(K, rcond=1e-4, moveaxes=True):
    '''
    compute P_PC using Moore-Penrose pseudoinverse (pinv)
    '''
    if moveaxes:
        K = np.moveaxis(K, [0, 1, 2, 3], [2, 3, 0, 1])

    swap = np.arange(K.ndim)
    swap[[-2, -1]] = swap[[-1, -2]]
    u, s, v = np.linalg.svd(K)
    cutoff = np.maximum.reduce(s, axis=-1, keepdims=True) * rcond

    mask = s > cutoff
    s[mask] = 1. / s[mask]
    s[~mask] = 0.

    a = np.einsum('...uv,...vw->...uw',
                  np.transpose(v, swap) * s[..., None, :],
                  np.transpose(u, swap))
    return np.moveaxis(a, [0, 1, 2, 3], [2, 3, 0, 1])

def P_from_K(K):
    '''
    compute straight inverse of all elements of K_PC [q, q, NX, NY]
    '''
    P_PC = np.moveaxis(
        np.linalg.inv(
            np.moveaxis(K, [0, 1, 2, 3], [2, 3, 0, 1])),
        [0, 1, 2, 3], [2, 3, 0, 1])
    return P_PC

def robust_project_onto_PCs(e, f, w=None):
    '''
    project a set of measured spectra with measurement errors onto
        the principal components of the model library

    params:
     - e: (q, l) array of eigenvectors
     - f: (l, x, y) array, where n is the number of spectra, and m
        is the number of spectral wavelength bins. [Flux-density units]
     - w: (l, x, y) array, containing the inverse-variances for each
        of the rows in spec [Flux-density units]
        (this functions like the array w_lam in REF [1])

    REFERENCE:
        [1] Connolly & Szalay (1999, AJ, 117, 2052)
            [particularly eqs. 3, 4, 5]
            (http://iopscience.iop.org/article/10.1086/300839/pdf)

    This should be done on the largest number of spectra possible
        simultaneously, for greatest speed. Rules of thumb to come.
    '''

    if w is None:
        w = np.ones_like(f)
    elif type(w) != np.ndarray:
        raise TypeError('w must be array')
    # make singular system non-singular
    else:
        w[w == 0] = eps

    M = np.einsum('kxy,ik,jk->xyij', w, e, e)
    F = np.einsum('kxy,kxy,jk->xyj', w, f, e)

    try:
        A = np.linalg.solve(M, F)
    except np.linalg.LinAlgError:
        warn('down-projection is non-exact, due to singular matrix',
             PCProjectionWarning)
        # try lstsq down-projection
        A = SVD_downproject(M, F)

    A = np.moveaxis(A, -1, 0)

    #A, resid, rank, S = np.linalg.lstsq(M, F)
    #A = np.moveaxis(A, -1, 0)

    return A

def propagate_weights_to_PCcov(e, w=None):
    '''
    propagate weights through linear system,
        using Eq 6 from Connolly & Szalay (1999)
    '''
    w = w.clip(min=eps)

    M = np.einsum('kxy,ik,jk->xyij', w, e, e)
    M_pinv = broadcasted_pinvert(M, moveaxes=False)

    K_PC = M_pinv / np.sum(w, axis=0)
    return K_PC

def propagate_cov_to_PCcov(K_spec, E, w, denom=None, return_denom=False):
    '''
    propagate spectral covariance through linear system,
        modulated by weights w
    '''
    # in this einsum, i & j refer to PC numbers, l & m to wavelength channels,
    # all other axes are preserved and brought to final two dimensions
    num = np.einsum('il,l...,lm,m...,jm->ij...', E, w, K_spec, w, E)
    if denom is None:
        denom = np.einsum('il,l...,m...,jm->ij...', E, w, w, E)

    K_PC = num / denom

    if return_denom:
        return K_PC, denom

    return K_PC

def propagate_cov_to_PCcov_single(K_spec, E, w, denom=None, return_denom=False):
    '''
    propagate covariance through linear system,
        modulated by weights w, for a single measurement vector
    '''
    num = np.einsum('il,l,lm,m,jm->ij', E, w, K_spec, w, E)
    if denom is None:
        denom = np.einsum('il,l,m,jm->ij', E, w, w, E)

    K_PC = num / denom

    if return_denom:
        return K_PC, denom

    return K_PC

def propagate_var_to_PCcov(var, E, w, denom=None, return_denom=False):
    '''
    propagate variance through linear system, modulated by weights
        which may be identical to inverse-variance
    '''
    num = np.einsum('il,l...,l...,m...,jm->ij...', E, w, var, w, E)
    if denom is None:
        denom = np.einsum('il,l...,m...,jm->ij...', E, w, w, E)

    K_PC = num / denom

    if return_denom:
        return K_PC, denom

    return K_PC

def make_PCcov(K_spec, E, w, var):
    '''
    propagate weights `w` (1), spectral covariance `K_spec` (2),
        and observational uncertainty (3) through linear system def'd by `E`

    propagations are also subject to weights `w`
    '''
    wtscov = propagate_weights_to_PCcov(E, w)
    covcov, denom = propagate_cov_to_PCcov(K_spec, E, w, return_denom=True)
    varcov = propagate_var_to_PCcov(var, E, w, denom=denom)

    full_PCcov = wtscov + covcov + varcov

    return full_PCcov

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
