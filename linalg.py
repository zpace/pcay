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

def P_from_K_pinv(K, rcond=1e-4):
    '''
    compute P_PC using Moore-Penrose pseudoinverse (pinv)
    '''

    Kprime = np.moveaxis(K, [0, 1, 2, 3], [2, 3, 0, 1])
    swap = np.arange(Kprime.ndim)
    swap[[-2, -1]] = swap[[-1, -2]]
    u, s, v = np.linalg.svd(Kprime)
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

    #M = np.einsum('sk,ik,jk->sij', w, e, e)
    #F = np.einsum('sk,sk,jk->sj', w, f, e)

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
