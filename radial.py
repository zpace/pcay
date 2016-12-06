import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Const


def radial_gp(r, q, q_unc, scale, q_bdy=[-np.inf, np.inf]):
    '''
    compute radial gaussian process for regressing some quantity against Re

    Parameters
    ----------

    r : :obj:`np.ndarray`
        radial coordinate, in units of Re

    q : :obj:`np.ndarray`
        quantity to regress against radius

    unc_q : :obj:`np.ndarray`
        uncertainty (standard error) of quantity `q`
    '''

    assert r.size == q.size == q_unc.size, 'provide same-sized arrays'

    if scale == 'log':
        q_gp, q_unc_gp = 10.**q, 10.**(q_unc) - 1.
    else:
        q_gp, q_unc_gp = q, q_unc

    r, q_gp, q_unc_gp = r.flatten(), q_gp.flatten(), q_unc_gp.flatten()
    r = np.ma.masked_outside(r, *q_bdy)
    r, q_gp, q_unc_gp = r[~r.mask], q_gp[~r.mask], q_unc_gp[~r.mask]
    r = np.atleast_2d(r).T

    nugget = (q_unc_gp / q_gp)**2.

    # build a very flexible kernel: can nominally handle factor
    # of 10 variation in 1 Re, but may not need to
    kernel = Const(.2, (.01, 10.)) * RBF(1, (.5, 3.))

    gp = GPR(kernel=kernel, alpha=nugget, n_restarts_optimizer=10)
    try:
        gp.fit(r, q_gp)
    except GPFitError:
        raise

    return gp

class GPFitError(Exception):
    pass
