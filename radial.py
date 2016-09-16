import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def radial_gp(r, q, q_unc):
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

    if map(type, [r, q, q_unc]) != [np.ndarray, ] * 3:
        raise ValueError('provide arrays')

    assert r.size == q.size == q_unc.size, 'provide same-sized arrays'

    r, q, q_unc = r.flatten(), q.flatten(), q_unc.flatten()
    r = np.atleast_2d(r).T

    kernel = C(1.0, (1e-2, 1e1)) * RBF(1, (1e-2, 5))
    gp = GaussianProcessRegressor(alpha=(q_unc / q)**2., kernel=kernel,
                                  n_restarts_optimizer=10)
    gp.fit(r, q)

    return gp
