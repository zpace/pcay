import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
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

    r, q, q_unc = r.flatten(), q.flatten(), q_unc.flatten()
    r = np.ma.masked_outside(r, *q_bdy)
    r, q, q_unc = r[~r.mask], q[~r.mask], q_unc[~r.mask]
    r = np.atleast_2d(r).T

    # build a very flexible kernel: can nominally handle factor
    # of 20 variation in 1/4 Re, but may not need to
    if scale == 'linear':
        kernel = Const(5., (1.0e-2, 20.)) * RBF(.25, (.05, 2.))
        nugget = (q_unc / q)**2.
    elif scale == 'log':
        kernel = (Const(np.log10(5.), (-2, np.log10(20.))) *
                  RBF(.25, (.05, 2.)))
        nugget = 10.**(2. * (q_unc - q))
    else:
        raise ValueError('invalid scale!')

    gp = GaussianProcessRegressor(kernel=kernel, alpha=nugget,
                                  n_restarts_optimizer=10)
    gp.fit(r, q)

    return gp
