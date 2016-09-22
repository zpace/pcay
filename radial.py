import numpy as np

from sklearn.gaussian_process import GaussianProcess


def radial_gp(r, q, q_unc, q_bdy=[-np.inf, np.inf]):
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

    nugget = (q_unc / q)**2.
    # this is bad and I should feel bad
    nugget = np.maximum(nugget, .05 * np.ones_like(nugget))

    gp = GaussianProcess(
        regr='constant', corr='squared_exponential', nugget=nugget,
        theta0=.5, thetaL=.05, thetaU=1.)
    gp.fit(r, q)

    return gp
