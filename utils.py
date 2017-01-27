import numpy as np
import pickle as pkl

from astropy import units as u, constants as c, table as t
from specutils import extinction
from speclite import redshift as slrs
from speclite import accumulate as slacc

from scipy.ndimage.filters import gaussian_filter1d as gf

ln10 = np.log(10.)

class ArrayPartitioner(object):
    '''
    partition arrays along an axis, and return an iterator

    Iterator produces all the subarrays of `arrays` along LAST TWO dimensions

    Parameters:
    -----------

    arrays : list
        list of arrays to partition identically

        Each of these arrays must have identical shape in final two dimensions

    lgst_ix : int (default: 0)
        index of `arrays` that corresponds to the array that limits the size
        of individual operations

    lgst_el_shape : tuple
        shape of each intermediate array element that needs to be handled

    memlim : int
        limit of memory for each block of several lgst_el_size to take up

    Example:
    --------

    `lim_array` has shape (47, 47, 24, 24), and `arrays` has elements with
    shape (..., 24, 24) (the ... could be nothing). In this case,
    `lgst_el_shape` = (24, 24). The iterator will return portions of
    the len-(47*47=2209) "flattened" array
    '''

    def __init__(self, arrays, lgst_el_shape, lgst_ix=0, memlim=2147483648):
        self.elshape = lgst_el_shape
        self.imshape = arrays[lgst_ix].shape[-2:]

        def f(a):
            if len(a.shape) == 2:
                return a.flatten()
            else:
                return np.moveaxis(a.reshape((self.elshape +
                                              (np.prod(self.imshape), ))),
                                   [0, 1, 2], [1, 2, 0])

        self.arrays = [f(a) for a in arrays]

        # calculate the max number of elements of size lgst_el_size
        # size in memory of largest intermediate element
        lgst_el_size = np.empty(lgst_el_shape).nbytes
        # how many intermediate subarrays are possible to
        # store at once given memlim
        self.M = memlim // lgst_el_size

        # the basic idea is that we get N partitions such that each partition
        # has at most size M
        self.N = np.prod(self.imshape) // self.M

        self.ct = 0  # sentinel value

    def __iter__(self):
        return self

    def __next__(self):
        if self.ct > self.N:
            raise StopIteration
        elif self.ct == self.N:
            r = (a[..., (np.prod(self.imshape) - self.M * self.N):]
                 for a in self.arrays)
        else:
            r = (a[..., (self.ct * self.M):((self.ct + 1) * self.M)]
                 for a in self.arrays)

        self.ct += 1

        return r

def pickle_loader(fname):
    with open(fname, 'rb') as f:
        p = pkl.load(f)

    return p

def matcher(x, s):
    if s in x:
        return True
    else:
        return False

def lin_transform(r1, r2, x):
    '''
    transform x from range 1 to range 2
    '''

    return (((x - r1[0]) * (r2[1] - r2[0])) / (r1[1] - r1[0])) + r2[0]

def gaussian_filter(spec, sig):
    '''
    variable-width convolution of a spectrum

    inspired by Michele Cappellari's similar routine

    params:
        - spec: vector of a single spectrum
        - sig: vector giving width of gaussian peak (in pixels)
    '''

    p = int(np.ceil(3. * np.max(sig)))
    m = 2 * p + 1  # kernel size
    x2 = np.linspace(-p, p, m)**2

    n = spec.size
    a = np.zeros((m, n))
    for j in range(m):   # Loop over the small size of the kernel
        a[j, p:-p] = spec[j:n - m + j + 1]

    gau = np.exp(-x2[:, None] / (2 * sig**2.))
    gau /= np.sum(gau, axis=0)[None, :]  # Normalize kernel

    f = np.sum(a * gau, axis=0)

    return f

def add_losvds(meta, spec, dlogl, vmin=10, vmax=500, nv=10, LSF=None):
    '''
    take spectra and blur each one a few times
    '''

    if LSF is None:
        LSF = np.zeros_like(spec[0, :])

    RS = np.random.RandomState()

    i_s = range(spec.shape[0])

    meta, spec = zip(*[_add_losvds_single(m, s, dlogl, vmin, vmax, nv,
                                          RS, LSF, i)
                       for m, s, i in zip(meta, spec, i_s)])
    meta = t.vstack(meta)
    spec = np.row_stack(spec)

    return meta, spec

def _add_losvds_single(meta, spec, dlogl, vmin, vmax, nv, RS, LSF, i):

    vels = RS.uniform(vmin, vmax, nv) * u.Unit('km/s')

    # dlogl is just redshift per pixel
    z_ = (vels / c.c).decompose().value
    sig = ln10 * (z_ / dlogl)
    sig = np.atleast_2d(sig).T
    sig = np.sqrt(sig**2. + LSF**2.)
    sig = sig.clip(min=.01, max=None)

    meta = t.vstack([meta, ] * len(vels))
    meta['sigma'] = vels.value

    spec = np.row_stack([gaussian_filter(spec, s) for s in sig])

    if i % 10 == 0:
        print('Done with {}'.format(i))

    return meta, spec

def weighted_quantile(p, w, q):
    '''
    like numpy.percentile, but supports weights and behaves in
        vectorized fashion. Basically computes pctls of n samples
        in (X, Y) grid

    params:
        - p: array-like, shape (n): parameter values
        - w: array-like, shape (n, X, Y): weights of parameters
        - q: array-like, shape (p): quantiles needed

    Note: assumes that map-like axes are final 2
    '''

    # sort p and w by increasing value of p
    sorter = np.argsort(p)
    p, w = p[sorter], w[sorter]

    wt_qtls = np.cumsum(w, axis=0) - 0.5 * w.sum(axis=0)
    wt_qtls /= w.sum(axis=0)

    interp = np.interp(q, wt_qtls, p)

def extinction_correct(l, f, EBV, r_v=3.1, ivar=None, **kwargs):
    '''
    wraps around specutils.extinction.reddening
    '''

    a_v = r_v * EBV
    r = extinction.reddening

    # output from reddening is inverse-flux-transmission
    f_itr = r(wave=l, a_v=a_v, r_v=r_v, **kwargs)[..., None, None]
    # to deredden, divide f by f_itr

    f /= f_itr

    if ivar is not None:
        ivar *= f_itr**2.
        return f, ivar
    else:
        return f

def add_redshifts(zs, axis=0):
    '''
    add redshifts in an array-like, along an axis
    '''

    z_tot = np.cumprod((1. + zs), axis=axis) - 1.
    return z_tot

def redshift(l, f, ivar, **kwargs):
    '''
    wraps around speclite.redshift, and does all the rule-making, etc

    assumes l, flam, ivar of flam
    '''

    s = f.shape
    l = np.tile(l[..., None, None], (1, ) + s[-2:])
    data = np.empty_like(
        f, dtype=[('l', float), ('f', float), ('ivar', float)])
    data['l'] = l
    data['f'] = f
    data['ivar'] = ivar
    rules = [dict(name='l', exponent=+1),
             dict(name='f', exponent=-1),
             dict(name='ivar', exponent=+2)]

    res = slrs(data_in=data, rules=rules, **kwargs)

    return res['l'], res['f'], res['ivar']

def coadd(l, f, ivar, l_ax=0, accum_axs=(1, 2), **kwargs):
    nl = f.shape[l_ax]
    map_shape = tuple(f.shape[i] for i in accum_axs)
    cube_shape = f.shape

    data = np.empty_like(f, dtype=[('lam', float), ('flam', float),
                                   ('ivar', float)])
    data['lam'] = l
    data['flam'] = f
    data['ivar'] = ivar

    result = None

    for ix in np.ndindex(map_shape):
        slc = [slice(None)] * len(cube_shape)
        for i, a in enumerate(accum_axs):
            slc[a] = slice(ix[i], ix[i] + 1)
        result = slacc(data1_in=result, data2_in=data[slc].flatten(),
                       data_out=result, join='lam', add='flam',
                       weight='ivar')

    lnew, fnew, ivarnew = result['lam'], result['flam'], result['ivar']

    return lnew, fnew, ivarnew

hdu_unit = lambda hdu: u.Unit(hdu.header['BUNIT'])
