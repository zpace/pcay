import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from astropy import units as u, constants as c, table as t
from astropy.io import fits
from astropy.stats import sigma_clip
import extinction
from speclite import redshift as slrs
from speclite import accumulate as slacc

from scipy.ndimage.filters import gaussian_filter1d as gf
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform

import os
import sys
from copy import copy
from functools import lru_cache

from importer import *

# personal
import manga_tools as m

ln10 = np.log(10.)

def gaussian_weightify(vals, mu, sigma=None, ivar=None, soft=1.):
    '''
    give a weight to each value in `vals` according to how close it is to `mu`,
        in a Gaussian sense

    params:
     - mu: nominal value in each spaxel
     - sigma: standard deviation of `mu`
     - vals: 1-d array of values to compare to `mu` and `sigma`
    '''

    dist = vals[:, None, None] - mu[None, :, :]

    if (ivar is None) and (sigma is None):
        raise ValueError('give me either sigma or ivar')
    elif (ivar is None):
        wts = np.exp(-dist**2. / (2. * sigma**2. * soft**2.))
    else:
        wts = np.exp(-dist**2. * ivar / (2. * soft**2.))
    return wts

def weighted_pctls_single(a, w=None, qtls=[50]):
    if w is None:
        w = np.ones_like(a)

    w[np.logical_or.reduce((~np.isfinite(a), ~np.isfinite(w)))] = 0

    i_ = np.argsort(a, axis=0)
    a, w = a[i_], w[i_]
    qvals = np.interp(
        qtls, 100. * w.cumsum() / w.sum(), a)
    return qvals

def copyFITS(fname):
    hdulist = fits.open(fname)
    hdulist_copy = copy(hdulist)

    hdulist.close()

    return hdulist_copy

class GaussPeak(object):
    def __init__(self, pos, wid, ampl=None, flux=None):
        self.pos = pos
        self.wid = wid

        if (not ampl) and (not flux):
            self.ampl = 1.
        elif not flux:
            pass
        elif not ampl:
            self.ampl = 1. / (self.wid * np.sqrt(2. * np.pi))
        else:
            raise ValueError('specify either or none of (ampl, flux)')

    @property
    def flux(self):
        return self.ampl * self.wid * np.sqrt(2. * np.pi)

    def __call__(self, x):
        return self.ampl * np.exp(-(x - self.pos)**2. / (2. * self.wid))


def multigaussflux(POSs, WIDs, FLUXs, x):
    peaks = [GaussPeak(pos=pos, wid=wid, flux=flux)
             for (pos, wid, flux) in zip(POSs, WIDs, FLUXs)]
    res = np.add.reduce([peak(x) for peak in peaks])

    return res

def replace_bad_data_with_wtdmean(a, ivar, mask, wid=201):
    '''
    replace bad data with weighted mean of surrounding `wid` pixels
    '''

    assert type(wid) is int, 'window width must be integer'
    if wid % 2 == 0:
        wid += 1

    spec_snr = np.abs(a) * np.sqrt(ivar)

    pad = wid // 2
    # pad `a` & `ivar` at the ends
    pad_width = [[pad, pad], [0, 0], [0, 0]]
    a_, ivar_ = (np.pad(a, pad_width=pad_width, mode='median',
                        stat_length=pad_width),
                 np.pad(ivar, pad_width=pad_width, mode='median',
                        stat_length=pad_width))
    mask_ = np.pad(mask, pad_width=pad_width, mode='edge')

    # now set up rolling median filter
    def rolling_window(a, wid):
        shape = (wid, a.shape[0] - wid + 1) + a.shape[1:]
        strides = (a.strides[0], ) + a.strides
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    a_windows = rolling_window(a_, wid)
    wt_windows = rolling_window(ivar_ * ~mask_, wid) + np.finfo(ivar.dtype).eps

    a_new = 1. * a
    a_new[mask] = 0.

    fill_value = np.average(a_windows, weights=wt_windows, axis=0)

    a_new[mask] = fill_value[mask]

    return a_new

def find_bad_data(a, ivar, wid=201, snr_mult_thresh=.1):
    '''
    find bad data in array `a`
    '''

    ivar_mult = snr_mult_thresh**2.  # if snr dec by f.o. 2, ivar dec by f.o. 4

    assert type(wid) is int, 'window width must be integer'
    if wid % 2 == 0:
        wid += 1

    spec_snr = np.abs(a) * np.sqrt(ivar)

    pad = wid // 2
    # pad `a` & `ivar` at the ends
    pad_width = [[pad, pad], [0, 0], [0, 0]]
    a_, ivar_ = (np.pad(a, pad_width=pad_width, mode='median',
                        stat_length=pad_width),
                 np.pad(ivar, pad_width=pad_width, mode='median',
                        stat_length=pad_width))

    # now set up rolling median filter
    def rolling_window(a, wid):
        shape = (wid, a.shape[0] - wid + 1) + a.shape[1:]
        strides = (a.strides[0], ) + a.strides
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    # find outlier pixels
    med_a_ = np.median(rolling_window(a_, wid), axis=0)
    a_wid_ = 0.5 * np.diff(
        np.percentile(rolling_window(a_, wid), q=[16., 84.], axis=0), axis=0)
    outside_nominal_range = (np.abs(a - med_a_) > 2. * a_wid_).squeeze()
    med_ivar_ = np.median(rolling_window(ivar_, wid), axis=0)
    med_snr_ = np.median(np.abs(rolling_window(a_ * np.sqrt(ivar_), wid)), axis=0)

    low_snr = spec_snr < (snr_mult_thresh * med_snr_)
    high_snr = spec_snr > (1. / snr_mult_thresh * med_snr_)

    # replace anomalously low or high snr values with local median
    baddata = np.logical_or.reduce((low_snr, high_snr, outside_nominal_range))

    return baddata

def combine_masks(shape, mask_spax=None, mask_spec=None, mask_cube=None):
    isgood = np.ones(shape, dtype=bool)
    if mask_spax is not None:
        isgood *= (~mask_spax)
    if mask_spec is not None:
        isgood *= (~mask_spec[:, None, None])
    if mask_cube is not None:
        isgood *= (~mask_cube)

    return ~isgood

class MaNGA_LSF(object):
    '''
    instrumental line-spread function of MaNGA
    '''
    def __init__(self, LSF_R_obs_gpr, **kwargs):
        self.LSF_R_obs_gpr = LSF_R_obs_gpr

    def LSF_pix_z(self, lam, dlogl, z):
        '''
        calculate width (pix) of LSF
        '''
        specres = self.LSF_R_obs_gpr.predict(
            np.atleast_2d(lam).T)
        dlnl = dlogl * ln10

        wpix = (1. / dlnl) * (1. / specres)
        wpix_z = wpix / (1. + z)

        return wpix_z

    @classmethod
    def from_drpall(cls, drpall, n=None, **kwargs):
        '''
        read in lots of IFUs' LSFs, assume a redshift
        '''
        import sklearn.gaussian_process as gp

        if n is None:
            n = len(drpall)

        lam, specres, dspecres = zip(
            *[m.hdu_data_extract(
                  hdulist=m.load_drp_logcube(
                      plate=row['plate'], ifu=row['ifudsgn'], mpl_v=mpl_v),
                  names=['WAVE', 'SPECRES', 'SPECRESD'])
              for row in drpall[:n]])

        lam = np.concatenate(lam)
        specres = np.concatenate(specres)
        dspecres = np.concatenate(dspecres)
        good = np.logical_and.reduce(
            list(map(np.isfinite, [lam, specres, dspecres])))
        lam, specres, dspecres = lam[good], specres[good], dspecres[good]

        kernel_ = gp.kernels.RBF(
                      length_scale=1., length_scale_bounds=(.2, 5.)) + \
                  gp.kernels.WhiteKernel(
                      noise_level=.02, noise_level_bounds=(.002, 2.))
        regressor = gp.GaussianProcessRegressor(
            normalize_y=True, kernel=kernel_)
        regressor.fit(X=np.atleast_2d(lam).T, y=specres)

        return cls(LSF_R_obs_gpr=regressor, **kwargs)

    def __call__(self, lam, dlogl, y, z):
        '''
        performs convolution with LSF appropriate to a given redshift
        '''
        wpix_z = self.LSF_pix_z(lam, dlogl, z)
        yfilt = np.row_stack(
            [gaussian_filter(spec=s, sig=wpix_z) for s in y])
        return yfilt

class SpecScaler(object):
    '''
    scale spectra to unit dispersion
    '''
    def __init__(self, X, pctls=(16., 84.)):
        '''
        params:
         - X (nspec, nl): array of spectra
        '''
        # first scale each spectrum (row) s.t. distance between
        # pctls[0] and pctls[1] is unity
        self.pctls = pctls
        pctls_v = np.percentile(X, pctls, axis=1)
        self.X_sc = X / np.diff(pctls_v, n=1, axis=0).squeeze()[:, None]

    def __call__(self, Y, lam_axis=0, map_axis=(1, 2)):
        '''
        apply the same scaling as is fit
        '''

        # first, scale to unit dispersion
        pctls_v = np.percentile(Y, self.pctls, axis=lam_axis)
        a = np.diff(pctls_v, n=1, axis=0).squeeze()
        Y_sc = Y / a[None, ...]

        return Y_sc, a

class MedianSpecScaler(object):
    '''
    scale spectra to unit median
    '''
    def __init__(self, X):
        '''
        params:
         - X (nspec, nl): array of spectra
        '''

        med = np.median(X, axis=1, keepdims=True)
        self.X_sc = X / med

    def __call__(self, Y, lam_axis=0, map_axis=(1, 2)):
        '''
        apply the same scaling as is fit
        '''

        med = np.median(Y, axis=lam_axis, keepdims=True)
        Y_sc = Y / med

        return Y_sc, med.squeeze()

class SqFromSqCacher(object):
    '''
    takes square subarray, caching results
    '''
    def __init__(self, large_array, n):
        self.large_array = large_array
        self.n = n

    @lru_cache(maxsize=128)
    def take(self, i0):
        return self.large_array[i0:i0 + self.n, i0:i0 + self.n]

class KPCGen(object):
    '''
    compute some spaxel's PC cov matrix
    '''

    def __init__(self, kspec_obs, i0_map, E, ivar_scaled):
        self.kspec_obs = kspec_obs
        self.i0_map = i0_map
        self.E = E
        self.q, self.nl = E.shape
        self.ivar_scaled = ivar_scaled

        self.sqfromsq = SqFromSqCacher(kspec_obs, self.nl)

    def __call__(self, i, j):
        i0_ = self.i0_map[i, j]
        kspec = self.sqfromsq.take(i0_)

        return (self.E @ (kspec) @ self.E.T)


def interp_large(x0, y0, xnew, axis, nchunks=1, **kwargs):
    '''
    large-array-tolerant interpolation
    '''

    success = False

    specs_interp = interp1d(x=x0, y=y0, axis=axis, **kwargs)

    while not success:
        # chunkify x array
        xchunks = np.array_split(xnew, nchunks)
        try:
            ynew = np.concatenate(
                [specs_interp(xc) for xc in xchunks], axis=axis)
        except MemoryError:
            nchunks += 1
        else:
            success = True

    return ynew


class FilterFuncs(object):
    '''
    turn a list of TableGroup-filtering functions into a single function
    '''
    def __init__(self, funcs):
        self.funcs = funcs

    def __call__(self, tab, *args):
        if (self.funcs is None):
            return True

        if len(self.funcs) == 0:
            return True

        for f in self.funcs:
            if f(tab, *args) is False:
                return False
        else:
            return True

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

    # point-slope method
    d1 = r1[1] - r1[0]
    d2 = r2[1] - r2[0]
    px, py = r1[0], r2[0]
    m = d2 / d1

    return (x - px) * m + py

def determine_dlogl(logl):
    dlogl = np.round(np.mean(logl[1:] - logl[:-1]), 8)
    return dlogl

def determine_dl(logl):
    dlogl = determine_dlogl(logl)
    logl_lbd = logl - dlogl / 2
    logl_ubd = logl + dlogl / 2
    l_lbd = 10.**logl_lbd
    l_ubd = 10.**logl_ubd

    dl = l_ubd - l_lbd

    return dl

def determine_dloglcube(logl):
    # make array of boundaries btwn logl pixels
    logl_midbds = 0.5 * (logl[:-1] + logl[1:])
    # this omits endpoints, so just assume the same dlogl between
    # 0 & 1 as 1 & 2, and the same between -1 & -2 as -2 & -3
    dlogl_ = logl_midbds[1:] - logl_midbds[:-1]
    dlogl = np.pad(
        dlogl_, pad_width=((1, 1), (0, 0), (0, 0)), mode='edge')
    return dlogl

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

def blur_cube_to_psf(l_ref, specres_ref, l_eval, spec_unblurred):
    '''
    blur a flux-density cube using a psf cube
    '''
    specres_obs = interp1d(
        x=l_ref, y=specres_ref,
        bounds_error=False, fill_value='extrapolate')(l_eval)
    cubeshape = l_eval.shape
    mapshape = cubeshape[1:]
    # convert specres of observations into dlnl
    dlnl_obs = 1. / specres_obs

    # dlogl of a pixel in model
    dloglcube_model = determine_dloglcube(np.log10(l_eval))
    # convert dlogl of pixel in model to dlnl
    dlnlcube_model = dloglcube_model * np.log(10.)
    # number of pixels is dlnl of obs div by dlnl of model
    specres_pix = dlnl_obs / dlnlcube_model

    # create placeholder for instrumental-blurred model
    spec_model_instblur = np.empty_like(specres_obs)

    # populate pixel-by-pixel (should take < 15 sec)
    for ind in np.ndindex(mapshape):
        spec_model_instblur[:, ind[0], ind[1]] = gaussian_filter(
            spec=spec_unblurred[:, ind[0], ind[1]],
            sig=specres_pix[:, ind[0], ind[1]])

    return spec_model_instblur

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

def extinction_atten(l, f, EBV, r_v=3.1, ivar=None, **kwargs):
    '''
    wraps around specutils.extinction.reddening
    '''

    a_v = r_v * EBV
    r = extinction.fitzpatrick99

    # output from reddening is inverse-flux-transmission
    f_itr = 2.5**r(wave=l, a_v=a_v, r_v=r_v, **kwargs)
    # to deredden, divide f by f_itr

    f_atten = f / f_itr

    if ivar is not None:
        ivar_atten = ivar * f_itr**2.
        return f_atten, ivar_atten
    else:
        return f_atten

def extinction_correct(l, f, EBV, r_v=3.1, ivar=None, **kwargs):
    '''
    wraps around specutils.extinction.reddening
    '''

    a_v = r_v * EBV
    r = extinction.fitzpatrick99

    # output from reddening is inverse-flux-transmission
    f_itr = 2.5**r(wave=l, a_v=a_v, r_v=r_v, **kwargs)[:, None, None]
    # to redden, mult f by f_itr

    f_att = f * f_itr

    if ivar is not None:
        ivar_att = ivar / f_itr**2.
        return f_att, ivar_att
    else:
        return f

def add_redshifts(zs, axis=0):
    '''
    add redshifts in an array-like, along an axis
    '''

    z_tot = np.prod((1. + zs), axis=axis) - 1.
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

def coadd(f, ivar):

    fnew = f.sum(axis=(1, 2))
    ivarnew = ivar.sum(axis=(1, 2))

    return fnew, ivarnew

def PC_cov(cov, snr, i0, E, nl, q):
    if snr < 1.:
        return 100. * np.ones((q, q))
    else:
        sl = [slice(i0, i0 + nl) for _ in range(2)]
        return E @ (cov[i0 : i0 + nl, i0 : i0 + nl]) @ E.T

def reshape_cube2rss(cubes):
    '''
    reshape datacube into pseudo-RSS (spaxel-wise)
    '''

    naxis1, naxis2, naxis3 = cubes[0].shape

    # reshape data
    # axis 0: observation; axis 1: spaxel; axis 2: wavelength

    rss = np.stack([cube.reshape((naxis1, -1)).T for cube in cubes])
    return rss

def apply_mask(A, good, axis=0):
    '''
    apply spaxel selection along axis
    '''

    if type(axis) is np.ndarray:
        A = (A, )

    if type(axis) is int:
        axis = (axis, ) * len(A)

    A_new = (np.compress(a=A_, condition=good, axis=ax) for A_, ax in zip(A, axis))

    return A_new

def bin_sum_agg(A, bins):
    '''
    sum-aggregate array A along bin numbers given in `bins`
    '''
    mask = np.zeros((bins.max()+1, len(bins)), dtype=bool)
    mask[bins, np.arange(len(bins))] = 1

    return mask.dot(A)

def random_cov_matrix(ndim):
    k0 = np.random.randn(ndim, ndim)
    K = k0 @ k0.T
    return K

def random_orthogonal_basis(shape):
    nsamp, ndim = shape
    K = random_cov_matrix(ndim)
    evals, evecs = np.linalg.eig(K)
    order = np.argsort(evals)[::-1][:nsamp]
    return evecs[:, order].T

def brokenstick(N, n):
    '''
    return broken-stick model for N, s.t.
        E_n = (1/n + 1/(n+1) + ... + 1/N) / N
    '''
    ns = np.linspace(n, N, N - n + 1)
    inv_ns = 1. / ns
    return inv_ns.sum() / N

def ba_to_i_holmberg(ba, alpha=0.2):
    '''
    return inclination angle from axis ratio
    '''
    cos2i = (ba**2. - alpha**2.) / (1. - alpha**2.)
    cosi = np.sqrt(cos2i)
    i = np.arccos(cosi)
    return i


class LogcubeDimError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__()

hdu_unit = lambda hdu: u.Unit(hdu.header['BUNIT'])
