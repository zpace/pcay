import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

from astropy import units as u, table as t  # , constants as c
from astropy.io import fits

from scipy.signal import medfilt
from scipy.linalg import pinv2
import linalg
import sklearn.covariance

import os, sys, io
from itertools import takewhile, combinations as comb

from glob import glob

from importer import *

import manga_tools as m
from spec_tools import air2vac, vac2air

if mangarc.voronoi_loc not in sys.path:
    sys.path.append(mangarc.voronoi_loc)

import voronoi_2d_binning as voronoi2d

import utils as ut
from partition import CovWindows

from contextlib import redirect_stdout
from functools import lru_cache

mpl_v = 'MPL-6'
eps = np.finfo(float).eps

# =====

print('MaNGA data-product info:', mpl_v, '({})'.format(m.MPL_versions[mpl_v]))
print('MaNGA data location:', mangarc.manga_data_loc[mpl_v])


class Cov_Obs(object):
    '''
    a class to precompute observational spectral covariance matrices
    '''

    def __init__(self, cov, lllim, dlogl, nobj):
        self.cov = cov # enforce_posdef(cov)
        self.nspec = len(cov)
        self.lllim = lllim
        self.loglllim = np.log10(self.lllim)
        self.dlogl = dlogl
        self.nobj = nobj

        self.precision, self.cov_rank = pinv2(self.cov, return_rank=True, rcond=1.0e-3)

    # =====
    # classmethods
    # =====

    @classmethod
    def from_BOSS_reobs(cls, spAll, dlogl=1.0e-4, n=None, platelim=None,
                        corr_zpt=False):
        '''
        returns a covariance object made from an spAll file
        '''

        # prepare catalog
        colnames = ['PLATE', 'MJD', 'FIBERID', 'OBJID']
        cols = [spAll[1].data[k] for k in colnames]
        obs = t.Table(cols, names=colnames)
        obs = obs[obs['OBJID'] != ' ' * 19]

        if platelim is not None:
            obs = obs[obs['PLATE'] < platelim]

        objs = obs.group_by('OBJID')

        def morethan1_(tab):
            return len(tab) > 1

        morethan1 =  np.array(list(map(morethan1_, objs.groups)))

        # grouped filter & final grouping
        objs_dupl = (objs.groups[morethan1])
        objs_dupl.write('BOSS_dupl.fits', format='fits', overwrite=True)
        objs_dupl.group_by('OBJID')
        objids = objs_dupl.groups.keys
        n_unique = len(objids)

        # if unspecified, use all galaxies
        if n is None:
            n = n_unique

        groups = [tab for tab in objs_dupl.groups]
        use = [True, ] * n + [False, ] * (n_unique - n)
        groups = [g for g, u in zip(groups, use) if u]

        # multiply-observed objects
        logls, diffs, ws = zip(
            *[Cov_Obs.process_BOSS_mult(grp) for grp in groups])

        good = lambda x: type(x) is np.ndarray
        # test whether elements returned are of allowed type
        # filter out otherwise
        logls = list(takewhile(good, logls))
        diffs = list(takewhile(good, diffs))
        ws = list(takewhile(good, ws))

        ndiffs = sum(list(map(len, diffs)))
        logl0 = min(list(map(min, logls)))
        loglf = max(list(map(max, logls)))

        # full wavelength grid for all objects
        logl_full = np.arange(logl0, loglf + dlogl, step=dlogl)

        # place differences and weights on common grid
        diffs_regr, ws_regr = zip(
            *[put_on_grid(logl_full, logl, A=(d, w))
              for logl, d, w in zip(logls, diffs, ws)])

        diffs_regr = np.row_stack(diffs_regr)
        ws_regr = np.row_stack(ws_regr)

        # subtract the mean of the difference, to roughly account for flat
        # differences in absolute flux calibration
        if corr_zpt:
            diffs_regr -= np.average(
                diffs_regr, weights=ws_regr, axis=1)[:, None]

        # compute covariance
        cov = wcov(diffs_regr, ws_regr)
        cov /= np.median(np.diag(cov))

        lllim = 10.**(logl_full.min())

        return cls(cov, lllim=lllim, dlogl=dlogl, nobj=n)

    @classmethod
    def from_MaNGA_reobs(cls, dlogl=1.0e-4, MPL_v=mpl_v, n=None):
        '''
        returns a covariance object made from reobserved MaNGA IFU LOGCUBEs
        '''

        drpall = m.load_drpall(mpl_v)

        drpall = drpall[(drpall['ifudesignsize'] != -9999) * \
                        (drpall['nsa_z'] != -9999)]

        # only use objects with multiple observations with same IFU size
        def onesize_(tab, *args):
            return len(set(tab['ifudesignsize'])) == 1

        def oneifudsgn_(tab, *args):
            return len(set(tab['ifudsgn'])) == 1

        groups = find_mult_obs(drpall, groupby_key='mangaid',
                               filter_funcs=[onesize_], minlen=2, n=n)

        # figure out lam low lim and nlam
        drp_ = m.load_drp_logcube('8083', '12704', MPL_v)
        lam = drp_['WAVE'].data
        drp_.close()
        lllim = lam.min()
        nl = len(lam)

        # process multiply-observed datacubes, up to limit set
        diffs, diffs_ivars = map(
            np.row_stack, zip(*[difference_reobs(
                                   grp, 'MPL-6', nspec=nl, SN_thresh=3.,
                                   f_mask_thresh=.01, voronoi=50)
                               for grp in groups]))

        cov = calc_cov_einsum(diffs)
        N = diffs.shape[0]

        return cls(cov, lllim=lllim, dlogl=dlogl, nobj=N)

    @classmethod
    def from_fits(cls, fname):
        hdulist = fits.open(fname)
        cov = hdulist[1].data
        h = hdulist[1].header
        lllim = 10.**h['LOGL0']
        dlogl = h['DLOGL']
        nobj = h['NOBJ']
        return cls(cov=cov, lllim=lllim, dlogl=dlogl, nobj=nobj)

    @classmethod
    def from_tremonti(cls, fname, *args, **kwargs):
        '''
        Christy's covariance calculations
        '''
        cov_super = fits.getdata(fname, ext=1)
        wave = cov_super['WAVE'][0]
        cov = cov_super['COV_MATRIX'][0]
        nobj = 0
        dlogl = ut.determine_dlogl(np.log10(wave))
        lllim = wave[0]
        return cls(cov=cov, lllim=lllim, dlogl=dlogl, nobj=nobj, *args, **kwargs)

    @classmethod
    def from_YMC_BOSS(cls, fname, logl0=3.5524001):
        hdulist = fits.open(fname)
        cov = hdulist[1].data
        h = hdulist[1].header
        lllim = 10.**logl0
        dlogl = 1.0e-4
        nobj = 48000
        return cls(cov=cov, lllim=lllim, dlogl=dlogl, nobj=nobj)

    # =====
    # methods
    # =====

    def _init_windows(self, w):
        self.windows = diag_windows(self.cov, w)

    @lru_cache(maxsize=256)
    def take(self, i0):
        return self.windows[i0]

    def precompute_Kpcs(self, E):
        '''
        precompute PC covs, based on given eigenvectors (projection matrix)
        '''

        ETE = E.T @ E
        inv_ETE = linalg.spla_chol_invert(
            ETE + np.diag(np.diag(ETE)), np.eye(*ETE.shape))
        H = inv_ETE @ E.T
        self.covwindows = CovWindows(self.cov, H.T)

    def write_fits(self, fname='cov.fits'):
        hdu_ = fits.PrimaryHDU()
        hdu = fits.ImageHDU(data=self.cov)
        hdu.header['LOGL0'] = np.log10(self.lllim)
        hdu.header['DLOGL'] = self.dlogl
        hdu.header['NOBJ'] = self.nobj
        hdulist = fits.HDUList([hdu_, hdu])
        hdulist.writeto(fname, overwrite=True)

    def make_im(self, kind, max_disp=0.4, llims=None):
        l = self.l
        fig = plt.figure(figsize=(18, 17), dpi=400)
        gs = GridSpec(1, 2, width_ratios=[36, 1])
        ax = plt.subplot(gs[0])
        cax_ = plt.subplot(gs[1])
        ax.tick_params(axis='both', bottom=True, top=True, left=True, right=True,
                       labelbottom=True, labeltop=True, labelleft=True,
                       labelright=False, labelsize=6)

        if llims is not None:
            ax.set_xlim(llims)
            ax.set_ylim(llims)

        vmax = np.abs(self.cov).max()**0.3
        extend = 'neither'
        if vmax > max_disp:
            vmax = max_disp
            extend = 'both'

        im = ax.imshow(
            np.sign(self.cov) * (np.abs(self.cov))**0.3,
            extent=[l.min(), l.max(), l.min(), l.max()], cmap='coolwarm',
            vmax=vmax, vmin=-vmax, interpolation='nearest', aspect='equal')
        xticker = mticker.MaxNLocator(nbins=15, steps=[1, 5, 10], integer=True)
        yticker = mticker.MaxNLocator(nbins=15, steps=[1, 5, 10], integer=True)
        ax.xaxis.set_major_locator(xticker)
        ax.yaxis.set_major_locator(yticker)
        ax.set_xlabel(r'$\lambda  ~ [{\rm \AA}]$', size=6)
        ax.set_ylabel(r'$\lambda ~ [{\rm \AA}]$', size=6)
        cb = plt.colorbar(im, cax=cax_, extend=extend)
        cb.set_label(r'$\textrm{sign}(K) ~ |K|^{0.3}$', size=6)
        cb.ax.tick_params(labelsize=6)
        plt.tight_layout()

        fig.savefig('cov_obs_{}.png'.format(kind), dpi=200)

    # =====
    # properties
    # =====

    @property
    def logl(self):
        return self.loglllim + np.linspace(
            0., self.dlogl * self.nspec, self.nspec)

    @property
    def l(self):
        return 10.**self.logl

    # =====
    # staticmethods
    # =====

    @staticmethod
    def process_BOSS_mult(tab, dlogl=1.0e-4):
        '''
        return differences between reobservations of same BOSS object
        '''

        # load all the files
        baseloc = mangarc.BOSS_lite_data_loc
        hdulists = load_BOSS_obj(tab, baseloc)
        nobs = len(hdulists)

        if nobs < 2:
            # file DNE errors in load_BOSS_obs may result in
            # length-zero, -one, or -two lists.
            # in that case, return a tuple of three Falses
            return False, False, False

        loglams = [h[1].data['loglam'] for h in hdulists]
        fluxs = [h[1].data['flux'] for h in hdulists]
        ivars = [h[1].data['ivar'] for h in hdulists]
        masks = [(h[1].data['and_mask'] > 0) for h in hdulists]

        # construct full wavelength grid
        logl0 = min(list(map(min, loglams)))
        loglf = max(list(map(max, loglams)))
        logl_full = np.arange(logl0, loglf + dlogl, step=dlogl)
        nl = len(logl_full)
        i0 = list(map(lambda x: np.argmin(np.abs(x[0] - logl_full)), loglams))

        ivars_regr = np.zeros((nobs, nl))
        fluxs_regr = np.zeros((nobs, nl))
        masks_regr = np.zeros((nobs, nl), dtype=bool)

        # put all objects onto same grid
        for i, i0_ in enumerate(i0):
            nl_ = len(fluxs[i])
            fluxs_regr[i, i0_:i0_ + nl_] = fluxs[i]
            ivars_regr[i, i0_:i0_ + nl_] = ivars[i]
            masks_regr[i, i0_:i0_ + nl_] = masks[i]

        # propagate masks into ivars
        ivars_regr *= ~masks_regr

        # combine fluxs into diffs
        pairs = comb(range(nobs), 2)
        diffs = np.row_stack([fluxs_regr[i1, ...] - fluxs_regr[i2, ...]
                              for (i1, i2) in pairs])

        # combine ivars into weights
        pairs = comb(range(nobs), 2)
        ws = 1. / np.row_stack(
            [(1. / (1. / ivars_regr[i1, ...]) + (1. / ivars_regr[i2, ...]))
             for i1, i2 in pairs])
        ws[ws == 0.] = eps

        return logl_full, diffs, ws


class ShrunkenCov(Cov_Obs):
    '''
    shrunken covariance matrix
    '''
    def __init__(self, cov, lllim, dlogl, nobj, shrinkage=0.):
        shrunken_cov = sklearn.covariance.shrunk_covariance(
            emp_cov=cov, shrinkage=shrinkage)
        super().__init__(shrunken_cov, lllim, dlogl, nobj)

def enforce_posdef(a, replace_val=1.0e-6):
    '''
    enforce positive-definiteness: calculate the nearest
        (in frobenius-norm sense) positive-definite matrix to
        supplied (symmetric) matrix `a`
    '''
    # eigen-decompose `a`
    evals, evecs = np.linalg.eig(a)

    # set all eigenvalues <= 0 to floating-point epsilon
    evals[evals <= 0] = replace_val

    # recompose approximation of original matrix
    a_new = evecs @ np.diag(evals) @ np.linalg.inv(evecs)

    return a_new

def diag_windows(x, n):
    from numpy.lib.stride_tricks import as_strided
    if x.ndim != 2 or x.shape[0] != x.shape[1] or x.shape[0] < n:
        raise ValueError("Invalid input")
    w = as_strided(x, shape=(x.shape[0] - n + 1, n, n),
                   strides=(x.strides[0]+x.strides[1], x.strides[0], x.strides[1]))
    return w

def find_mult_obs(tab, groupby_key, filter_funcs=[], minlen=2, n=None,
                  keep_cols='all'):
    '''
    from a table of observational data, find duplicate observations

    params:
     - tab: table of observations
     - groupby_key: column name to group by
     - filter_funcs: list of functions used to filter rows of tab
     - minlen: minimum number of observations for a given object
     - n: number of groups to consider
     - keep_cols: which columns to keep (preserves memory) (default: all)

    returns:
     - groups: a list of tables, each of which is a unique object
    '''

    objs = tab.group_by(groupby_key)

    # restrict length to GE minlen
    len_filter = lambda tab, *args: True if len(tab) >= minlen else False

    # aggregate filtering
    agg_filter = ut.FilterFuncs(filter_funcs + [len_filter, ])
    objs_dupl = objs.groups.filter(agg_filter)

    # re-group by previous key
    objs_dupl.group_by(groupby_key)
    ids = objs_dupl.groups.keys
    n_unique = len(ids)

    # throw away unnecessary keys
    if not isinstance(keep_cols, str):
        if groupby_key not in keep_cols:
            keep_cols += groupby_key
        objs_dupl.keep_columns(keep_cols)
    elif keep_cols.lower() == 'all':
        pass
    else:
        pass

    # if unspecified, use all galaxies
    if n is None:
        n = n_unique

    groups = [tab for i, tab in enumerate(objs_dupl.groups) if i < n]

    return groups

def load_MaNGA_mult(tab, MPL_v, nspec, maskbits=[0, 1, 2, 3, 10]):
    '''
    do initial read-in of MaNGA reobservations
    '''
    logcubes = [m.load_drp_logcube(str(p), i, MPL_v)
                for p, i in zip(tab['plate'], tab['ifudsgn'])]

    # handle the case where LOGCUBEs have different shapes
    if len(set(map(lambda h: h['FLUX'].data.shape, logcubes))) > 1:
        raise ut.LogcubeDimError('LOGCUBE dimensions do not match')

    # extract & reshape data
    # axis 0: observation; axis 1: spaxel; axis 2: wavelength
    fluxs = ut.reshape_cube2rss([cube['FLUX'].data for cube in logcubes])
    ivars = ut.reshape_cube2rss([cube['IVAR'].data for cube in logcubes])
    masks = ut.reshape_cube2rss([m.mask_from_maskbits(a=cube['MASK'].data,
                                                      b=maskbits)
                                 for cube in logcubes])
    RIMGs = np.stack([cube['RIMG'].data.flatten() for cube in logcubes])
    mapshape = logcubes[0]['RIMG'].data.shape

    for lc in logcubes:
        lc.close()

    return fluxs, ivars, masks, RIMGs, mapshape

def filter_spaxels_(fluxs, ivars, masks, RIMGs, mapshape,
                     SN_thresh=3., f_mask_thresh=.05, return_all=False):
    '''
    take composed array of fluxs, ivars, and masks, and return

    '''
    # where is there no signal at all?
    XX, YY = np.indices(mapshape)
    XX, YY = XX.flatten(), YY.flatten()
    observed = np.all((RIMGs != 0), axis=0)

    # all spectra along LOS must have med SNR >= thresh
    goodSN = np.all(
        np.median(fluxs * np.sqrt(ivars), axis=2) >= SN_thresh,
        axis=0)

    toomanymasked = np.any(
        np.mean(masks.astype(float), axis=2) >= f_mask_thresh, axis=0)

    goodspax = np.logical_and.reduce((observed, goodSN, ~toomanymasked))

    # select spaxels with enough signal
    fluxs, ivars, masks, RIMGs, XX, YY = ut.apply_mask(
        A=(fluxs, ivars, masks, RIMGs, XX, YY), good=goodspax,
        axis=(1, 1, 1, 1, 0, 0))

    if return_all:
        return fluxs, ivars, masks, RIMGs, mapshape, (XX, YY)
    else:
        return fluxs, ivars, masks

def apply_op_(op, fluxs, n_reobs):
    pairs = comb(range(n_reobs), 2)
    res = np.row_stack([op(fluxs[i1, ...], fluxs[i2, ...]) for (i1, i2) in pairs])
    return res

def difference_reobs(tab, MPL_v, nspec=4563, SN_thresh=3.,
                       f_mask_thresh=.05, voronoi=50):
    '''
    construct pairs of resobservations:
    1. load all cubes for a MaNGA-ID, confirm same shape
    2. choose spaxels to use (exlude low median S/N and spectra with lots of masks)
    3. replace bad values with median of neighbors
    4. normalize each spectrum to weighted average flux of 1
    5. voronoi-bin spectra to specified S/N: sum fluxes, invert sum of inverse-ivars
    6. difference pairs of identical reobservations:
        F = f1 - f0; I = 1 / (1 / i0 + 1 / i1)
    7. reshape F, I like a list of spectra (npairs, nlam)
    '''

    try:
        mults = load_MaNGA_mult(tab, MPL_v, nspec)
    except ut.LogcubeDimError:
        dummy = np.empty((0, nspec))
        return dummy, dummy
    else:
        fluxs, ivars, masks, RIMGs, mapshape = mults

    # return only good spaxels
    fluxs, ivars, masks, RIMGs, mapshape, (XX, YY) = filter_spaxels_(
        fluxs, ivars, masks, RIMGs, mapshape, return_all=True,
        f_mask_thresh=f_mask_thresh, SN_thresh=SN_thresh)

    # bad pixels get replaced by width-101
    # rolling median of their spectral neighbors
    goodpix = ((~masks) * (fluxs * np.sqrt(ivars) >= SN_thresh))
    roll_med = medfilt(fluxs, [1, 1, 101])
    fluxs[~goodpix] = roll_med[~goodpix]
    n_reobs, n_spax, _ = fluxs.shape

    # normalize individual spectra
    a = np.average(fluxs, weights=ivars, axis=2)[:, :, None]
    fluxs /= a
    ivars *= a**2.

    # voronoi bin
    if voronoi > 0:
        binnum = voronoi_bin_multiple(
            XX, YY, fluxs, ivars, targetSN=voronoi, quiet=True, plot=False)
        binfluxs = np.stack(
            [ut.bin_sum_agg(A=s, bins=binnum) for s in fluxs])
        binivars = 1. / np.stack(
            [ut.bin_sum_agg(A=(1./(s + 10. * eps)), bins=binnum) for s in ivars])
        bincounts, _ = np.histogram(
            binnum, bins=np.linspace(-.5, .5 + binnum.max(),
                                     binnum.max() + 1))
    else:
        binfluxs = fluxs
        binivars = ivars
        bincounts = np.ones(fluxs.shape[1])

    a_bin = np.average(binfluxs, weights=binivars, axis=2)[:, :, None]
    binfluxs /= a_bin
    binivars *= a_bin**2.

    # find differences
    diffs = apply_op_(np.subtract, binfluxs, n_reobs)

    # function aggregates ivars
    def ivar_agg(ivar1, ivar2):
        var1, var2 = 1. / ivar1, 1. / ivar2
        varsum = var1 + var2
        ivar_agg = 1. / varsum
        ivar_agg[~np.isfinite(ivar_agg)] = 0.
        return ivar_agg

    # find ivars of differences
    diffs_ivars = apply_op_(ivar_agg, binivars, n_reobs)

    return diffs, diffs_ivars

def voronoi_bin_multiple(XX, YY, fluxs, ivars, targetSN, quiet=False, **kwargs):
    '''
    apply Cappellari's voronoi binning method, `voronoi` gives desired SN
    '''
    # choose lower-S/N galaxy to use as basis for binning
    sn_i = np.argmin(np.median(fluxs * np.sqrt(ivars), axis=(1, 2)))

    # use median flux & ivar to define signal & noise
    S = np.median(fluxs[sn_i, ...], axis=-1)
    IV = ivars[sn_i, ...]
    N_ = 1. / np.sqrt(IV)
    N_[IV <= 1.0e-9] = 100. * fluxs[sn_i][IV <= 1.0e-9]
    N = np.median(N_, axis=-1)

    if quiet:
        f = io.StringIO()
    else:
        f = sys.stdout

    with redirect_stdout(f):
        voronoi_res = voronoi2d.voronoi_2d_binning(
            x=XX, y=YY, signal=S, noise=N, targetSN=targetSN, **kwargs)

    binnum, *_, XBAR, YBAR, SN_final, npix_final, scale = voronoi_res

    return binnum

def load_BOSS_obs(row, baseloc):
    plate, mjd, fiberid = row['PLATE'], row['MJD'], row['FIBERID']
    fname = '{0}/spec-{0}-{1}-{2:04}.fits'.format(plate, mjd, fiberid)
    fname_full = os.path.join(baseloc, fname)
    try:
        hdulist = fits.open(fname_full)
    except IOError:
        # if file DNE, return False
        hdulist = False

    return hdulist

def load_BOSS_obj(tab, baseloc):
    obss = [load_BOSS_obs(row, baseloc) for row in tab]
    # look for file DNE result, and filter out
    obss = [obs for obs in obss if obs != False]
    return obss

def gen_BOSS_fname(row):
    plate, mjd, fiberid = row['PLATE'], row['MJD'], row['FIBERID']
    return '{0}/spec-{0}-{1}-{2:04}.fits'.format(plate, mjd, fiberid)

def put_on_grid(logl_full, logl_sub, A):
    '''
    use padding to place several arrays (elements of tuple A)
        onto uniform grid
    '''

    logl0 = logl_sub[0]
    ix0 = np.argmin(np.abs(logl_full - logl0))
    lpad = ix0
    rpad = len(logl_full) - (ix0 + len(logl_sub))
    A_new = tuple(np.pad(
                      A_, pad_width=[[0, 0], [lpad, rpad]], mode='constant',
                      constant_values=[[0., 0.], [0., 0.]]) for A_ in A)
    return A_new

def wcov(diffs, ws):
    wdiffs = (ws * diffs)
    cov = wdiffs.T.dot(wdiffs) / (2. * ws.T.dot(ws))
    return cov

def calc_cov_wtd_ctrd(diffs, ivars=None):
    if ivars is None:
        ivars = np.ones_like(diffs)
    wtd_diff = diffs * ivars
    cov = (wtd_diff.T @ wtd_diff) / (2. * ivars.T @ ivars - 1.)
    return cov

def calc_cov_einsum(diffs, ivars=None):
    N, nl = diffs.shape
    cov = (2. * N)**-1. * np.einsum('na,nb->ab', diffs, diffs)
    return cov

def display_cov(cov, dv):
    plt.imshow(np.sign(cov) * np.abs(cov)**.3, cmap='coolwarm', vmin=-dv, vmax=dv)
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    '''
    spAll_loc = os.path.join(mangarc.zpace_sdss_data_loc,
                             'boss', 'spAll-v5_9_0.fits')
    spAll = fits.open(spAll_loc, memmap=True)
    Cov_boss = Cov_Obs.from_BOSS_reobs(spAll=spAll, n=None, platelim=6700,
                                       corr_zpt=True)
    Cov_boss.write_fits('boss_Kspec.fits')
    Cov_boss.make_im(kind='boss')
    '''
    # =====
    #'''
    Cov_manga = Cov_Obs.from_MaNGA_reobs()
    Cov_manga.make_im(kind='manga', max_disp=0.4)
    Cov_manga.write_fits('manga_Kspec_new.fits')
    #'''
