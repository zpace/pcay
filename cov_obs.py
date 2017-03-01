import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy import units as u, table as t  # , constants as c
from astropy.io import fits

import os
import sys
from itertools import takewhile, combinations as comb

from glob import glob

# add manga RC location to path, and import config
if os.environ['MANGA_CONFIG_LOC'] not in sys.path:
    sys.path.append(os.environ['MANGA_CONFIG_LOC'])

import mangarc

if mangarc.tools_loc not in sys.path:
    sys.path.append(mangarc.tools_loc)

import manga_tools as m

mpl_v = 'MPL-5'
eps = np.finfo(float).eps

# =====

print('MaNGA data-product info:', mpl_v, '({})'.format(m.MPL_versions[mpl_v]))
print('MaNGA data location:', mangarc.manga_data_loc[mpl_v])


class Cov_Obs(object):
    '''
    a class to precompute observational spectral covariance matrices
    '''

    def __init__(self, cov, lllim, dlogl, nobj):
        cov /= np.median(np.diag(cov))
        cov[cov > 100.] = 100.
        self.cov = cov
        self.nspec = len(cov)
        self.lllim = lllim
        self.loglllim = np.log10(self.lllim)
        self.dlogl = dlogl
        self.nobj = nobj

    # =====
    # classmethods
    # =====

    @classmethod
    def from_BOSS_reobs(cls, spAll, dlogl=1.0e-4, n=None, platelim=None):
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

        drpall = drpall[drpall['ifudesignsize'] != -9999]
        objs = drpall.group_by('mangaid')

        # only use objects with multiple observations with same IFU size
        def onesize_(tab):
            return len(np.unique(tab['ifudesignsize'])) == 1

        def morethan1_(tab):
            return len(tab) > 1

        onesize = np.array(list(map(onesize_, objs.groups)))
        morethan1 = np.array(list(map(morethan1_, objs.groups)))

        # grouped filter & final grouping
        objs_dupl = (objs.groups[morethan1 & onesize])
        objs_dupl.group_by('mangaid')
        mangaids = objs_dupl.groups.keys
        n_unique = len(mangaids)

        # if unspecified, use all galaxies
        if n is None:
            n = n_unique

        groups = [tab for tab in objs_dupl.groups]
        use = [True, ] * n + [False, ] * (n_unique - n)
        groups = [g for g, u in zip(groups, use) if u]

        # figure out lam low lim and nlam
        lam = m.load_drp_logcube('8083', '12704', MPL_v)['WAVE'].data
        lllim = lam.min()
        nl = len(lam)

        # process multiply-observed datacubes, up to limit set
        diffs, w, good = zip(
            *[Cov_Obs.process_MaNGA_mult(grp, MPL_v, nl)
              for grp in groups])

        # stack all the differences
        diffs = np.row_stack([p for (p, g) in zip(diffs, good) if g])
        w = np.row_stack([p for (p, g) in zip(w, good) if g])

        N = w.shape[0]

        cov = wcov(diffs, w)
        cov /= np.median(np.diag(cov))

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
    def from_YMC_BOSS(cls, fname, logl0=3.5524001):
        hdulist = fits.open(fname)
        cov = hdulist[1].data
        cov /= np.median(np.diag(cov))
        h = hdulist[1].header
        lllim = 10.**logl0
        dlogl = 1.0e-4
        nobj = 48000
        return cls(cov=cov, lllim=lllim, dlogl=dlogl, nobj=nobj)

    # =====
    # methods
    # =====

    def write_fits(self, fname='cov.fits'):
        hdu_ = fits.PrimaryHDU()
        hdu = fits.ImageHDU(data=self.cov)
        hdu.header['LOGL0'] = np.log10(self.lllim)
        hdu.header['DLOGL'] = self.dlogl
        hdu.header['NOBJ'] = self.nobj
        hdulist = fits.HDUList([hdu_, hdu])
        hdulist.writeto(fname, overwrite=True)

    def make_im(self, kind):
        l = self.l
        fig = plt.figure(figsize=(6, 5), dpi=300)
        ax = fig.add_subplot(111)

        vmax = np.abs(self.cov).max()**0.3
        extend = 'neither'
        if vmax > 5.:
            vmax = 5
            extend = 'both'

        im = ax.imshow(
            np.sign(self.cov) * (np.abs(self.cov))**0.3,
            extent=[l.min(), l.max(), l.min(), l.max()], cmap='coolwarm',
            vmax=vmax, vmin=-vmax, interpolation='None', aspect='equal')
        ax.set_xlabel(r'$\lambda  ~[\AA]$', size=10)
        ax.set_ylabel(r'$\lambda ~ [\AA]$', size=10)
        cb = plt.colorbar(im, ax=ax, shrink=0.8, extend=extend)
        cb.set_label(r'$\textrm{sign}(K) ~ |K|^{0.3}$')
        plt.tight_layout()
        plt.savefig('cov_obs_{}.png'.format(kind), dpi=300)

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
    def process_MaNGA_mult(tab, MPL_v, nspec=4563):
        '''
        check for presence of datacubes for a single multiply-observed
            MaNGA object, and (if the data exists) process the observations
            into an array of residuals

        Note: this assumes that all data is present at the correct location.
            Missing data (such that zero or one observations are available)
            will result in an empty array (with correct dimensions)
            being returned

        Returns:
         - diffs: differences between pairs of spectra of
             nominally same location (presumed normed to zero)
         - wt: weights on diffs
         - good: indicates whether resids_normed is good
        '''

        # load all LOGCUBES for a given MaNGA-ID
        # files have form 'manga-<PLATE>-<IFU>-LOGCUBE.fits.gz'
        fnames = [os.path.join(mangarc.manga_data_loc[mpl_v],
                               'drp/', str(plate), 'stack/',
                               '-'.join(('manga', str(plate), str(ifu),
                                         'LOGCUBE.fits.gz')))
                  for plate, ifu in zip(tab['plate'], tab['ifudsgn'])]

        # otherwise, move forward and load all files
        logcubes = [m.load_drp_logcube(str(p), i, MPL_v)
                    for p, i in zip(tab['plate'], tab['ifudsgn'])]

        naxis1, naxis2, naxis3 = logcubes[0]['FLUX'].data.shape

        # handle the case where LOGCUBEs have different shapes
        if len(set(map(lambda h: h['FLUX'].data.shape, logcubes))) > 1:
            dummy = np.zeros((0, nspec))
            return dummy, dummy, False

        # extract & reshape data
        # axis 0: observation; axis 1: spaxel; axis 2: wavelength
        fluxs = np.stack([cube['FLUX'].data.reshape((naxis1, -1)).T
                          for cube in logcubes])
        ivars = np.stack([cube['IVAR'].data.reshape((naxis1, -1)).T
                          for cube in logcubes])
        mb_mask = np.stack([m.mask_from_maskbits(
                                a=cube['MASK'].data.reshape((naxis1, -1)).T,
                                b=[0, 1, 2, 3, 10])
                            for cube in logcubes])
        ivars *= (~ mb_mask)
        ivars[ivars == 0.] = eps

        # differences between pairs
        n_reobs = len(logcubes)
        pairs1 = comb(range(n_reobs), 2)
        pairs2 = comb(range(n_reobs), 2)
        w = 1. / np.row_stack([1. / (1. / ivars[i1, ...] + 1. / ivars[i2, ...])
                              for i1, i2 in pairs1])
        diffs = np.row_stack([fluxs[i1, ...] - fluxs[i2, ...]
                              for (i1, i2) in pairs2])

        return diffs, w, True

    @staticmethod
    def process_BOSS_mult(tab, dlogl=1.0e-4):
        '''
        return differences between reobservations of same BOSS object
        '''

        # load all the files
        baseloc = mangarc.BOSS_data_loc
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

        return logl_full, diffs, ws


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
    cov = wdiffs.T.dot(wdiffs) / (ws.T.dot(ws) - 1.)
    return cov


if __name__ == '__main__':
    #'''
    spAll_loc = os.path.join(mangarc.zpace_sdss_data_loc,
                             'boss', 'spAll-v5_9_0.fits')
    spAll = fits.open(spAll_loc, memmap=True)
    Cov_boss = Cov_Obs.from_BOSS_reobs(spAll=spAll, n=None, platelim=5987)
    Cov_boss.write_fits('boss_Kspec.fits')
    Cov_boss.make_im(kind='boss')
    #'''
    # =====
    '''
    Cov_manga = Cov_Obs.from_MaNGA_reobs(n=None)
    Cov_manga.write_fits('manga_Kspec.fits')
    Cov_manga.make_im(kind='manga')
    '''
