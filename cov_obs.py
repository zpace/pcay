import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy import units as u, table as t  # , constants as c
from astropy.io import fits

import os
import sys

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

boss_calib_dir = os.path.join(mangarc.zpace_sdss_data_loc, 'boss/calib')


class Cov_Obs(object):
    '''
    a class to precompute observational spectral covariance matrices
    '''

    def __init__(self, cov, lllim, dlogl, nobj):
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
    def from_spAll(cls, spAll, lllim=3650.059970708618, nspec=4378,
                   dlogl=1.0e-4):
        '''
        returns a covariance object made from an spAll file
        '''

        # dict of multiply-observed objects
        mults, SB_r_mean = Cov_Obs._mults(spAll)
        del spAll  # clean up!

        stack = [
            Cov_Obs.load_zeronormed_obj_spec(
                *Cov_Obs.download_obj_specs(obj),
                lllim=lllim, nspec=nspec, i=i)
            for i, (k, obj) in enumerate(mults.items())]

        resids = np.concatenate([s[:len(s) // 2] for s in stack], axis=0)
        ivars = np.concatenate([s[len(s) // 2:] for s in stack], axis=0)

        # filter out bad rows
        bad_rows = (np.isnan(resids).sum(axis=1) > 0)
        resids = resids[~bad_rows, :]
        ivars = ivars[~bad_rows, :]
        nobj, nspec = resids.shape

        qw = resids * ivars
        cov = qw.T.dot(qw) / ivars.T.dot(ivars)

        return cls(cov, lllim=lllim, dlogl=dlogl, nobj=nobj)

    @classmethod
    def from_MaNGA_reobs(cls, lllim=3650.059970708618, nspec=4563,
                         dlogl=1.0e-4, MPL_v=mpl_v, n=None):
        '''
        returns a covariance object made from reobserved MaNGA IFU LOGCUBEs
        '''

        drpall = t.Table.read(os.path.join(
            mangarc.manga_data_loc[mpl_v],
            'drpall-{}.fits'.format(m.MPL_versions[mpl_v])))

        drpall = drpall[drpall['ifudesignsize'] != -9999]
        objs = drpall.group_by('mangaid')

        start = np.array(objs.groups.indices[:-1])
        stop = np.array(objs.groups.indices[1:])

        repeat = stop - start > 1

        # only use objects with multiple observations with same IFU size
        def onesize_(tab):
            return len(np.unique(tab['ifudesignsize'])) == 1
        onesize = np.array(list(map(onesize_, objs.groups)))

        # grouped filter
        obs_dupl = objs.groups[repeat & onesize]

        # final grouping
        objs_dupl = obs_dupl.group_by('mangaid')
        mangaids = objs_dupl.groups.keys
        n_unique = len(mangaids)

        # if unspecified, use all galaxies
        if n is None:
            n = n_unique

        groups = [tab for tab in objs_dupl.groups]
        use = [True if i <= n else False for i in range(n_unique)]
        groups = [g for g, u in zip(groups, use) if u]

        # process multiply-observed datacubes, up to limit set
        diffs, good, n = zip(
            *[Cov_Obs.process_MaNGA_mult(grp, nspec)
              for grp in groups])

        diffs = np.row_stack([p for (p, g) in zip(diffs, good) if g]).T
        n = [nn for (nn, g) in zip(n, good) if g]
        N = np.sum(n)

        cov = np.cov(diffs.T)

        print(np.median(np.diag(cov)))

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
        hdulist.writeto(fname, clobber=True)

    def make_im(self, kind):
        l = self.l
        fig = plt.figure(figsize=(6, 6), dpi=300)
        ax = fig.add_subplot(111)
        im = ax.imshow(
            np.abs(self.cov), extent=[l.min(), l.max(), l.min(), l.max()],
            vmax=100., vmin=1.0e-8, interpolation='None',
            aspect='equal', norm=LogNorm())
        ax.set_xlabel(r'$\lambda$', size=8)
        ax.set_ylabel(r'$\lambda$', size=8)
        plt.colorbar(im, ax=ax, shrink=0.8)
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
    def process_MaNGA_mult(tab, nspec=4563):
        '''
        check for presence of datacubes for a single multiply-observed
            MaNGA object, and (if the data exists) process the observations
            into an array of residuals

        Note: this assumes that all data is present at the correct location.
            Missing data (such that zero or one observations are available)
            will result in an empty array (with correct dimensions)
            being returned

        Returns:
         - resids_normed: SNR-1-normed variance (or zeros array of right shape)
         - good: indicates whether resids_normed is good
        '''

        from itertools import combinations as comb
        from scipy.signal import medfilt

        # load all LOGCUBES for a given MaNGA-ID
        # files have form 'manga-<PLATE>-<IFU>-LOGCUBE.fits.gz'
        fnames = [os.path.join(mangarc.manga_data_loc[mpl_v],
                               'drp/', str(plate), 'stack/',
                               '-'.join(('manga', str(plate), str(ifu),
                                         'LOGCUBE.fits.gz')))
                  for plate, ifu in zip(tab['plate'], tab['ifudsgn'])]

        # otherwise, move forward and load all files
        logcubes = [fits.open(f) for f in fnames]

        # handle the odd case where there aren't enough observations loaded
        n_reobs = len(logcubes)
        naxis1, naxis2, naxis3 = logcubes[0]['FLUX'].data.shape
        if n_reobs < 2:
            return np.zeros((0, nspec)), False, 0

        # handle the case where LOGCUBEs have different shapes
        if len(set(map(lambda h: h['FLUX'].data.shape, logcubes))) > 1:
            return np.zeros((0, nspec)), False, 0

        # extract & reshape data
        fluxs = np.stack([cube['FLUX'].data.reshape((naxis1, -1)).T
                          for cube in logcubes])
        ivars = np.stack([cube['IVAR'].data.reshape((naxis1, -1)).T
                          for cube in logcubes])

        # replace bad pixels by median of surrounding 300 pixels
        meds = medfilt(fluxs, [1, 301, 1])
        fluxs[ivars == 0] = meds[ivars == 0]

        # load rimg extension of only one observation
        rimg = logcubes[0]['RIMG'].data.flatten()
        rimg /= rimg.max()

        # exclude rows where any observation has zero total weight
        # and exclude rows with low r-band signal (< 1% of max)
        good = np.all(ivars.sum(axis=-1) != 0, axis=0)
        good *= (rimg > .01)
        ivars = ivars[:, good, :]
        fluxs = fluxs[:, good, :]

        # norm the spectra and their uncertainties
        norms = np.average(fluxs, weights=ivars, axis=2)
        specs = (fluxs / norms[..., None]).reshape((n_reobs, fluxs.shape[1], -1))
        uncs_f = (np.sqrt(ivars) * specs).reshape((n_reobs, fluxs.shape[1], -1))

        # differences between pairs
        pairs_ixs = comb(range(n_reobs), 2)
        diffs = np.row_stack([specs[i1, ...] - specs[i2, ...]
                          for (i1, i2) in pairs_ixs])

        return diffs, True, n_reobs * good.sum()

    @staticmethod
    def _mults(spAll, i_lim=10):
        '''
        return a dict of duplicate observations of the same object, using
            astropy table grouping

        also return a mean object surface brightness (nMgy/arcsec2) to aid
            in scaling the covariance matrix against MaNGA spaxels
        '''

        (objid, plate, mjd, fiberid) = (
            spAll[1].data['OBJID'], spAll[1].data['PLATE'],
            spAll[1].data['MJD'], spAll[1].data['FIBERID'])
        # average surface brightness within .67 arcsec of object center
        # (from photometric pipeline)
        SB_r = t.Column(
            data=spAll[1].data['APERFLUX'][:, 2, 1] / (np.pi * 0.67**2.),
            name='SB_r')
        obs = t.Table([objid, plate, mjd, fiberid, SB_r],
                      names=['objid', 'plate', 'mjd', 'fiberid', 'SB_r'])
        obs = obs[obs['objid'] != '                   ']
        obs = obs[np.nonzero(obs['SB_r'])]
        obs['objid'] = obs['objid'].astype(int)
        objs = obs.group_by('objid')

        start = np.array(objs.groups.indices[:-1])
        stop = np.array(objs.groups.indices[1:])
        # use objects with more than two observations
        repeat = stop - start > 2
        obs_dupl = objs.groups[repeat]  # duplicate OBSERVATIONS
        objs_dupl = obs_dupl.group_by('objid')  # the OBJECTS corresponding
        objids = list(objs_dupl.groups.keys['objid'])

        if i_lim is None:
            i_lim = len(objids)

        mults_dict = {objids[i]:
                      t.Table(objs_dupl['plate', 'mjd', 'fiberid'].groups[i])
                      for i in range(i_lim - 1)}

        SB_r_mean = np.mean(obs['SB_r']) * 1.0e-9 * m.Mgy / (u.arcsec)**2.

        return mults_dict, SB_r_mean

    @staticmethod
    def download_obj_specs(tab):
        '''
        for all objects in a `mults`-style dict, download their FITS spectra
        '''

        def make_remote_fname(row):
            return '{0}/spec-{0}-{1}-{2:04d}.fits'.format(*row)
        remote_fnames = list(map(make_remote_fname, tab))

        def make_local_fname(row):
            return os.path.join(
                boss_calib_dir, 'spec-{0}-{1}-{2:04d}.fits'.format(*row))
        local_fnames = list(map(make_local_fname, tab))

        success = [False, ] * len(remote_fnames)
        for i, (fname_r, fname_l) in enumerate(zip(remote_fnames,
                                                   local_fnames)):
            # if file has already been downloaded, move on
            if os.path.isfile(fname_l):
                success[i] = True
                continue

            # if not, retrieve it over rsync!
            print('Downloading remote file:', fname_r)
            q = 'rsync -raz --password-file={0} rsync://sdss@{1} {2}'.format(
                mangarc.password_file,  # SAS password
                os.path.join(  # what file we're looking for
                    m.base_url,
                    'ebosswork/eboss/spectro/redux/v5_9_0/spectra/lite',
                    fname_r),
                boss_calib_dir)  # where to put it
            # print(q)
            s_ = os.system(q)  # os.system() returns 0 on success
            if s_ == 0:
                success[i] = True
            elif s_ == 2:
                raise KeyboardInterrupt

        return boss_calib_dir, local_fnames, success

    @staticmethod
    def load_obj_spec(base_dir, fnames, success, data_name,
                      lam_ix0s=None, nspec=None):
        '''
        for all files in a list, load and return an array of fluxes
        '''

        # handle cacse that we want everything
        if (lam_ix0s is None) or (nspec is None):
            data = [fits.open(f)['COADD'].data[data_name]
                    for f, s in zip(fnames, success) if s]
        else:
            try:
                data = [fits.open(f)['COADD'].data[data_name][i0: i0 + nspec]
                        for f, s, i0 in zip(fnames, success, lam_ix0s) if s]
            # handle cases where wavelength solution is outside bounds
            # shouldn't just throw out individual spectra, since that
            # could list bring down to length-one and mess up statistics
            except IndexError:
                return None
            # if things have different lengths
            if True in list(map(lambda x: len(x) != nspec, data)):
                return None

        return data

    @staticmethod
    def load_zeronormed_obj_spec(base_dir, fnames, success,
                                 lllim, nspec, i):
        loglam = Cov_Obs.load_obj_spec(
            base_dir, fnames, success, data_name='loglam')
        # figure out where to start and end
        lllim_log = np.log10(lllim)
        lam_ix0s = [np.argmin((logl - lllim_log)**2.) for logl in loglam]

        flux = Cov_Obs.load_obj_spec(
            base_dir, fnames, success, data_name='flux', lam_ix0s=lam_ix0s,
            nspec=nspec)
        ivar = Cov_Obs.load_obj_spec(
            base_dir, fnames, success, data_name='ivar', lam_ix0s=lam_ix0s,
            nspec=nspec)

        if (flux is None) or (ivar is None):
            return np.nan * np.ones(nspec)

        flux = np.row_stack(flux)
        ivar = np.row_stack(ivar)
        ivar = np.maximum(ivar, eps)
        normed = flux - np.average(flux, weights=ivar, axis=0)

        return np.concatenate([normed, ivar], axis=0)


if __name__ == '__main__':
    '''
    spAll_loc = os.path.join(mangarc.zpace_sdss_data_loc,
                             'boss', 'spAll-v5_9_0.fits')
    spAll = fits.open(spAll_loc, memmap=True)
    Cov_boss = Cov_Obs.from_spAll(spAll=spAll)
    Cov_boss.write_fits('boss_Kspec.fits')
    '''
    # =====

    Cov_manga = Cov_Obs.from_MaNGA_reobs(n=None)
    Cov_manga.write_fits('manga_Kspec.fits')
    Cov_manga.make_im(kind='manga')
