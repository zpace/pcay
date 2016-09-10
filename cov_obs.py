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


class Cov_Obs(object):
    '''
    a class to precompute observational spectral covariance matrices
    '''

    def __init__(self, cov, lllim, dlogl, nobj, SB_r_mean):
        self.cov = cov
        self.nspec = len(cov)
        self.lllim = lllim
        self.loglllim = np.log10(self.lllim)
        self.dlogl = dlogl
        self.nobj = nobj
        self.SB_r_mean = SB_r_mean

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
            for i, (k, obj) in enumerate(mults.iteritems())]

        resids = np.concatenate([s[:len(s) // 2] for s in stack], axis=0)
        ivars = np.concatenate([s[len(s) // 2:] for s in stack], axis=0)

        # filter out bad rows
        bad_rows = (np.isnan(resids).sum(axis=1) > 0)
        resids = resids[~bad_rows, :]
        ivars = ivars[~bad_rows, :]
        nobj, nspec = resids.shape

        qw = resids * ivars
        cov = qw.T.dot(qw) / ivars.T.dot(ivars)

        return cls(cov, lllim=lllim, dlogl=dlogl, nobj=nobj,
                   SB_r_mean=SB_r_mean)

    @classmethod
    def from_MaNGA_reobs(cls, lllim=3650.059970708618, nspec=4378,
                         dlogl=1.0e-4, MPL_v=mpl_v):
        '''
        returns a covariance object made from reobserved MaNGA IFU LOGCUBEs
        '''

        drpall = t.Table.read(os.path.join(
            mangarc.manga_data_loc,
            'drpall-{}.fits'.format(m.MPL_versions[MPL_v])))

        drpall = drpall[drpall['ifudesignsize'] != -9999]
        objs = drpall.group_by('mangaid')

        start = np.array(objs.groups.indices[:-1])
        stop = np.array(objs.groups.indices[1:])

        repeat = stop - start > 1

        # only use objects with multiple observations with same IFU size
        def onesize_(tab):
            return len(np.unique(tab['ifudesignsize'])) == 1
        onesize = map(onesize_, objs.groups)

        # grouped filter
        obs_dupl = objs.groups[repeat & onesize]

        # final grouping
        objs_dupl = objs_dupl = obs_dupl.group_by('mangaid')
        # mangaids = objs_dupl.groups.keys
        # print mangaids

        # process multiply-observed datacubes

        mults = [
            Cov_Obs.process_MaNGA_mult(tab, nspec) for tab in objs_dupl.groups]

        mults = np.row_stack(mults)
        nobj = mults.shape[0]

        cov = np.cov(mults.T)

        return cls(cov, lllim=lllim, dlogl=dlogl, nobj=nobj, SB_r_mean=None)

    @classmethod
    def from_fits(cls, fname):
        hdulist = fits.open(fname)
        cov = hdulist[1].data
        h = hdulist[1].header
        lllim = 10.**h['LOGL0']
        dlogl = h['DLOGL']
        nobj = h['NOBJ']
        SB_r_mean = h['SBRMEAN'] * 1.0e-9 * m.Mgy / (u.arcsec)**2.
        return cls(cov=cov, lllim=lllim, dlogl=dlogl, nobj=nobj,
                   SB_r_mean=SB_r_mean)

    # =====
    # methods
    # =====

    def write_fits(self, fname='cov.fits'):
        hdu_ = fits.PrimaryHDU()
        hdu = fits.ImageHDU(data=self.cov)
        hdu.header['LOGL0'] = np.log10(self.lllim)
        hdu.header['DLOGL'] = self.dlogl
        hdu.header['NOBJ'] = self.nobj
        hdu.header['SBRMEAN'] = self.SB_r_mean.value
        hdulist = fits.HDUList([hdu_, hdu])
        hdulist.writeto(fname, clobber=True)

    def make_im(self):
        l = self.l
        fig = plt.figure(figsize=(6, 6), dpi=300)
        ax = fig.add_subplot(111)
        im = ax.imshow(
            np.abs(self.cov), extent=[l.min(), l.max(), l.min(), l.max()],
            vmax=np.max(np.abs(self.cov)),
            aspect='equal', norm=LogNorm())
        ax.set_xlabel(r'$\lambda$', size=8)
        ax.set_ylabel(r'$\lambda$', size=8)
        plt.colorbar(im, ax=ax)
        plt.savefig('cov_obs.png', dpi=300)

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
    def process_MaNGA_mult(tab, nspec):
        '''
        check for presence of datacubes for a single multiply-observed
            MaNGA object, and (if the data exists) process the observations
            into an array of residuals

        Note: this assumes that all data is present at the correct location.
            Missing data (such that zero or one observations are available)
            will result in an empty array (with correct dimensions)
            being returned
        '''

        # load all LOGCUBES for a given MaNGA-ID
        # files have form 'manga-<PLATE>-<IFU>-LOGCUBE.fits.gz'
        fnames = [os.path.join(mangarc.manga_data_loc,
                               'drp/', plate, 'stack/',
                               '-'.join(('manga', plate, ifu,
                                         'LOGCUBE.fits.gz')))
                  for plate, ifu in zip(tab['plate'], tab['ifudsgn'])]

        # handle the case where there aren't enough observations loaded
        if sum(map(os.path.exists, fnames)) < 2:
            return np.zeros((0, nspec))

        # otherwise, move forward and load all files
        logcubes = [fits.open(f) for f in fnames]

        # extract & reshape data
        ivars = [cube['IVAR'].data for cube in logcubes]

        # handle the case where LOGCUBEs have different shapes
        if len(set([i.shape for i in ivars])) > 1:
            return np.zeros((0, nspec))

        # rearrange datacube into faux RSS
        ivars = np.stack(
            [ivars.reshape(ivar.shape[0], -1).T for ivar in ivars])
        fluxs = [cube['FLUX'].data for cube in logcubes]
        fluxs = np.stack(
            [flux.reshape(ivars.shape[0], -1).T for flux in fluxs])

        # exclude rows where any observation has zero total weight
        good = np.all(ivars.sum(axis=-1) != 0, axis=0)
        ivars = ivars[:, good, :]
        fluxs = fluxs[:, good, :]

        # account for spectral elements with zero total weights
        ivars += eps
        # mean of each row
        mean = np.average(fluxs, axis=0, weights=ivars)
        resids = (fluxs - mean).reshape((-1, mean.shape[1]))

        return resids

    @staticmethod
    def _mults(spAll, i_lim=100):
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

        mults_dict = {objids[i]: t.Table(objs_dupl.groups[i])
                      for i in range(i_lim)}

        SB_r_mean = np.mean(obs['SB_r']) * 1.0e-9 * m.Mgy / (u.arcsec)**2.

        return mults_dict, SB_r_mean

    @staticmethod
    def download_obj_specs(tab, base_dir='calib/'):
        '''
        for all objects in a `mults`-style dict, download their FITS spectra
        '''

        def make_full_fname(row):
            return '{0}/spec-{0}-{1}-{2:04d}.fits'.format(*row)
        full_fnames = map(make_full_fname, tab)

        success = [False, ] * len(full_fnames)
        for i, fname in enumerate(full_fnames):
            # if file has already been downloaded, move on
            if os.path.isfile(os.path.join(base_dir, fname)):
                success[i] = True
                continue

            # if not, retrieve it over rsync!
            q = 'rsync -raz --password-file={0} rsync://sdss@{1} {2}'.format(
                os.path.join(m.drpall_loc, m.pw_loc),  # password file
                os.path.join(
                    m.base_url,
                    'ebosswork/eboss/spectro/redux/v5_9_0/spectra/lite',
                    fname),
                'calib')
            s_ = os.system(q)  # os.system() returns 0 on success
            if s_ == 0:
                success[i] = True
            elif s_ == 2:
                raise KeyboardInterrupt

        def make_final_fname(row):
            return os.path.join(
                base_dir, 'spec-{0}-{1}-{2:04d}.fits'.format(*row))

        final_fnames = map(make_final_fname, tab)

        return base_dir, final_fnames, success

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
            if True in map(lambda x: len(x) != nspec, data):
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
    spAll_loc = os.path.join(mangarc.zpace_sdss_data_loc,
                             'boss', 'spAll-v5_9_0.fits')
    spAll = fits.open(spAll_loc, memmap=True)
    Cov_boss = Cov_Obs.from_spAll(spAll=spAll)
    Cov_boss.write_fits('boss_Kspec.fits')

    # =====

    Cov_manga = Cov_Obs.from_MaNGA_reobs()
    Cov_manga.write_fits('manga_Kspec.fits')
