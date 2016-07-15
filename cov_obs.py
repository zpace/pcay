import numpy as np
import matplotlib.pyplot as plt

from astropy import constants as c, units as u, table as t
from astropy.io import fits
from astropy import coordinates as coords

import os
import sys
import cPickle as pkl

import spec_tools
import ssp_lib
import manga_tools as m

from itertools import izip, product
from glob import glob

eps = np.finfo(float).eps

# =====

class Cov_Obs(object):
    '''
    a class to precompute observational spectral covariance matrices
    '''
    def __init__(self, cov, lllim, dlogl):
        self.cov = cov
        self.nspec = len(cov)
        self.lllim = lllim
        self.dlogl = dlogl

    def write_fits(self, fname='cov.fits'):
        hdu_ = fits.PrimaryHDU()
        hdu = fits.ImageHDU(data=self.cov)
        hdu.header['LOGL0'] = np.log10(self.lllim)
        hdu.header['DLOGL'] = self.dlogl
        hdulist = fits.HDUList([hdu_, hdu])
        hdulist.writeto(fname, clobber=True)

    @staticmethod
    def _mults(spAll):
        (objid, plate, mjd, fiberid) = (
            spAll[1].data['OBJID'], spAll[1].data['PLATE'],
            spAll[1].data['MJD'], spAll[1].data['FIBERID'])
        obs = t.Table([objid, plate, mjd, fiberid],
                      names=['objid', 'plate', 'mjd', 'fiberid'])
        obs = obs[obs['objid'] != '                   ']
        obs['objid'] = obs['objid'].astype(int)
        objs = obs.group_by('objid')

        start = np.array(objs.groups.indices[:-1])
        stop = np.array(objs.groups.indices[1:])
        # use objects with more than two observations
        repeat = stop - start > 2
        repeat_sf_ixs = np.column_stack([start, stop])[stop - start > 1, :]
        obs_dupl = objs.groups[repeat]
        objs_dupl = objs_dupl = obs_dupl.group_by('objid')
        objids = objs_dupl.groups.keys

        return dict(zip(
            objids, objs_dupl['plate', 'mjd', 'fiberid'].groups)[:5])

    @staticmethod
    def download_obj_specs(tab, base_dir='calib/'):
        '''
        for all objects in a `mults`-style dict, download their FITS spectra
        '''

        make_full_fname = lambda row: '{0}/spec-{0}-{1}-{2:04d}.fits'.format(
            *row)
        make_final_fname = lambda row: os.path.join(
            base_dir, 'spec-{0}-{1}-{2:04d}.fits'.format(
                *row))
        full_fnames = map(make_full_fname, tab)
        final_fnames = map(make_final_fname, tab)
        success = [False, ] * len(full_fnames)
        for i, fname in enumerate(full_fnames):
            # if one does exist, move on
            if os.path.isfile(os.path.join(base_dir, fname)):
                success[i] = True
                continue

            # if not, retrieve it over rsync!
            q = 'rsync -raz --password-file={0} rsync://sdss@{1} {2}'.format(
                os.path.join(m.drpall_loc, m.pw_loc), # password file
                os.path.join(
                    m.base_url,
                    'ebosswork/eboss/spectro/redux/v5_9_0/spectra/lite',
                    fname),
                'calib')
            s_ = os.system(q) # os.system() returns 0 on success
            if s_ == 0:
                success[i] = True
            elif s_ == 2:
                raise KeyboardInterrupt
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
                    for f, s in izip(fnames, success) if s]
        else:
            try:
                data = [fits.open(f)['COADD'].data[data_name][i0 : i0 + nspec]
                        for f, s, i0 in izip(fnames, success, lam_ix0s) if s]
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

        return normed

    @classmethod
    def from_spAll(cls, spAll, lllim=3650.059970708618, nspec=4378,
                   dlogl=1.0e-4):
        '''
        returns a covariance object made from an spAll file
        '''

        # dict of multiply-observed objects
        mults = Cov_Obs._mults(spAll)
        del spAll # clean up!

        normed_specs = np.row_stack([
            Cov_Obs.load_zeronormed_obj_spec(
                *Cov_Obs.download_obj_specs(obj),
                lllim=lllim, nspec=nspec, i=i)
            for i, (k, obj) in enumerate(mults.iteritems())])

        # filter out bad rows
        bad_rows = (np.isnan(normed_specs).sum(axis=1) > 0)
        normed_specs = normed_specs[~bad_rows, :]
        cov = np.cov(normed_specs.T)

        return cls(cov, lllim=lllim, dlogl=dlogl)

if __name__ == '__main__':
    spAll = fits.open('spAll-v5_9_0.fits', memmap=True)
    Cov = Cov_Obs.from_spAll(spAll=spAll)
    Cov.write_fits()
