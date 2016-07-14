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

# =====

class Cov_Obs(object):
    '''
    a class to precompute observational spectral covariance matrices
    '''
    def __init__(self, cov):
        self.cov = cov
        self.nspec = len(cov)

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

        start = np.array(objs.groups.indices)
        stop = np.append(start[1:], start[-1] + 1)
        # use objects with more than two observations
        repeat = stop - start > 2
        repeat_sf_ixs = np.column_stack([start, stop])[stop - start > 1, :]

        obs_dupl = objs.groups[repeat]
        objs_dupl = objs_dupl = obs_dupl.group_by('objid')
        objids = objs_dupl.groups.keys

        return dict(zip(objids, objs_dupl['plate', 'mjd', 'fiberid'].groups))

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
        return base_dir, final_fnames, success

    @staticmethod
    def load_obj_spec(base_dir, fnames, success, data_name):
        '''
        for all files in a list, load and return an array of fluxes
        '''

        data = [fits.open(f)['COADD'].data[data_name]
                for f, s in izip(fnames, success) if s]

        return data

    @staticmethod
    def load_zeronormed_obj_spec(base_dir, fnames, success):
        flux = Cov_Obs.load_obj_spec(
            base_dir, fnames, success, data_name='flux')
        ivar = Cov_Obs.load_obj_spec(
            base_dir, fnames, success, data_name='ivar')
        loglam = Cov_Obs.load_obj_spec(
            base_dir, fnames, success, data_name='loglam')

        normed = flux - np.average(flux, weights=ivar, axis=0)

        return normed

    @classmethod
    def from_spAll(cls, spAll, lam):
        '''
        returns a covariance object made from an spAll file
        '''
        # dict of multiply-observed objects
        mults = Cov_Obs._mults(spAll)
        del spAll # clean up!

        # build list of fluxes
        normed_specs = np.row_stack([
            Cov_Obs.load_zeronormed_obj_spec(
                *Cov_Obs.download_obj_specs(obj))
            for k, obj in mults.iteritems()])

        return cls(cov)
