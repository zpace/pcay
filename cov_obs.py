import numpy as np
import matplotlib.pyplot as plt

from astropy import constants as c, units as u, table as t
from astropy.io import fits
from astropy import coordinates as coords

import os
import sys

import spec_tools
import ssp_lib
import manga_tools as m

from itertools import izip, product
from glob import glob

# =====

spec_locs = '/media/SAMSUNG/SDSSIII/v5_4_45'

def extract_duplicate_spectra(group, lllim, lulim):
    '''
    for a single object, extract duplicate spectra in the correct
        wavelength range, and output into two stacked arrays (flux and ivar)
    '''

def compute_cov(objs_fs, objs_ivars, dest='cov_obs.fits'):
    '''
    given a list of spectra, grouped by object, find the spectral covariance
    '''

    # for each object (group of spectra), subtract the mean spectrum
    # (mean is weighted by ivar + eps)

    objs_normed = [f - np.average(f, weights=i, axis=0) for f in obj_fs]
    objs_normed = np.row_stack(objs_normed)

    objs_ivars = np.row_stack(objs_ivars)
    # and take the (weighted) covariance of the residual

    # this loop is bad and I should feel bad
    covar = np.empty((objs_fs[0].shape[1], ) * 2)
    for j, k in product(range(covar.shape[0]), range(covar.shape[1])):
        covar[j, k] = np.average(
            (objs_normed[:, j] * objs_normed[:, k]),
            weights=(objs_ivars[:, j] * objs_ivars[:, k]))

    hdulist = [fits.PrimaryHDU(covar)]
    hdulist = fits.HDUList(hdulist)
    hdulist.writeto(dest)

    return covar

def find_BOSS_duplicates():
    '''
    using spAll file, find objIDs that have multiple observations, and
        the associated [plate/mjd/fiberid] combinations
    '''
    # read in file that contains all BOSS observation coordinates
    spAll = fits.open('spAll-v5_9_0.fits', memmap=True)
    (objid, specprimary, plate, mjd, fiberid) = (
        spAll[1].data['OBJID'], spAll[1].data['SPECPRIMARY'],
        spAll[1].data['PLATE'], spAll[1].data['MJD'],
        spAll[1].data['FIBERID'])
    obs_t = t.Table([objid, specprimary, plate, mjd, fiberid],
                     names=['objid', 'specprimary', 'plate', 'mjd', 'fiberid'])
    obs_t = obs_t[obs_t['objid'] != '                   ']
    del spAll # clean up!

    groups = {}
    # touch the download list file to clear it
    with open('specfiles.txt', 'w') as f:
        pass

    obs_t_by_object = obs_t.group_by('objid')
    for i, (o, g) in enumerate(izip(obs_t_by_object.groups.keys,
                                    obs_t_by_object.groups)):
        if len(g) <= 1:
            continue

        groups[o] = g['plate', 'mjd', 'fiberid']

        # add to the download list file
        # lines have form PLATE/spec-PLATE-MJD-FIBERID.fits
        with open('specfiles.txt', 'a') as f:
            for row in g:
                f.write('{0}/spec-{0}-{1}-{2}.fits\n'.format(
                    row['plate'], row['mjd'], row['fiberid']))

    print len(groups)

if __name__ == '__main__':
    find_BOSS_duplicates()
