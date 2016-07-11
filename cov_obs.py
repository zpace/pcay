import numpy as np
import matplotlib.pyplot as plt

from astropy import constants as c, units as u, table as t
from astropy.io import fits
from astropy import coordinates as coords

import os

import spec_tools
import ssp_lib
import manga_tools as m

from itertools import izip, product
from glob import glob

def extract_duplicate_spectra(lllim, lulim):
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
            weights=(objs_ivars[:, j]*objs_ivars[:, k]))

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
    obs_t = t.Table.read()

    # objectIDs for all "primary objects"
    prim = obs_t[obs_t['specprimary']]['objid']

    collated = {}
    for p in prim:
        obj = obs_t[obs_t['objid'] == p]
        # when there's only one observation, we don't care!
        if len(obj) <= 1:
            continue

        # when there's more than one, output [plate, mjd, fiber] to dict
        collated[p] = [r['plate', 'mjd', 'fiberid'] r in obj]
