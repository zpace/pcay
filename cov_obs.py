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

    @classmethod
    def from_spAll(cls, spAll):
        '''
        returns a covariance object made from an spAll file
        '''
        # dict of multiply-observed objects
        mults = self._mults(spAll)
        del spAll # clean up!

        # build list of fluxes and ivars

        return cls(cov)

def extract_duplicate_spectra(objid, group, lllim, nspec):
    '''
    for a single object, extract duplicate spectra in the correct
        wavelength range, and output into two stacked arrays (flux and ivar)
    '''

    fnames = [None, ] * len(group)
    # check to see if spectra exist
    for i, specobj in enumerate(group):
        fname = '{0}/spec-{0}-{1}-{2:04d}.fits\n'.format(*specobj)
        # if one doesn't exist, get it!
        if ~os.path.isfile(os.path.join('calib/', fname)):
            q = 'rsync -raz --password-file={0} rsync://sdss@{1} {2}'.format(
                    os.path.join(m.drpall_loc, m.pw_loc),
                    os.path.join(
                        m.base_url,
                        'ebosswork/eboss/spectro/redux/v5_9_0/spectra/lite',
                        fname.rstrip('\n')),
                    'calib')
            os.system(q)

        fnames[i] = os.path.join(
            'calib/', 'spec-{0}-{1}-{2:04d}.fits'.format(*specobj))

    # load each spectrum
    group_hdulist = [fits.open(fname) for fname in fnames]
    # decide where each of them start and stop
    lllim_i = [np.argmin((10.**hdulist['COADD'].data['loglam'] - lllim)**2.) for hdulist in group_hdulist]
    lulim_i = [li + nspec for li in lllim_i]
    obj_fs = np.row_stack(
        [group_hdulist[i]['COADD'].data['flux'][lllim_i[i]:lulim_i[i]]
         for i in range(len(group_hdulist))])

    obj_ivars = np.row_stack(
        [group_hdulist[i]['COADD'].data['ivar'][lllim_i[i]:lulim_i[i]]
         for i in range(len(group_hdulist))])

    return obj_fs, obj_ivars

def compute_cov(objs_fs, objs_ivars, dest='cov_obs.fits'):
    '''
    given a list of spectra, grouped by object, find the spectral covariance
    '''

    # for each object (group of spectra), subtract the mean spectrum
    # (mean is weighted by ivar + eps)

    objs_normed = [f - np.average(f, weights=i, axis=0) for f in objs_fs]
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
    # obs_t = obs_t[:1e5]
    del spAll # clean up!

    groups = {}
    '''
    # touch the download list file to clear it
    with open('specfiles.txt', 'w') as f:
        pass
    '''

    obs_t_by_object = obs_t.group_by('objid')
    for i, (o, g) in enumerate(izip(obs_t_by_object.groups.keys,
                                    obs_t_by_object.groups)):
        if len(g) <= 1:
            continue

        groups[o] = g['plate', 'mjd', 'fiberid']

        # add to the download list file
        # lines have form PLATE/spec-PLATE-MJD-FIBERID.fits
        '''
        with open('specfiles.txt', 'a') as f:
            for row in g:
                f.write('{0}/spec-{0}-{1}-{2:04d}.fits\n'.format(
                    row['plate'], row['mjd'], row['fiberid']))
        '''

    pkl.dump(groups, open('groups.pkl', 'wb'))
    return groups

if __name__ == '__main__':
    groups = pkl.load(open('groups.pkl', 'rb'))

    objs_fs = [None, ] * len(groups)
    objs_ivars = [None, ] * len(groups)

    for i, (k, obj) in enumerate(groups.iteritems()):
        objs_fs[i], objs_ivars[i] = extract_duplicate_spectra(
            k, obj, lllim=4000, nspec=500)
        if i == 2: break

    print objs_fs[:2], objs_ivars[:2]
