import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import units as u, constants as c, table as t

from scipy.interpolate import interp1d
from scipy.signal import medfilt

import os, sys
from copy import copy

# local
import utils as ut
import indices
from spectrophot import Spec2Phot

from importer import *

import manga_tools as m

def noisify_cov(cov, mapshape):
    cov_noise = np.random.multivariate_normal(
        mean=np.zeros_like(np.diag(cov.cov)),
        cov=cov.cov, size=mapshape)
    cov_noise = np.moveaxis(cov_noise, [0, 1, 2], [1, 2, 0])
    #cov_noise = cov_noise[i0:i0 + nl, :, :]
    rms = 50. * np.sqrt(np.mean(cov_noise**2., axis=0))
    return cov_noise, rms

def compute_snrcube(flux, ivar, filtersize_l=15, return_rms_map=False):
    '''
    compute the (appx) SNR cube, and if specified, return the spaxelwise
        rms map of the cube
    '''

    snrcube = medfilt(flux * np.sqrt(ivar), [filtersize_l, 1, 1])
    rms_map = 1. / np.mean(snrcube, axis=0)

    if return_rms_map:
        return snrcube, rms_map
    else:
        return snrcube

class FakeData(object):
    '''
    Make a fake IFU and fake DAP stuff
    '''
    def __init__(self, lam_model, spec_model, meta_model,
                 row, drp_base, dap_base, plateifu_base, model_ix,
                 Kspec_obs=None):
        '''
        create mocks of DRP LOGCUBE and DAP MAPS

        1. characterize SNR of observed data
        2. redshift model to observed-frame
        3. attenuate according to MW reddening law
        4. blur according to instrumental dispersion
        5. resample onto rectified (observed) wavelength grid
        6. add noise from full covariance prescription OR from SNR
        7. scale according to r-band surface brightness
        8. mask where there's no flux
        '''

        self.model_ix = model_ix

        flux_obs = drp_base['FLUX'].data
        ivar_obs = drp_base['IVAR'].data
        lam_obs = drp_base['WAVE'].data
        specres_obs = drp_base['SPECRES'].data

        cubeshape = drp_base['FLUX'].data.shape
        nl_obs, *mapshape = cubeshape
        mapshape = tuple(mapshape)

        '''STEP 1'''
        # find SNR of each pixel in cube (used to scale noise later)
        snrcube_obs, rmsmap_obs = compute_snrcube(
            flux=flux_obs, ivar=ivar_obs,
            filtersize_l=15, return_rms_map=True)

        '''STEP 2'''
        # compute the redshift map
        z_cosm = row['nsa_zdist']
        z_pec = (dap_base['STELLAR_VEL'].data * u.Unit('km/s') / c.c).to('').value
        z_obs = (1. + z_cosm) * (1. + z_pec) - 1.

        # create a placeholder model cube since flexible broadcasting is hard
        spec_model_cube = np.tile(spec_model[:, None, None], (1, ) + mapshape)
        ivar_model_cube = np.ones_like(spec_model_cube)
        lam_model_z, spec_model_z, ivar_model_z = ut.redshift(
            l=lam_model, f=spec_model_cube, ivar=ivar_model_cube,
            z_in=0., z_out=z_obs)

        '''STEP 3'''
        spec_model_mwred, ivar_model_mwred = ut.extinction_atten(
            lam_model_z * u.AA, f=spec_model_z, ivar=ivar_model_z,
            EBV=drp_base[0].header['EBVGAL'])

        '''STEP 4'''
        # specres of observed cube at model wavelengths
        spec_model_instblur = ut.blur_cube_to_psf(
            l_ref=drp_base['WAVE'].data, specres_ref=drp_base['SPECRES'].data,
            l_eval=lam_model_z, spec_unblurred=spec_model_mwred)

        ivar_model_instblur = ivar_model_mwred

        '''STEP 5'''
        # create placeholder arrays for ivar and flux
        final_fluxcube = np.empty(cubeshape)

        # wavelength grid for final cube
        l_grid = drp_base['WAVE'].data

        # populate flam and ivar pixel-by-pixel
        for ind in np.ndindex(mapshape):
            final_fluxcube[:, ind[0], ind[1]] = np.interp(
                xp=lam_model_z[:, ind[0], ind[1]],
                fp=spec_model_instblur[:, ind[0], ind[1]],
                x=l_grid)
        # normalize each spectrum to mean 1
        final_fluxcube /= np.mean(final_fluxcube, axis=0, keepdims=True)

        '''STEP 6'''
        # spectrophotometric noise
        cov_noise = noisify_cov(Kspec_obs, mapshape=mapshape)
        # random noise: signal * (gauss / snr)
        random_noise = np.random.randn(*cubeshape) / snrcube_obs
        fluxscaled_random_noise = random_noise * final_fluxcube

        final_fluxcube += (cov_noise + fluxscaled_random_noise)

        '''STEP 7'''
        # normalize everything to have the same observed-frame r-band flux
        u_flam = 1.0e-17 * (u.erg / (u.s * u.cm**2 * u.AA))
        rband_drp = Spec2Phot(
            lam=drp_base['WAVE'].data,
            flam=drp_base['FLUX'].data * u_flam).ABmags['sdss2010-r']
        rband_model = Spec2Phot(
            lam=drp_base['WAVE'].data,
            flam=final_fluxcube * u_flam).ABmags['sdss2010-r']
        # flux ratio map
        r = 10.**(-0.4 * (rband_drp - rband_model))

        final_fluxcube *= r[None, ...]
        # initialize the ivar cube according to the SNR cube
        # of base observations
        # this is because while we think we know the actual spectral covariance,
        # that is not necessarily reflected in the quoted ivars!!!
        final_ivarcube = snrcube_obs**2. / final_fluxcube

        '''STEP 8'''
        # mask where the native datacube has no signal
        rimg = drp_base['RIMG'].data
        nosignal = (rimg == 0.)[None, ...]
        nosignal_cube = np.broadcast_to(nosignal, final_fluxcube.shape)
        final_fluxcube[nosignal_cube] = 0.
        final_ivarcube[final_fluxcube == 0.] = 0.

        # mask where there's bad velocity info
        badvel = m.mask_from_maskbits(
            dap_base['STELLAR_VEL_MASK'].data, [30])[None, ...]
        final_ivarcube[np.tile(badvel, (nl_obs, 1, 1))] = 0.

        # replace infinite flux elements with median-filtered
        flux_is_inf = ~np.isfinite(final_fluxcube)
        final_fluxcube[flux_is_inf] = medfilt(
            np.nan_to_num(final_fluxcube), [11, 1, 1])[flux_is_inf]

        self.dap_base = dap_base
        self.drp_base = drp_base
        self.fluxcube = final_fluxcube
        self.fluxcube_ivar = final_ivarcube
        self.row = row
        self.metadata = meta_model

        self.plateifu_base = plateifu_base

    @classmethod
    def from_FSPS(cls, fname, i, plateifu_base, pca, row, K_obs,
                  mpl_v='MPL-5', kind='SPX-GAU-MILESHC'):

        # load models
        models_hdulist = fits.open(fname)
        models_specs = models_hdulist['flam'].data
        models_lam = models_hdulist['lam'].data

        # restrict wavelength range to same as PCA
        lmin, lmax = pca.l.value.min(), pca.l.value.max()
        lmin, lmax = 3000., 10500.
        goodlam = (models_lam >= lmin) * (models_lam <= lmax)
        models_lam, models_specs = models_lam[goodlam], models_specs[:, goodlam]

        models_logl = np.log10(models_lam)

        models_meta = t.Table(models_hdulist['meta'].data)
        models_meta['tau_V mu'] = models_meta['tau_V'] * models_meta['mu']

        stellar_indices = get_stellar_indices(l=models_lam, spec=models_specs.T)
        models_meta = t.hstack([models_meta, stellar_indices])

        models_meta.keep_columns(pca.metadata.colnames)

        for n in models_meta.colnames:
            if pca.metadata[n].meta.get('scale', 'linear') == 'log':
                models_meta[n] = np.log10(models_meta[n])

        # choose specific model
        if i is None:
            # pick at random
            i = np.random.choice(len(models_meta))

        model_spec = models_specs[i, :]
        model_spec /= np.median(model_spec)
        model_meta = models_meta[i]

        # load data
        plate, ifu, *newparams = tuple(plateifu_base.split('-'))

        drp_base = m.load_drp_logcube(plate, ifu, mpl_v)
        dap_base = m.load_dap_maps(plate, ifu, mpl_v, kind)

        return cls(lam_model=models_lam, spec_model=model_spec,
                   meta_model=model_meta, row=row, plateifu_base=plateifu_base,
                   drp_base=drp_base, dap_base=dap_base, model_ix=i,
                   Kspec_obs=K_obs)

    def resample_spaxel(self, logl_in, flam_in, logl_out):
        '''
        resample the given spectrum to the specified logl grid
        '''

        interp = interp1d(x=logl_in, y=flam_in, kind='linear', bounds_error=False,
                          fill_value=0.)
        # 0. is a sentinel value, that tells us where to mask later
        return interp(logl_out)

    def write(self):
        '''
        write out fake LOGCUBE and DAP
        '''

        fname_base = self.plateifu_base
        basedir = 'fakedata'
        drp_fname = os.path.join(basedir, '{}_drp.fits'.format(fname_base))
        dap_fname = os.path.join(basedir, '{}_dap.fits'.format(fname_base))
        truthtable_fname = os.path.join(
            basedir, '{}_truth.tab'.format(fname_base))

        new_drp_cube = fits.HDUList([hdu for hdu in self.drp_base])
        new_drp_cube['FLUX'].data = self.fluxcube

        new_drp_cube['IVAR'].data = self.fluxcube_ivar

        new_dap_cube = fits.HDUList([hdu for hdu in self.dap_base])
        sig_a = np.sqrt(
            self.metadata['sigma']**2. * \
                np.ones_like(new_dap_cube['STELLAR_SIGMA'].data) + \
            new_dap_cube['STELLAR_SIGMACORR'].data**2.)
        new_dap_cube['STELLAR_SIGMA'].data = sig_a

        new_drp_cube[0].header['MODEL0'] = self.model_ix
        new_dap_cube[0].header['MODEL0'] = self.model_ix

        truth_tab = t.Table(
            rows=[self.metadata],
            names=self.metadata.colnames)
        truth_tab.write(truthtable_fname, overwrite=True, format='ascii')
        new_drp_cube.writeto(drp_fname, overwrite=True)
        new_dap_cube.writeto(dap_fname, overwrite=True)


def get_stellar_indices(l, spec):
    inds = t.Table(
        data=[t.Column(indices.StellarIndex(n)(l=l, flam=spec), n)
              for n in indices.data['ixname']])

    return inds
