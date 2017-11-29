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

# add manga RC location to path, and import config
if os.environ['MANGA_CONFIG_LOC'] not in sys.path:
    sys.path.append(os.environ['MANGA_CONFIG_LOC'])

import mangarc

if mangarc.tools_loc not in sys.path:
    sys.path.append(mangarc.tools_loc)

# personal
import manga_tools as m

def noisify_cov(cov, mapshape):
    assert cov is not None, 'covariance object not passed'

    cov_noise = np.random.multivariate_normal(
        mean=np.zeros_like(np.diag(cov.cov)),
        cov=cov.cov, size=mapshape)
    cov_noise = np.moveaxis(cov_noise, [0, 1, 2], [1, 2, 0])
    #cov_noise = cov_noise[i0:i0 + nl, :, :]
    rms = np.sqrt(np.mean(cov_noise, axis=0))
    print(cov_noise.shape, rms.shape)
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
                 noisify_method='cov', Kspec_obs=None):
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

        '''STEP 1'''
        # find SNR of each pixel in cube (used to scale noise later)
        drp_snr, spax_rms = compute_snrcube(
            flux=drp_base['FLUX'].data, ivar=drp_base['IVAR'].data,
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
        specres_obs = interp1d(
            x=drp_base['WAVE'].data, y=drp_base['SPECRES'].data,
            bounds_error=False, fill_value='extrapolate')(lam_model_z)
        # convert specres of observations into dlnl
        dlnl_obs = 1. / specres_obs

        # dlogl of a pixel in model
        dloglcube_model = ut.determine_dloglcube(np.log10(lam_model_z))
        # convert dlogl of pixel in model to dlnl
        dlnlcube_model = dloglcube_model * np.log(10.)
        # number of pixels is dlnl of obs div by dlnl of model
        specres_pix = dlnl_obs[:, None, None] / dlnlcube_model

        # create placeholder for instrumental-blurred model
        spec_model_instblur = np.empty_like()

        # populate pixel-by-pixel (should take < 15 sec)
        for ind in np.ndindex(mapshape):
            spec_model_instblur[:, ind[0], ind[1]] = ut.gaussian_filter(
                spec=spec_model_mwred[:, ind[0], ind[1]],
                sig=specres_pix[:, ind[0], ind[1]])

        ivar_model_instblur = ivar_model_mwred

        '''STEP 5'''
        # create placeholder arrays for ivar and flux
        spec_model_rect = np.zeros(cubeshape)
        ivar_model_rect = np.zeros(cubeshape)

        # wavelength grid for final cube
        l_grid = drp_base['WAVE'].data

        # populate flam and ivar pixel-by-pixel
        for ind in np.ndindex(mapshape):
            final_fluxcube[:, ind[0], ind[1]] = np.interp(
                xp=lam_model_z[:, ind[0], ind[1]],
                fp=spec_model_instblur[:, ind[0], ind[1]],
                x=l_grid)
            final_ivarcube[:, ind[0], ind[1]] = np.interp(
                xp=lam_model_z[:, ind[0], ind[1]],
                fp=ivar_model_instblur[:, ind[0], ind[1]],
                x=l_grid)

        '''STEP 6'''
        if noisify_method == 'cov':
            # assume cosmological redshift
            # key onto **final** value in model wavelengths
            ifinal = np.argmin(np.abs(l_model[-1] * (1. + z_cosm) - Kspec_obs.l))
            i0 = ifinal - nl
            noise, noise_rms = noisify_cov(
                Kspec_obs, mapshape=mapshape)
        elif noisify_method == 'ivar':
            noise = np.random.randn(*cubeshape) / drp_snr
            noise_rms = np.sqrt(np.mean(noise**2., axis=0))

        # scale noise
        noise_rmsscaled = noise * (spax_rms / noise_rms)[None, :, :]
        # add noise
        final_fluxcube += noise_rmsscaled

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
        r = 10.**(-0.4 * (m_r_drp - m_r_mzc))
        # occasionally this produces inf or nan for some reason,
        # so just take care of that quickly
        r[np.log10(r) > 4.] = 4.
        r[np.log10(r) < -4] = -4.

        final_fluxcube *= r[None, ...]
        final_ivarcube *= (noise_rmsscaled * r)**2.

        '''STEP 8'''
        # mask where the native datacube has no signal
        rimg = drp_base['RIMG'].data
        nosignal = (rimg == 0.)[None, ...]
        nosignal_cube = np.broadcast_to(nosignal, final_fluxcube.shape)
        final_fluxcube[nosignal_cube] = 0.
        final_ivarobs[final_fluxcube == 0.] = 0.

        # mask where there's bad velocity info
        badvel = m.mask_from_maskbits(
            dap_base['STELLAR_VEL_MASK'].data, [30])[None, ...]

        # logical_or.reduce doesn't work (different size arrays?)
        bad = np.logical_or(~np.isfinite(final_fluxcube), badvel)
        fakecube[bad] = 0.

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
