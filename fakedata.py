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

def noisify_cov(self, cov, mapshape, i0, nl):
    assert cov is not None, 'covariance object not passed'

    cov_noise = np.random.multivariate_normalrvs(
        mean=np.zeros_like(np.diag(cov.cov)),
        cov=cov.cov, size=mapshape)
    cov_noise = np.moveaxis(cov_noise, [0, 1, 2], [2, 0, 1])
    cov_noise = cov_noise[i0:i0 + nl, :, :]
    rms = np.sqrt(np.mean(cov_noise, axis=0))
    return cov_noise, rms

class FakeData(object):
    '''
    Make a fake IFU and fake DAP stuff
    '''
    def __init__(self, l_model, s_model, meta_model,
                 row, drp_base, dap_base, plateifu_base, model_ix,
                 noisify_method='cov', Kspec_obs=None):

        self.model_ix = model_ix

        # find SNR of each pixel in cube (used to scale noise later)
        drp_snr = medfilt(drp_base['FLUX'].data * np.sqrt(drp_base['IVAR'].data),
                          [15, 1, 1])
        spax_rms = np.mean(drp_snr, axis=0)

        # redshift the model spectrum
        z_cosm = row['nsa_zdist']
        z_pec = (dap_base['STELLAR_VEL'].data * u.Unit('km/s') / c.c).to('').value
        z_out = (1. + z_cosm) * (1. + z_pec) - 1.

        # instrumental blurring
        # interpolate specres to model wavelength grid
        specres = interp1d(
            x=drp_base['WAVE'].data, y=drp_base['SPECRES'].data,
            bounds_error=False, fill_value='extrapolate')(l_model)
        velres = 1. / specres * c.c
        dlogl_model = ut.determine_dlogl(np.log10(l_model))
        vpix = dlogl_model * np.log(10.) * c.c
        # we're blurring rest frame: observed frame is a narrower blur
        # (Cappellari 2017)
        blur_npix = ((velres / vpix).to('').value) / (1. + z_cosm)
        s_model_blur = ut.gaussian_filter(s_model, blur_npix)

        # make cube
        cubeshape = drp_base['FLUX'].data.shape
        nl, *mapshape = cubeshape
        mapshape = tuple(mapshape)

        s_norm = s_model_blur.max()
        f = np.tile(s_model_blur[..., None, None] / s_norm, (1, ) + mapshape)

        # redshift model cube
        l_m_z, s_m_z, _ = ut.redshift(l=l_model, f=f, ivar=np.ones_like(f),
                                      z_in=0., z_out=z_out)
        logl_m_z = np.log10(l_m_z)

        # redden model cube according to galactic E(B-V)
        s_m_z = ut.extinction_atten(
            l=l_m_z * u.AA, f=s_m_z, EBV=drp_base[0].header['EBVGAL'])

        # interpolate redshifted model cube
        logl_f = np.log10(drp_base['WAVE'].data)
        s_m_z_c = np.zeros(cubeshape)

        for ind in np.ndindex(mapshape):
            newspec = self.resample_spaxel(
                logl_in=logl_m_z[:, ind[0], ind[1]],
                flam_in=s_m_z[:, ind[0], ind[1]], logl_out=logl_f)
            s_m_z_c[:, ind[0], ind[1]] = newspec

        # add error vector based on SNR calculated above
        # SNR = F sqrt(I) ===> (SNR / F)^2 = I

        if noisify_method == 'cov':
            # assume cosmological redshift
            # key onto **final** value in model wavelengths
            ifinal = np.argmin(np.abs(l_model[-1] * (1. + z_cosm) - cov.l))
            i0 = ifinal - nl
            noise, noise_rms = noisify_cov(
                Kspec_obs, mapshape=mapshape, i0=i0, nl=nl)
        elif noisify_method == 'ivar':
            noise = np.random.randn(*cubeshape) / drp_snr
            noise_rms = np.sqrt(np.mean(noise**2., axis=0))

        noise_rmsscaled = noise * (spax_rms / noise_rms)[None, :, :]
        fakecube = s_m_z_c * (1. + noise_rmsscaled)

        # normalize everything to have the same observed-frame r-band flux
        u_flam = 1.0e-17 * (u.erg / (u.s * u.cm**2 * u.AA))
        m_r_drp = Spec2Phot(lam=drp_base['WAVE'].data,
                            flam=drp_base['FLUX'].data * u_flam).ABmags['sdss2010-r']
        m_r_mzc = Spec2Phot(lam=drp_base['WAVE'].data,
                            flam=s_m_z_c * u_flam).ABmags['sdss2010-r']
        # flux ratio map
        r = 10.**(-0.4 * (m_r_drp - m_r_mzc))
        # occasionally this produces inf or nan for some reason,
        # so just take care of that quickly
        r[np.log10(r) > 4.] = 4.
        r[np.log10(r) < -4] = -4.
        fakecube *= r[None, ...]

        ivar_cube = (noise_rmsscaled * r)**2.

        # mask where the native datacube has no signal
        rimg = drp_base['RIMG'].data
        nosignal = (rimg == 0.)[None, ...]
        nosignal_cube = np.broadcast_to(nosignal, fakecube.shape)
        fakecube[nosignal_cube] = 0.
        ivar_obs[s_m_z_c == 0.] = 0.

        # mask where there's bad velocity info
        badvel = m.mask_from_maskbits(
            dap_base['STELLAR_VEL_MASK'].data, [30])[None, ...]

        # logical_or.reduce doesn't work (different size arrays?)
        bad = np.logical_or(~np.isfinite(fakecube), badvel)
        fakecube[bad] = 0.

        self.dap_base = dap_base
        self.drp_base = drp_base
        self.fluxcube = fakecube
        self.fluxcube_ivar = ivar_obs
        self.row = row
        self.metadata = meta_model

        self.plateifu_base = plateifu_base

    @classmethod
    def from_FSPS(cls, fname, i, plateifu_base, pca, row,
                  mpl_v='MPL-5', kind='SPX-GAU-MILESHC'):

        # load models
        models_hdulist = fits.open(fname)
        models_specs = models_hdulist['flam'].data
        models_lam = models_hdulist['lam'].data

        # restrict wavelength range to same as PCA
        goodlam = (models_lam >= pca.l.value.min()) * \
                  (models_lam <= pca.l.value.max())
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

        return cls(l_model=models_lam, s_model=model_spec,
                   meta_model=model_meta, row=row, plateifu_base=plateifu_base,
                   drp_base=drp_base, dap_base=dap_base, model_ix=i,
                   Kspec_obs=pca.Kspec_obs)

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
