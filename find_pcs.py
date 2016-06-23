import numpy as np
import matplotlib.pyplot as plt

import spec_tools
import ssp_lib

from astropy import constants as c, units as u
from itertools import izip

class StellarPop_PCA(object):
    '''
    class for determining PCs of a library of synthetic spectra
    '''
    def __init__(self, l, spectra, gen_dicts, metadata,
                 mask_half_dv=500.*u.Unit('km/s'), dlogl=None):
        '''
        params:
         - l: length-n array-like defining the wavelength bin centers
            (should be log-spaced)
         - spectra: m-by-n array of spectra (individual spectrum contained
            along one index in dimension 0), in units of 1e-17 erg/s/cm2/AA
         - gen_dicts: length-m list of FSPS_SFHBuilder.FSPS_args dicts,
            ordered the same as `spectra`
         - metadata: table of derived SSP properties used for regression
            (D4000 index, Hd_A index, r-band luminosity-weighted age,
             mass-weighted age, i-band mass-to-light ratio,
             z-band mass-to-light ratio, mass fraction formed in past 1Gyr,
             formation time, eftu, metallicity, tau_V, mu, sigma)
            this somewhat replicates data in `gen_dicts`, but that's ok
        '''

        self.l = l
        self.logl = np.log10(l.to('AA').value)
        if dlogl is None:
            dlogl = np.round(np.mean(logl[1:] - logl[:-1]), 8)
        self.dlogl = dlogl

        self.mask_half_dv = mask_half_dv

        self.spectra = spectra

    # =====
    # methods
    # =====

    def run_pca_models(self):
        '''
        run PCA on library of model spectra
        '''
        from sklearn.decomposition import PCA
        # first run some prep on the model spectra

        # find lower and upper edges of each wavelength bin,
        # and compute width of bins
        l_lower = 10.**(self.logl - self.dlogl/2.)
        l_upper = 10.**(self.logl + self.dlogl/2.)
        dl = self.dl = l_upper - l_lower

        # scale each spectrum such that the mean flux between
        # 3700 and 5000 AA is unity
        norm_flux = (self.spectra * dl)[:, (3700.*u.AA < self.l) * \
                                           (self.l < 5000.*u.AA)].sum(axis=1)
        self.normed_spectra = self.spectra/norm_flux[:, np.newaxis]
        self.mean_spectrum = np.mean(self.normed_spectra, axis=0)

        pca_models = PCA()
        pca_models.fit(self.spectra - self.mean_spectrum)
        [plt.plot(self.l, pca_models.components_[i] + i*.1, label=str(i),
                  drawstyle='steps-mid')
            for i in range(10)]
        plt.legend(loc='best', prop={'size': 8})
        plt.show()

    # =====
    # properties
    # =====

    @property
    def model_eline_mask(self):
        from astropy import units as u, constants as c
        half_dv = self.mask_half_dv
        line_ctrs = u.Unit('AA') * \
            np.array([3727.09, 3729.88, 3889.05, 3969.81, 3968.53,
        #              [OII]    [OII]      H8    [NeIII]  [NeIII]
                      4341.69, 4102.92, 4862.69, 5008.24, 4960.30])
        #                Hg       Hd       Hb     [OIII]   [OIII]

        # compute mask edges
        mask_ledges = line_ctrs * (1 - (half_dv / c.c).to(''))
        mask_uedges = line_ctrs * (1 + (half_dv / c.c).to(''))

        # find whether each wavelength bin is used in for each eline's mask
        full_antimask = np.row_stack(
            [~((lo < self.l) * (self.l < up))
                for lo, up in izip(mask_ledges, mask_uedges)])
        antimask = np.prod(full_antimask, axis=0)
        return ~antimask.astype(bool)

class MaNGA_deredshift(object):
    '''
    class to deredshift reduced MaNGA data based on velocity info from DAP

    preserves cube information, in general
    '''
    def __init__(self, drp_hdulist, dap_hdulist,
                 max_vel_unc=90.*u.Unit('km/s'), drp_logl=None):
        self.drp_hdulist = drp_hdulist
        self.dap_hdulist = dap_hdulist

        self.vel = dap_hdulist['STELLAR_VEL'].data * u.Unit('km/s')
        self.vel_ivar = dap_hdulist['STELLAR_VEL_IVAR'].data * u.Unit(
            'km-2s2')
        # mask all the spaxels that have high stellar velocity uncertainty
        self.vel_ivar_mask = 1./np.sqrt(vel_ivar) > max_vel_unc

        self.drp_logl = np.log10(drp_hdulist['WAVE'])
        if drp_dlogl is None:
            drp_dlogl = spec_tools.determine_dlogl(self.drp_logl)
        self.drp_dlogl = drp_dlogl

        self.flux = self.drp_hdulist['FLUX'].data

    @classmethod
    def from_filenames(cls, drp_fname, dap_fname):
        drp_hdulist = fits.open(drp_fname)
        dap_hdulist = fits.open(dap_fname)
        return cls(drp_hdulist, dap_hdulist)

    def regrid_to_rest(self, template_logl, template_dlogl=None):
        '''
        regrid flux density measurements from MaNGA DRP logcube results
            to a specified logl grid, essentially picking the pixels that
            fall in the logl grid's range, after being de-redshifted

        (this does not perform any fancy interpolation, just "shifting")
        (nor are emission line features masked--that must be done in post-)
        '''
        if template_dlogl is None:
            template_dlogl = spec_tools.determine_dlogl(template_logl)

        if template_dlogl != drp_dlogl:
            raise ssp_lib.TemplateCoverageError(
                'template and input spectra must have same dlogl: ' +\
                'template\'s is {}; input spectra\'s is {}'.format(
                    template_dlogl, self.drp_dlogl))

        # initialize a cube with spatial dimensions of DAP/DRP cube,
        # and spectral dimension of template wavelength array
        regridded_cube = np.zeros(self.vel.shape[:-1] + (len(filler),))

        # determine starting index for each of the spaxels

        template_logl0 = template_logl[0]
        template_logl0_z = template_logl0 + (self.vel/c.c).to('').value
        # find the index for the wavelength that best corresponds to
        # an appropriately redshifted wavelength grid
        ix_logl0_z = np.argmin((template_logl0_z - drp_logl)**2.,
                               axis=-1)
        # test whether wavelength grid extends beyond MaNGA coverage
        # in any spaxels
        bad_logl_extent = (ix_logl0_z + len(template_logl)) >= len(drp_logl)
        bad_ = (bad_logl_extent | self.vel_ivar_mask)[:, :, np.newaxis]
        # select len(template_logl) values from self.flux, w/ diff starting
        # (see http://stackoverflow.com/questions/37984214/
        # pure-numpy-expression-for-selecting-same-length-
        # subarrays-with-different-startin)
        I, J, _ = np.ix_(*[range(i) for i in self.flux.shape])
        # (and also filter out spaxels that are bad)
        regridded_cube[~bad_] = self.flux[
            I, J, ix_logl0_z[..., None] + np.arange(
            len(template_logl))][~bad_]

        self.bad_ = bad_

        self.regridded_cube = regridded_cube

        return self.regridded_cube, self.bad_
