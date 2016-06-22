import numpy as np
import matplotlib.pyplot as plt

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
        [plt.plot(self.l, pca_models.components_[i] + i*.1, label=str(i))
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
