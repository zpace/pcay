import numpy as np
from astropy import constants as c, units as u

class StellarPop_PCA(object):
    '''
    class for determining PCs of a library of synthetic spectra
    '''
    def __init__(self, l, spectra, dlogl=None):
        '''
        params:
         - l: length-n array-like defining the wavelength bin centers
            (should be log-spaced)
         - spectra: m-by-n array of spectra (individual spectrum contained
            along one index in dimension 0)
        '''
        self.l = l
        if dlogl is None:
            dlogl = np.mean(l[1:]/l[:-1]) - 1.
        self.dlogl = dlogl
        self.spectra = spectra

        self.mean_spectrum = np.mean(self.spectra, axis=0)

    @property
    def eline_masks(self, half_dv=500.*u.Unit('km/s')):
        from itertools import izip
        line_ctrs = u.AA * \
            np.array([3727.09, 3729.88, 3889.05, 3969.81, 3968.53,
        #              [OII]    [OII]      H8    [NeIII]  [NeIII]
                      4341.69, 4102.92, 4862.69, 5008.24, 4960.30])
        #                Hg       Hd       Hb     [OIII]   [OIII]

        # compute mask edges
        mask_ledges = line_ctrs * (1 - (half_dv / c.c).to(''))
        mask_uedges = line_ctrs * (1 + (half_dv / c.c).to(''))

        # find whether each wavelength bin is used in for each eline's mask
        full_antimask = np.row_stack(
            [~((mask_ledges < self.l) * (self.l < mask_uedges))
                for l, u in izip(mask_ledges, mask_uedges)])
        antimask = np.prod(full_antimask, axis=0)
        return antimask.astype(bool)
