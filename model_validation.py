import numpy as np
import matplotlib.pyplot as plt

from astropy import table as t
from statsmodels.nonparametric.kernel_density import KDEMultivariate

import os
import sys

# local
import find_pcs

# add manga RC location to path, and import config
if os.environ['MANGA_CONFIG_LOC'] not in sys.path:
    sys.path.append(os.environ['MANGA_CONFIG_LOC'])

import mangarc

if mangarc.tools_loc not in sys.path:
    sys.path.append(mangarc.tools_loc)

# personal
import manga_tools as m
import spec_tools

eps = np.finfo(float).eps


class ModelDataCompare(object):
    """
    Compare the model set and the data in a bunch of spaces
    """
    def __init__(self, pca, dereds):
        self.pca = pca
        self.dereds = dereds

        for p in ['Hdelta_A', 'D4000']:
            # retrieve appropriate function from spec_tools
            f = getattr(spec_tools, '{}_index'.format(p))

            # compute index
            setattr(self, p,
                    [f(l=self.pca.l,
                       s=d.regrid_to_rest(template_logl=self.pca.logl,
                                          template_dlogl=self.pca.dlogl))[0]
                     for d in self.dereds])

    def D4000_Hd_fig(self):
        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = fig.add_subplot(111)

        KDE_data = KDEMultivariate(data=[self.D4000.flatten(),
                                         self.Hdelta_A.flatten()],
                                   var_type='cc')

        KDE_models = KDEMultivariate(data=[self.pca.metadata['D4000'],
                                           self.pca.metadata['Hdelta_A']],
                                     var_type='cc')

        nx, ny = 100, 100

        XX, YY = np.meshgrid(np.linspace(1., 2.5, nx),
                             np.linspace(-6., 12., ny))

        XXYY = np.column_stack((XX.flatten(), YY.flatten()))

        ZZ_data = KDE_data.pdf(XXYY).reshape((nx, ny))
        ZZ_models = KDE_models.pdf(XXYY).reshape((nx, ny))

        ax.contour(XX, YY, ZZ_data, c='r', label='MaNGA data')
        ax.contour(XX, YY, ZZ_models, c='b', label='Models')

        ax.legend(loc='best')

        ax.set_xlabel('D4000')
        ax.set_ylabel(r'D$\delta_{A}$')

        return fig
