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
    def __init__(self, pca, dereds, rimgs, rimg_thr=0.):
        self.pca = pca
        self.dereds = dereds
        self.rimgs = rimgs

        self.mask = np.concatenate(
            [(r <= rimg_thr).flatten() for r in rimgs])

        for p in ['Hdelta_A', 'D4000']:
            # retrieve appropriate function from spec_tools
            f = getattr(spec_tools, '{}_index'.format(p))

            # compute index
            setattr(
                self, p,
                [f(l=self.pca.l.value,
                   s=d.regrid_to_rest(
                       template_logl=self.pca.logl,
                       template_dlogl=self.pca.dlogl)[0]).flatten()
                 for d, i in zip(self.dereds, self.rimgs)])

    def D4000_Hd_fig(self):
        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = fig.add_subplot(111)

        data = np.column_stack([np.concatenate(self.D4000),
                                np.concatenate(self.Hdelta_A)])
        data = data[~self.mask]

        KDE_data = KDEMultivariate(data=data, var_type='cc')

        KDE_models = KDEMultivariate(data=[self.pca.metadata['D4000'],
                                           self.pca.metadata['Hdelta_A']],
                                     var_type='cc')

        nx, ny = 100, 100

        XX, YY = np.meshgrid(np.linspace(1., 2.5, nx),
                             np.linspace(-6., 12., ny))

        XXYY = np.column_stack((XX.flatten(), YY.flatten()))

        ZZ_data = KDE_data.pdf(XXYY).reshape((nx, ny))
        ZZ_models = KDE_models.pdf(XXYY).reshape((nx, ny))

        ax.contour(XX, YY, ZZ_data, colors='r')
        ax.contour(XX, YY, ZZ_models, colors='b')

        ax.plot([0., 1.], [20., 20.], c='r', label='MaNGA data')
        ax.plot([0., 1.], [20., 20.], c='b', label='CSP models')

        ax.legend(loc='best', prop={'size': 6})

        ax.set_ylim([-6., 12.])
        ax.set_xlim([1., 2.5])

        ax.set_xlabel('D4000')
        ax.set_ylabel(r'H$\delta_{A}$')

        plt.tight_layout()

        return fig


if __name__ == '__main__':

    from find_pcs import *

    mpl_v = 'MPL-5'

    pca, K_obs = setup_pca(fname='pca.pkl', redo=False, pkl=True, q=7)

    gals = ['8566-12705', '8567-12701', '8939-12704', '8083-12704',
            '8134-12702', '8134-9102', '8135-12701', '8137-12703',
            '8140-12703', '8140-12701', '8140-3701', '8243-12704']

    p_i = [g.split('-') for g in gals]

    dereds = [MaNGA_deredshift.from_plateifu(
        plate=int(p), ifu=int(i), MPL_v=mpl_v) for (p, i) in p_i]
    rimgs = [d.drp_hdulist['RIMG'].data for d in dereds]

    MDComp = ModelDataCompare(pca=pca, dereds=dereds, rimgs=rimgs)
    mdcomp_fig = MDComp.D4000_Hd_fig()
    mdcomp_fig.savefig('D4000-HdA.png', dpi=300)
