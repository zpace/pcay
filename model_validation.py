import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm

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
    def __init__(self, pca, dereds, rimgs, Hd_em_EWs, deps, rimg_thr=0.):
        self.pca = pca
        self.dereds = dereds
        self.rimgs = rimgs

        self.Hd_em_EWs = np.concatenate(
            [h.flatten() for h in Hd_em_EWs])

        self.mask = np.concatenate(
            [(r <= rimg_thr).flatten() for r in rimgs])

        self.rimgs = np.concatenate(
            [r.flatten() for r in rimgs])

        self.dep_d = np.concatenate(
            [dep.d.flatten() for dep in deps])

        for p in ['Hdelta_A', 'Dn4000']:
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

        data = np.column_stack(
            [np.concatenate(self.Dn4000),
             np.concatenate(self.Hdelta_A) + self.Hd_em_EWs])
        data = data[~self.mask]

        KDE_data = KDEMultivariate(data=data, var_type='cc')

        KDE_models = KDEMultivariate(data=[self.pca.metadata['Dn4000'],
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

        ax.set_xlabel(self.pca.metadata['Dn4000'].meta['TeX'])
        ax.set_ylabel(self.pca.metadata['Hdelta_A'].meta['TeX'])

        plt.tight_layout()

        return fig

    def D4000_Hd_scatter_fig(self):
        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = fig.add_subplot(111)

        data = np.column_stack(
            [np.concatenate(self.Dn4000),
             np.concatenate(self.Hdelta_A) + self.Hd_em_EWs])
        data = data[~self.mask]

        KDE_models = KDEMultivariate(data=[self.pca.metadata['D4000'],
                                           self.pca.metadata['Hdelta_A']],
                                     var_type='cc')

        nx, ny = 100, 100

        XX, YY = np.meshgrid(np.linspace(1., 2.5, nx),
                             np.linspace(-6., 12., ny))

        XXYY = np.column_stack((XX.flatten(), YY.flatten()))

        ZZ_models = KDE_models.pdf(XXYY).reshape((nx, ny))

        ax.contour(XX, YY, ZZ_models, colors='b')

        # define alpha-less color
        d = self.dep_d[~self.mask]
        c = mplcm.viridis(d)

        # make a dummy plot to reflect color dist, at alpha=1
        imin, imax = np.argmin(d), np.argmax(d)
        dmin, dmax = d[imin], d[imax]
        c_plt = ax.scatter([0., 1.], [20., 20.], c=[dmin, dmax],
                           s=3, alpha=1., cmap='viridis')

        sb = self.rimgs[~self.mask]
        # e-folding SB decrement
        efsbd = 0.5
        c[:, -1] = np.exp(-sb / (efsbd * sb.max()))

        r_plt = ax.scatter(data[:, 0], data[:, 1], c=c,
                           edgecolor='None', s=3,
                           cmap='viridis', label='MaNGA data')
        cb = plt.colorbar(c_plt)

        ax.plot([0., 1.], [20., 20.], c='b', label='CSP models')

        ax.legend(loc='best', prop={'size': 6})

        ax.set_ylim([-6., 12.])
        ax.set_xlim([1., 2.5])

        ax.set_xlabel(self.pca.metadata['D4000'].meta['TeX'])
        ax.set_ylabel(self.pca.metadata['Hdelta_A'].meta['TeX'])

        plt.tight_layout()

        return fig


if __name__ == '__main__':

    from find_pcs import *

    mpl_v = 'MPL-5'

    pca, K_obs = setup_pca(fname='pca.pkl', redo=False, pkl=True, q=7)

    gals = ['8566-12705']
    ''', '8567-12701', '8939-12704', '8083-12704',
            '8134-12702', '8134-9102', '8135-12701', '8137-12703',
            '8140-12703', '8140-12701', '8140-3701', '8243-12704',
            '8244-9101', '8247-9101', '8249-12703', '8249-12704',
            '8252-12705', '8252-6102', '8253-6104', '8254-3704', '8254-9101',
            '8257-12701', '8257-6101', '8257-6103', '8258-12704']'''

    #gals = ['8567-12701']

    p_i = [g.split('-') for g in gals]

    dereds = [MaNGA_deredshift.from_plateifu(
        plate=int(p), ifu=int(i), MPL_v=mpl_v) for (p, i) in p_i]
    rimgs = [d.drp_hdulist['RIMG'].data for d in dereds]
    Hdelt_em_EWs = [d.dap_hdulist['EMLINE_SEW'].data[14, ...] for d in dereds]
    deps = [m.deproject.from_plateifu(plate=p, ifu=i, MPL_v='MPL-5')
            for (p, i) in p_i]

    MDComp = ModelDataCompare(pca=pca, dereds=dereds, rimgs=rimgs,
                              Hd_em_EWs=Hdelt_em_EWs, deps=deps)
    mdcomp_fig = MDComp.D4000_Hd_scatter_fig()
    mdcomp_fig.savefig('D4000-HdA_scatter.png', dpi=300)
