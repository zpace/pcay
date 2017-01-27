import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm

from astropy import table as t
from astropy.cosmology import WMAP9 as cosmo
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
    def __init__(self, pca, dereds, rimgs, Hd_em_EWs, deps, K_obs,
                 rimg_thr=0.):

        self.pca = pca
        self.dereds = dereds
        self.rimgs = rimgs

        self.Hd_em_EWs = np.concatenate(
            [h.flatten() for h in Hd_em_EWs])

        self.rimgs = np.concatenate(
            [r.flatten() for r in rimgs])

        self.Reff = np.concatenate([d.Reff.flatten() for d in dereds])

        for p in ['Hdelta_A', 'Dn4000']:
            # retrieve appropriate function from spec_tools
            f = getattr(spec_tools, '{}_index'.format(p))

            # compute index
            setattr(
                self, p,
                np.concatenate(
                    [f(l=self.pca.l.value,
                       s=d.correct_and_match(
                       template_logl=self.pca.logl,
                       template_dlogl=self.pca.dlogl)[0]).flatten()
                     for d in self.dereds]))

        PCA_Results = [
            find_pcs.PCA_Result(
                pca=pca, dered=dered, K_obs=K_obs,
                z=dered.drpall_row['nsa_zdist'][0], cosmo=cosmo,
                norm_params={'norm': 'L2', 'soft': False}, figdir='.')
            for dered in dereds]

        self.Dn4000_infer = np.concatenate(
            [(r.pca.param_cred_intvl(
                  qty='Dn4000', W=r.w)[0]).flatten()
             for r in PCA_Results])

        self.Hdelta_A_infer = np.concatenate(
            [(r.pca.param_cred_intvl(
                  qty='Hdelta_A', W=r.w)[0]).flatten()
             for r in PCA_Results])

        self.infer_badPDF = np.concatenate(
            [r.badPDF.flatten() for r in PCA_Results])

        self.mask = np.logical_or.reduce(((self.rimgs < rimg_thr),
                                          ~np.isfinite(self.Dn4000),
                                          ~np.isfinite(self.Hdelta_A)))

    def D4000_Hd_fig(self):
        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = fig.add_subplot(111)

        # directly observed indices
        data = np.column_stack(
            [self.Dn4000, self.Hdelta_A + self.Hd_em_EWs])
        data = data[~self.mask]

        # inferred indices
        infer = np.column_stack(
            [self.Dn4000_infer, self.Hdelta_A_infer, self.infer_badPDF])
        infer_good = infer[(~self.mask) * (~self.infer_badPDF)]
        infer_bad = infer[(~self.mask) * (self.infer_badPDF)]

        KDE_data = KDEMultivariate(data=data, var_type='cc')
        KDE_models = KDEMultivariate(data=[self.pca.metadata['Dn4000'],
                                           self.pca.metadata['Hdelta_A']],
                                     var_type='cc')
        KDE_infer_good = KDEMultivariate(
            data=infer_good[:, :-1],
            var_type='cc')
        KDE_infer_bad = KDEMultivariate(
            data=infer_bad[:, :-1],
            var_type='cc')

        nx, ny = 200, 200

        XX, YY = np.meshgrid(np.linspace(1., 2.5, nx),
                             np.linspace(-6., 12., ny))

        XXYY = np.column_stack((XX.flatten(), YY.flatten()))

        ZZ_data = KDE_data.pdf(XXYY).reshape((nx, ny))
        ZZ_models = KDE_models.pdf(XXYY).reshape((nx, ny))
        ZZ_infer_bad = KDE_infer_bad.pdf(XXYY).reshape((nx, ny))
        ZZ_infer_good = KDE_infer_good.pdf(XXYY).reshape((nx, ny))

        ax.contour(XX, YY, ZZ_data, colors='k')
        ax.contour(XX, YY, ZZ_models, colors='b')
        ax.contour(XX, YY, ZZ_infer_bad, colors='r')
        ax.contour(XX, YY, ZZ_infer_good, colors='g')

        ax.plot([0., 1.], [20., 20.], c='k', label='MaNGA: direct')
        ax.plot([0., 1.], [20., 20.], c='b', label='CSP models')
        ax.plot([0., 1.], [20., 20.], c='r', label='MaNGA: inferred (bad)')
        ax.plot([0., 1.], [20., 20.], c='g', label='MaNGA: inferred (good)')

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
            [self.Dn4000, self.Hdelta_A + self.Hd_em_EWs])
        data = data[~self.mask]

        infer = np.column_stack(
            [self.Dn4000_infer, self.Hdelta_A_infer, self.infer_badPDF])
        infer = infer[~self.mask]

        KDE_models = KDEMultivariate(data=[self.pca.metadata['Dn4000'],
                                           self.pca.metadata['Hdelta_A']],
                                     var_type='cc')

        nx, ny = 200, 200

        XX, YY = np.meshgrid(np.linspace(1., 2.5, nx),
                             np.linspace(-6., 12., ny))

        XXYY = np.column_stack((XX.flatten(), YY.flatten()))

        ZZ_models = KDE_models.pdf(XXYY).reshape((nx, ny))

        ax.contour(XX, YY, ZZ_models, colors='b', zorder=1)

        # raw index measurements
        ax.scatter(data[:, 0], data[:, 1], facecolor='k',
                   edgecolor='None', s=.5, alpha=.5, label='MaNGA: direct')

        # inferred index measurements
        # plot bad PDF data in red, good in green
        ax.scatter(infer[:, 0], infer[:, 1], facecolor=infer[:, 2],
                   edgecolor='None', s=.5, cmap='RdYlGn_r', alpha=.5,
                   vmin=0., vmax=1., label='MaNGA: inferred')

        ax.plot([0., 1.], [20., 20.], c='b', label='CSP models')

        ax.legend(loc='best', prop={'size': 6})

        ax.set_ylim([-6., 12.])
        ax.set_xlim([1., 2.5])

        ax.set_xlabel(self.pca.metadata['Dn4000'].meta['TeX'])
        ax.set_ylabel(self.pca.metadata['Hdelta_A'].meta['TeX'])

        plt.tight_layout()

        return fig


if __name__ == '__main__':

    from find_pcs import *

    mpl_v = 'MPL-5'

    pca, K_obs = setup_pca(
        fname='pca.pkl', base_dir='CSPs_new', base_fname='CSPs',
        redo=True, pkl=True, q=8, src='FSPS', nfiles=None)

    gals = ['8566-12705', '8567-12701', '8939-12704', '8083-12704']
    '''        '8134-12702', '8134-9102', '8135-12701', '8137-12703',
            '8140-12703', '8140-12701', '8140-3701', '8243-12704']
            '8244-9101', '8247-9101', '8249-12703', '8249-12704',
            '8252-12705', '8252-6102', '8253-6104', '8254-3704', '8254-9101',
            '8257-12701', '8257-6101', '8257-6103', '8258-12704']'''

    #gals = ['8253-6104']

    p_i = [g.split('-') for g in gals]

    dereds = [MaNGA_deredshift.from_plateifu(
        plate=int(p), ifu=int(i), MPL_v=mpl_v) for (p, i) in p_i]
    rimgs = [d.drp_hdulist['RIMG'].data for d in dereds]
    Hdelt_em_EWs = [d.dap_hdulist['EMLINE_SEW'].data[14, ...] for d in dereds]
    deps = [m.deproject.from_plateifu(plate=p, ifu=i, MPL_v='MPL-5')
            for (p, i) in p_i]

    MDComp = ModelDataCompare(pca=pca, dereds=dereds, rimgs=rimgs,
                              Hd_em_EWs=Hdelt_em_EWs, deps=deps, K_obs=K_obs)
    mdcomp_fig = MDComp.D4000_Hd_scatter_fig()
    mdcomp_fig.savefig('D4000-HdA_scatter.png', dpi=300)

    mdcont_fig = MDComp.D4000_Hd_fig()
    mdcont_fig.savefig('D4000-HdA_contours.png', dpi=300)
