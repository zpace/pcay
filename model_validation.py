import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
from matplotlib import ticker as mticker
import mpl_scatter_density

from astropy import table as t
from astropy.cosmology import WMAP9 as cosmo
from astropy.visualization import hist as ahist
from astropy.stats import sigma_clip
from statsmodels.nonparametric.kernel_density import KDEMultivariate

import os
import sys

# local
import find_pcs
import figures_tools

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
    def __init__(self, pca, dereds, rimgs, Hd_em_EWs, K_obs,
                 rimg_thr=0.):

        self.pca = pca
        self.dereds = dereds
        self.rimgs = rimgs

        self.Hd_em_EWs = np.concatenate(
            [h.flatten() for h in Hd_em_EWs])

        self.rimgs = np.concatenate(
            [r.flatten() for r in rimgs])

        self.Reff = np.concatenate([d.Reff.flatten() for d in dereds])

        PCA_Results = [
            find_pcs.PCA_Result(
                pca=pca, dered=dered, K_obs=K_obs,
                z=dered.drpall_row['nsa_zdist'], cosmo=cosmo, figdir='.')
            for dered in dereds]

        self.dir_meas = {}
        self.ind_meas = {}

        for p in ['Hdelta_A', 'Dn4000', 'Mg_b', 'Na_D', 'Ca_HK']:
            # retrieve appropriate function from spec_tools
            f = indices.StellarIndex(p)

            self.dir_meas[p] = np.concatenate(
                [f(self.pca.l.value, *d.correct_and_match(
                     template_logl=self.pca.logl,
                     template_dlogl=self.pca.dlogl)[:-1]).flatten()
                 for d in self.dereds])

            self.ind_meas[p] = np.concatenate(
                [(r.pca.param_cred_intvl(qty=p, W=r.w)[0]).flatten()
                 for r in PCA_Results])

        self.infer_badPDF = np.concatenate(
            [r.badPDF.flatten() for r in PCA_Results])

        self.mask = np.logical_or.reduce(
            ((self.rimgs <= rimg_thr), ) + \
            tuple(~self._in_param_range(self.dir_meas[k], k)
                  for k in self.dir_meas),
            tuple(pca_r.SNR_med < 10 for pca_r in PCA_Results))

    def _in_param_range(self, a, n, nstd=10.):
        '''
        construct mask for array y where it's in a reasonable range of the
            training data
        '''

        p = self.pca.metadata[n]
        m, s = p.mean(), p.std()

        return (a >= m - nstd * s) * (a <= m + nstd * s)

    def _hist(self, a, minmax, ax, bad=None, **kwargs):
        if bad is not None:
            a = a[~bad]

        h_, edges = np.histogram(a, bins=50, range=minmax)
        ctrs = 0.5 * (edges[:-1] + edges[1:])
        h = ax.plot(ctrs, h_ / h_.max(), drawstyle='steps-mid',
                    linewidth=0.5, **kwargs)

    def param_compare(self):

        names = self.ind_meas.keys()
        nparams = len(names)

        gs, fig = figures_tools.gen_gridspec_fig(N=nparams)

        for i, n in enumerate(names):
            # set up subplots
            ax = fig.add_subplot(gs[i])

            p = self.pca.metadata[n]
            m, s = p.mean(), p.std()
            minmax = (m - 4. * s, m + 4. * s)

            bad = np.logical_or(self.infer_badPDF, self.mask)

            self._hist(self.dir_meas[n], minmax, ax, bad=bad, color='k', label='MaNGA: direct')
            self._hist(self.ind_meas[n], minmax, ax, bad=bad, color='g', label='MaNGA: inferred')
            self._hist(p, minmax, ax, color='b', label='CSP models')

            locx = mticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10])
            locy = mticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10])
            locy_ = mticker.NullLocator()
            ax.xaxis.set_major_locator(locx)
            ax.yaxis.set_major_locator(locy)

            ax.tick_params(axis='both', color='k', labelsize=6)
            ax.text(x=np.percentile(ax.get_xlim(), 10), y=0.85,
                    s=self.pca.metadata[n].meta['TeX'], size=6)

            ax.set_ylim([-.02, 1.1])

            if i == 0:
                ax.legend(loc='best', prop={'size': 4})

        plt.suptitle('Lick Indices')

        return fig

    def D4000_Hd_fig(self):
        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = fig.add_subplot(111, projection='scatter_density')

        # directly observed indices
        direct = np.column_stack(
            [self.dir_meas['Dn4000'],
             self.dir_meas['Hdelta_A'] + self.Hd_em_EWs])
        direct = direct[~self.mask]

        # inferred indices
        infer = np.column_stack(
            [self.ind_meas['Dn4000'], self.ind_meas['Hdelta_A']])
        infer_good = infer[(~self.mask) * (~self.infer_badPDF)]
        infer_bad = infer[(~self.mask) * (self.infer_badPDF)]

        ax.scatter_density(
            direct[:, 0], direct[:, 1], color='k', label='MaNGA: direct')
        ax.scatter_density(
            self.pca.metadata['Dn4000'], self.pca.metadata['Hdelta_A'],
            color='g', label='CSP models')
        ax.scatter_density(
            infer_bad[:, 0], infer_bad[:, 1], color='r', label='MaNGA: bad fits')
        ax.scatter_density(
            infer_good[:, 0], infer_good[:, 1], color='g', label='MaNGA: good fits')

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

        # directly observed indices
        direct = np.column_stack(
            [self.dir_meas['Dn4000'],
             self.dir_meas['Hdelta_A'] + self.Hd_em_EWs])
        direct = direct[~self.mask]

        # inferred indices
        infer = np.column_stack(
            [self.ind_meas['Dn4000'], self.ind_meas['Hdelta_A'],
             self.infer_badPDF])
        infer_good = infer[(~self.mask) * (~self.infer_badPDF)]
        infer_bad = infer[(~self.mask) * (self.infer_badPDF)]

        KDE_models = KDEMultivariate(
            data=[self.pca.metadata['Dn4000'],
                  self.pca.metadata['Hdelta_A']],
            var_type='cc')

        nx, ny = 200, 200

        XX, YY = np.meshgrid(np.linspace(0., 3.5, nx),
                             np.linspace(-8., 12., ny))

        XXYY = np.column_stack((XX.flatten(), YY.flatten()))

        ZZ_models = KDE_models.pdf(XXYY).reshape((nx, ny))

        ax.contour(XX, YY, ZZ_models, colors='b', zorder=1)

        # raw index measurements
        ax.scatter(direct[:, 0], direct[:, 1], facecolor='k',
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

    howmany = 10
    cosmo = WMAP9
    warn_behav = 'ignore'
    dered_method = 'supersample_vec'
    dered_kwargs = {'nper': 10}
    CSPs_dir = '/usr/data/minhas2/zpace/CSPs/CSPs_CKC14_MaNGA_20171025-1/'

    mpl_v = 'MPL-5'

    drpall = m.load_drpall(mpl_v, index='plateifu')
    drpall = drpall[drpall['nsa_z'] != -9999]
    lsf = ut.MaNGA_LSF.from_drpall(drpall=drpall, n=2)
    pca_kwargs = {'lllim': 3700. * u.AA, 'lulim': 8800. * u.AA,
                  'lsf': lsf, 'z0_': .04}

    pca_pkl_fname = os.path.join(CSPs_dir, 'pca.pkl')
    pca, K_obs = setup_pca(
        fname=pca_pkl_fname, base_dir=CSPs_dir, base_fname='CSPs',
        redo=False, pkl=True, q=10, fre_target=.005, nfiles=40,
        pca_kwargs=pca_kwargs)


    gals = ['8566-12705', '8567-12701', '8939-12704', '8083-12704',
            '8134-12702', '8134-9102', '8135-12701', '8137-12703',
            '8140-12703', '8140-12701', '8140-3701', '8243-12704',
            '8244-9101', '8247-9101', '8249-12703', '8249-12704',
            '8252-12705', '8252-6102', '8253-6104', '8254-3704', '8254-9101',
            '8257-12701', '8257-6101', '8257-6103', '8258-12704']

    gals = ['8259-1902']

    p_i = [g.split('-') for g in gals]

    dereds = [MaNGA_deredshift.from_plateifu(
        plate=int(p), ifu=int(i), MPL_v=mpl_v) for (p, i) in p_i]
    rimgs = [d.drp_hdulist['RIMG'].data for d in dereds]
    Hdelt_em_EWs = [d.dap_hdulist['EMLINE_SEW'].data[14, ...] for d in dereds]

    from warnings import warn, filterwarnings, catch_warnings, simplefilter

    with catch_warnings():
        simplefilter(warn_behav)
        MDComp = ModelDataCompare(
            pca=pca, dereds=dereds, rimgs=rimgs,
            Hd_em_EWs=Hdelt_em_EWs, K_obs=K_obs)

    #mdcomp_fig = MDComp.D4000_Hd_scatter_fig()
    #mdcomp_fig.savefig('D4000-HdA_scatter.png', dpi=300)

    mdcont_fig = MDComp.D4000_Hd_fig()
    mdcont_fig.savefig('D4000-HdA_contours.png', dpi=300)

    param_compare_fig = MDComp.param_compare()
    param_compare_fig.savefig('Lick_inds.png', dpi=300)
