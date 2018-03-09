import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

import figures_tools as ftools

from itertools import product as iproduct

from astropy.io import fits

class Diagnostic(object):
    def __init__(self, results, metadata, drpall):
        self.results = results
        self.metadata = metadata
        self.drpall = drpall

    def make_diag_figure(self, xnames, ynames):
        nobj = len(xnames)

        # initialize subplot size
        gs, fig = ftools.gen_gridspec_fig(
            nobj, add_row=False, border=(0.6, 0.6, 0.2, 0.4),
            space=(0.6, 0.35))

        # set up subplot interactions
        gs_geo = gs.get_geometry()

        fgrid_r, fgrid_c = tuple(list(range(n)) for n in gs_geo)
        gs_iter = iproduct(fgrid_r, fgrid_c)

        # set up kwargs for matplotlib errorbar
        # prefer to change color, then marker
        colors_ = ['C{}'.format(cn) for cn in range(10)]
        markers_ = ['o', 'v', 's', 'P', 'X', 'D', 'H']

        # iterate through rows & columns (rows change fastest)
        # which correspond to different quantities
        for (i, (ri, ci)) in enumerate(gs_iter):

            if i >= len(xnames):
                continue

            # choose axis
            ax = fig.add_subplot(gs[ri, ci])
            kwarg_cycler = cycler(marker=markers_) * \
                           cycler(c=colors_)

            xqty = xnames[i]
            yqty = ynames[i]

            # now iterate through results hdulists
            for (j, (result, kwargs)) in enumerate(
                zip(self.results, kwarg_cycler)):

                kwargs['label'] = result[0].header['PLATEIFU']

                ax = self._add_log_offset_plot(
                    j, xqty=xqty, yqty=yqty, ax=ax, **kwargs)

                ax.tick_params(labelsize=5)

            if i == 0:
                handles_, labels_ = ax.get_legend_handles_labels()
                plt.figlegend(
                    handles=handles_, labels=labels_,
                    loc='upper right', prop={'size': 4.})

        fig.suptitle('PCA fitting diagnostics', size=8.)

        return fig

    def get_log_yerrs(self, i, qty):
        '''
        get the log-ratio between the measured & ground-truth
        '''
        # retrieve appropriate entry from results attribute
        P50, l_unc, u_unc = tuple(
            map(np.squeeze, np.split(self.results[i][qty].data,
                                     3, axis=0)))
        logscale = self.results[i][qty].header['LOGSCALE']

        # if the quantity's on a log scale already, return as-is
        if (logscale) or ('log' in qty):
            return P50, l_unc, u_unc

        # otherwise, log everything, subtract, and return
        logP50 = np.log10(P50)
        log_l_unc = logP50 - np.log10(P50 - l_unc)
        log_u_unc = np.log10(P50 + u_unc) - logP50

        return logP50, log_l_unc, log_u_unc

    def get_meas_vs_truth(self, i, qty):
        logP50, log_l_unc, log_u_unc = self.get_log_yerrs(i, qty)
        truth = self.results[i][qty].header['TRUTH']

        # compare measured range to truth
        P50_ratio = np.log10(10.**logP50 / 10.**truth)
        l_unc_ratio = np.log10(10.**(logP50 - log_l_unc) / 10.**truth)
        u_unc_ratio = np.log10(10.**(logP50 + log_u_unc) / 10.**truth)

        return P50_ratio, l_unc_ratio, u_unc_ratio

    def _munge_xqty(self, xqty, i, arr_len):
        # get x qty
        if 'drpall' in xqty:
            # fetch value from drpall
            _, drpall_col = xqty.split('-')
            plateifu = self.results[i][0].header['PLATEIFU']
            x = self.drpall.loc[plateifu][drpall_col] * \
                np.ones(arr_len)
            xlabel = r'${{\rm {}}}$'.format(
                drpall_col.replace('_', '\_'))
        elif 'hdr' in xqty:
            # fetch value from header
            _, hdr_i, hdr_key = xqty.split('-')
            x = self.results[i][int(hdr_i)].header[hdr_key] * \
                np.ones(arr_len)
            xlabel = r'${{\rm {}}}$'.format(
                hdr_key.replace('_', '\_'))
        else:
            x_hdu = self.results[i][xqty]

            if xqty in self.metadata.colnames:
                xstr = self.metadata[xqty].meta.get('TeX', xqty).strip('$')
            else:
                xstr = xqty

            # check if "truth" value is available: if not, use array
            xtruth = x_hdu.header.get('TRUTH', None)
            if xtruth is not None:
                x = np.ones(arr_len) * xtruth
                xlabel = r'${{ \rm {0} }}$'.format(xstr)
            else:
                # if value used is not true value, denote with tilde
                xlabel = r'${{ \rm \tilde{{{0}}} }}$'.format(xstr)
                xdata = x_hdu.data
                if len(xdata.shape) == 3:
                    x = xdata[0, ...]
                else:
                    x = xdata

        return x, xlabel

    def _munge_yqty(self, yqty, i):
        P50_ratio, l_unc_ratio, u_unc_ratio = self.get_meas_vs_truth(
            i, yqty)

        if 'log' in yqty:
            ylabel = r'$\tilde{{{0}}} - {0}$'.format(
                self.metadata[yqty].meta.get('TeX', yqty).strip('$'))
        else:
            ylabel = r'$\log_{{10}}{{\frac{{\tilde{{{0}}}}}{{{0}}}}}$'.format(
                self.metadata[yqty].meta.get('TeX', yqty).strip('$'))

        return P50_ratio, l_unc_ratio, u_unc_ratio, ylabel

    def _add_log_offset_plot(self, i, xqty, yqty, ax, **kwargs):
        '''
        plot ratio of best-fit vs some other qty
        '''
        P50_ratio, l_unc_ratio, u_unc_ratio, ylabel = self._munge_yqty(yqty, i)
        x, xlabel = self._munge_xqty(xqty, i, arr_len=P50_ratio.shape)

        ax.scatter(x, P50_ratio, s=1., edgecolor='None',
                   alpha=0.6, vmin=0., vmax=1., **kwargs)
        ax.axhline(0., c='k', linewidth=0.25)

        ax.set_ylabel(ylabel, size='x-small')

        ax.set_xlabel(xlabel, size=5)
        ax.set_ylim([-1.25, 1.25])

        if xqty == 'SNRMED':
            ax.set_xscale('log')
            ax.set_xlim([.1, 1.1 * ax.get_xlim()[1]])

        return ax

if __name__ == '__main__':
    from find_pcs import *
    from glob import glob

    cosmo = WMAP9
    warn_behav = 'ignore'
    dered_method = 'supersample_vec'
    dered_kwargs = {'nper': 5}
    CSPs_dir = '/usr/data/minhas2/zpace/CSPs/CSPs_CKC14_MaNGA_20180130-1/'

    mpl_v = 'MPL-6'

    drpall = m.load_drpall(mpl_v, index='plateifu')
    drpall = drpall[drpall['nsa_z'] != -9999]

    lsf = ut.MaNGA_LSF.from_drpall(drpall=drpall, n=2)
    pca_kwargs = {'lllim': 3700. * u.AA, 'lulim': 8800. * u.AA,
                  'lsf': lsf, 'z0_': .04}

    pca, K_obs = setup_pca(
        fname='pca.pkl', base_dir=CSPs_dir, base_fname='CSPs',
        redo=False, pkl=True, q=10, fre_target=.005, nfiles=50,
        pca_kwargs=pca_kwargs)

    # find appropriate files
    hdulists = list(map(fits.open, glob('fakedata/results/*/*_res.fits')))

    # make diag plots
    plt.close('all')
    diag = Diagnostic(results=hdulists, metadata=pca.metadata, drpall=drpall)
    diag_fig = diag.make_diag_figure(
        xnames=['SNRMED', 'tau_V', 'tau_V mu', 'MLV'],
        ynames=['MLV', 'MLV', 'MLV', 'MLV'])
    diag_fig.savefig('diagplot.png');

