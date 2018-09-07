import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import powerlaw

from astropy.io import fits
from astropy import table as t

import os
import glob
from itertools import product as iterprod

from collections import Counter

import figures_tools
from find_pcs import *

from spectrophot import reddener

class Comparator(object):
    '''
    compares models to data
    '''
    def __init__(self, trn_metadata, test_metadata, workdir,
                 mocks_results_fnames, nsub):
        self.trn_metadata = trn_metadata
        self.workdir = workdir
        self.test_metadata = test_metadata

        self.mocks_results_fnames = mocks_results_fnames
        self.trn_usage_cts = self._compute_trn_usage()

        self.nsub = nsub

    def _compute_trn_usage(self):
        self.n_trn = len(self.trn_metadata)
        ctr = Counter(dict(zip(range(self.n_trn),
                               np.zeros(self.n_trn, dtype=int))))
        for mocks_result_fname in self.mocks_results_fnames:
            res = fits.open(mocks_result_fname)
            mask = res['MASK'].data
            model = res['MODELNUM'].data
            ctr.update(model.flatten()[~mask.astype(bool).flatten()])
            res.close()

        counts = np.array(list(ctr.items()))[:, 1][:self.n_trn]

        return counts

    def _overplot_models(self, ngroups, ax, xdata, ydata):
        nsfhgroups = len(self.trn_metadata) // self.nsub
        start_inds = np.random.choice(list(range(nsfhgroups)), size=ngroups,
                                      replace=False)
        for i0 in start_inds:
            all_inds = i0 + np.array(range(self.nsub))
            x_use, y_use = xdata[all_inds], ydata[all_inds]
            ax.scatter(x_use, y_use, s=2., edgecolor='None')

    def _add_color_arrow(self, c1, c2, ax, l0=np.array([.2, .2])):
        '''
        appx. change in model colors resulting from dust attenuation
        '''
        _, c1b1, c1b2, *_ = c1
        _, c2b1, c2b2, *_ = c2

        bp_fmt = lambda b: '-'.join(('sdss2010', b))

        dc1, dc2 = reddener(c1=tuple(map(bp_fmt, (c1b1, c1b2))),
                            c2=tuple(map(bp_fmt, (c2b1, c2b2))))

        ax.arrow(l0[0], l0[1], dc1[0], dc2[0])

    def make_weightfig(self, xqty, yqty, xbds=[None, None], ybds=[None, None],
                       logx=False, logy=False, xbins=20, ybins=20, overplot_ngroups=0):
        xdata = self.trn_metadata[xqty].data
        ydata = self.trn_metadata[yqty].data

        if logx:
            xdata = np.log10(xdata)
        if logy:
            ydata = np.log10(ydata)

        xbds = figures_tools.decide_lims_pctls(xdata, bds=xbds)
        ybds = figures_tools.decide_lims_pctls(ydata, bds=ybds)
        xbinedges = np.linspace(*xbds, xbins + 1)
        ybinedges = np.linspace(*ybds, ybins + 1)
        xbinctrs = 0.5 * (xbinedges[:-1] + xbinedges[1:])
        ybinctrs = 0.5 * (ybinedges[:-1] + ybinedges[1:])

        rawhist, *_ = np.histogram2d(xdata, ydata, bins=[xbinedges, ybinedges])
        wthist, *_ = np.histogram2d(xdata, ydata, bins=[xbinedges, ybinedges],
                                    weights=self.trn_usage_cts)
        Xgrid, Ygrid = np.meshgrid(xbinctrs, ybinctrs, indexing='ij')

        fig, ax = plt.subplots(1, 1)
        raw_cs = ax.contour(
            Xgrid, Ygrid, rawhist, colors='C0', linewidths=0.25)
        wtd_cs = ax.contour(
            Xgrid, Ygrid, wthist, colors='C1', linewidths=0.25)

        raw_cs.collections[-1].set_label('all')
        wtd_cs.collections[-1].set_label('wtd')

        if overplot_ngroups > 0:
            self._overplot_models(overplot_ngroups, ax, xdata, ydata)

        if (xqty[0] == 'C') and (yqty[0] == 'C'):
            self._add_color_arrow(c1=xqty, c2=yqty, ax=ax)

        ax.legend()

        ax.set_xlabel(xqty.replace('_', '\_'), size='x-small')
        ax.set_ylabel(yqty.replace('_', '\_'), size='x-small')

        fname = '_'.join(('wtfig', xqty, yqty)) + '.png'
        figures_tools.savefig(fig, fname=fname, fdir=self.workdir)

    def make_weighthist(self, **kwargs):
        '''
        histogram of total template usage
        '''
        fig, ax = plt.subplots(1, 1)
        histkws = dict(histtype='step', normed=True)
        histkws.update(**kwargs)
        bins = np.unique(
            np.logspace(0, np.log10(len(self.trn_usage_cts) + 1),
                        21, dtype=int)) - .5
        ax.hist(self.trn_usage_cts + 1, bins=bins, **histkws)

        ax.set_xlabel(r'$N + 1$', size='x-small')
        ax.set_xscale('log')
        ax.set_yscale('log')

        fname = 'wthist.png'
        figures_tools.savefig(fig, fname=fname, fdir=self.workdir)

    def make_wtvsparamval(self, allparams=True, paramlist=None):
        '''
        template usage vs parameter value
        '''
        if allparams:
            paramlist = set(self.test_metadata.colnames) & set(pca.metadata.colnames)
            paramlist = [n for n in paramlist
                         if self.test_metadata[n].shape == (len(self.test_metadata), )]
            paramlist = sorted(paramlist)

        gs, fig = figures_tools.gen_gridspec_fig(
            len(paramlist), border=(1., 1., 0.5, 0.5), space=(0.7, 0.4),
            spsize=(2.5, 1.25))
        subplot_inds = iterprod(range(gs._nrows), range(gs._ncols))

        fig_axes = {n: fig.add_subplot(gs[ii, jj])
                    for (ii, jj), n in zip(subplot_inds, paramlist)}

        for n, ax in fig_axes.items():
            ax.hist(self.trn_metadata[n].data, weights=self.trn_usage_cts,
                       histtype='step', density=True)
            ax.set_yscale('log')
            ax.set_ylabel(r'density', size='x-small')
            ax.tick_params(labelsize='x-small')
            ax.set_xlabel(n.replace('_', '\_'))

        figures_tools.savefig(fig, fname='wtsvsparams.png', fdir=self.workdir)

if __name__ == '__main__':
    CSPs_dir = '/usr/data/minhas2/zpace/CSPs/CSPs_CKC14_MaNGA_20180523-1/'
    all_metadata_fnames = glob.glob(os.path.join(CSPs_dir, 'CSPs_*.fits'))
    trn_metadata_fnames = [f for f in all_metadata_fnames
                           if (('validation' not in f) and ('test' not in f))]
    val_metadata_fnames = [f for f in all_metadata_fnames if 'validation' in f]
    test_metadata_fnames = [f for f in all_metadata_fnames if 'test' in f]

    trn_metadata = t.vstack(list(map(t.Table.read, trn_metadata_fnames)))
    test_metadata = t.vstack(list(map(t.Table.read, test_metadata_fnames)))

    mocks_results_fnames = glob.glob(os.path.join(CSPs_dir, 'results/*/*_res.fits'))
    nsub = fits.getval(trn_metadata_fnames[0], ext=0, keyword='NSUBPER')

    mpl_v = 'MPL-6'
    drpall = m.load_drpall(mpl_v, index='plateifu')
    drpall = drpall[drpall['nsa_z'] != -9999]
    lsf = ut.MaNGA_LSF.from_drpall(drpall=drpall, n=2)
    pca_kwargs = {'lllim': 3700. * u.AA, 'lulim': 8800. * u.AA,
                  'lsf': lsf, 'z0_': .04}

    pca_pkl_fname = os.path.join(CSPs_dir, 'pca.pkl')
    pca, K_obs = setup_pca(
        fname=pca_pkl_fname, base_dir=CSPs_dir, base_fname='CSPs',
        redo=False, pkl=True, q=10, fre_target=.005, nfiles=None,
        pca_kwargs=pca_kwargs)

    comp = Comparator(trn_metadata=pca.metadata, test_metadata=test_metadata,
                      workdir=CSPs_dir, mocks_results_fnames=mocks_results_fnames,
                      nsub=nsub)
    comp.make_weightfig('tau_V', 'mu', xbins=10, ybins=10)
    comp.make_weightfig('Dn4000', 'Hdelta_A', xbins=50, ybins=50)
    comp.make_weightfig('tf', 'd1', xbins=10, ybins=10)
    comp.make_weighthist()
    comp.make_wtvsparamval()
