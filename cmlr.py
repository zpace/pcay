import numpy as np
from scipy.stats import gaussian_kde, multivariate_normal
import scipy.interpolate as interp

import matplotlib.pyplot as plt
from matplotlib import ticker
import mpl_scatter_density

from astropy import table as t

from importer import *
from functools import partial

from sklearn.neighbors import KDTree

def fit_cmlr(log_ml_a, color_a):
    p, cov = np.polyfit(color_a, log_ml_a, deg=1, cov=True)
    return p, cov

def overplot_cmlr(poly, ax, color_range=[0., 1.5], ycorr=0., **kwargs):
    x = np.linspace(*color_range, 20)
    ax.plot(x, np.polyval(poly, x) + ycorr, **kwargs)

def binlims(data, num, limss=None):
    if type(num) is int:
        num = [num for _ in data]
    elif type(num) in (list, tuple, np.ndarray):
        if len(num) != len(data):
            raise ValueError('list/tuple `num` specifies num of bins, must match data length')
    else:
        raise TypeError('`num` must be list or tuple')

    if limss is None:
        limss = [[d.min(), d.max()] for d in data]
    elif len(limss) != len(data):
        raise ValueError('`limss` must have same number of elements as `data`')
    elif not all([type(ls) in (list, tuple, np.ndarray) for ls in limss]):
        raise TypeError('each element of `limss` must be list or tuple')
    elif not all([len(ls) == 2 for ls in limss]):
        raise ValueError('each element of `limss` must have length 2')
    else:
        pass

    binedges = [np.linspace(*l, n + 1) for l, n in zip(limss, num)]
    return binedges

def hist2d(data1, data2, bins=20, limss=None):
    if (bins is None) or (type(bins) is int):
        if bins is None:
            bins = 20
        bins = binlims([data1, data2], num=bins, limss=limss)
    elif type(bins) in (list, tuple, np.ndarray):
        if all(type(be) is int for be in bins):
            bins = binlims([data1, data2], num=bins, limss=limss)
        elif all(type(be) in (list, tuple, np.ndarray) for be in bins):
            pass
        else:
            raise ValueError('`bins` must be list/tuple of two integers or two lists/tuples')

    hist, *_ = np.histogram2d(
        x=data1, y=data2, bins=bins,
        density=False)

    return hist

def find_knn(pts0, eval_pts, k=15):
    '''
    find the points within `pts0` closest to `eval_pts`
    '''
    pts0range = (pts0.max(axis=0) - pts0.min(axis=0))
    neigh = KDTree(pts0 / pts0range)

    nni = neigh.query(eval_pts / pts0range, k=k, return_distance=False)
    return nni

def medabs(a, axis):
    return np.median(np.abs(a), axis=axis) 

def med(a, axis):
    return np.median(a, axis=axis)

def in_bin_format(*bins, naxes):
    '''
    check if contents of `*bins` are in bins format: `naxes` number of them,
    and each element is list/tuple
    '''

    if len(bins) != naxes:
        return False

    if not all([type(b) in (tuple, list, np.ndarray) for b in bins]):
        return False

    return True

class NdKDE(object):
    '''
    n-dimensional KDE with different bandwidths on each dimension
    '''
    def __init__(self, data, bws='auto', numbins=50, datalims=None, bins=None):
        '''
        data: `npts` by `nd` array
        '''
        self.data = data
        self.npts, self.nd = self.data.shape

        if bws == 'auto':
            bws = self.auto_bw()
        self.sigma = np.diag(bws**2.)
        self.detsigma = np.linalg.det(self.sigma)
        self.invsigma = np.linalg.inv(self.sigma)

        if bins is None:
            self.bins = binlims(self.data.T, numbins, limss=datalims)
        else:
            self.bins = bins

        self.coo_eval = np.meshgrid(*self.bins, indexing='ij')

    def auto_bw(self, fracrange=.05):
        '''
        automatically set the bandwidth as some multiple of the range
        '''
        maxval = self.data.max(axis=0)
        minval = self.data.min(axis=0)
        return fracrange * (maxval - minval)

    def eval_on_grid(self, grids='auto'):
        norm = norm = 1. / np.sqrt((2. * np.pi)**self.nd * self.detsigma)

        # if auto, use native grids
        if grids == 'auto':
            grids = self.bins
            coo_eval = np.stack(self.coo_eval, axis=-1)
        else:
            coo_eval = np.stack(np.meshgrid(*grids, indexing='ij'), axis=-1)

        # loop through data points and build up spatial pdf
        agg_pdf = np.zeros_like(coo_eval[..., 0])
        for pt in self.data:
            agg_pdf += self.ndgau(
                coo_eval, mu=pt, invsigma=self.invsigma, detsigma=self.detsigma, k=self.nd,
                norm=norm)

        return agg_pdf

    @staticmethod
    def ndgau(X, mu, invsigma, detsigma, k, dX=None, norm=None):
        '''
        evaluate an n-d gaussian at coordinates X

        X: 
        '''
        if dX is None:
            dX = X - mu
        if norm is None:
            norm = 1. / np.sqrt((2. * np.pi)**k * detsigma)
        return norm * np.exp(-0.5 * np.einsum('...i,ij,...j->...', dX, invsigma, dX))

    def plot_kde_contours(self, ax, quantiles, n=1000, **kwargs):

        agg_pdf = self.eval_on_grid()
        z = agg_pdf / agg_pdf.sum()
        t = np.linspace(0, z.max(), n)
        integral = ((z >= t[:, None, None]) * z).sum(axis=(1,2))
        f = interp.interp1d(integral, t)
        t_contours = f(quantiles)

        kde_contours = ax.contour(
            z.T, t_contours, extent=[self.bins[0].min(), self.bins[0].max(),
                                     self.bins[1].min(), self.bins[1].max()],
            **kwargs)

        return kde_contours


class CMLR_Diag(object):
    '''
    CMLR diagnostic figure creator
    '''
    projection1 = projection2 = None

    def __init__(self, csp_tab, mlb='i', cb1='g', cb2='r'):
        self.csp_tab = csp_tab
        self.mlb, self.cb1, self.cb2 = mlb, cb1, cb2
        self.cmlr, self.cmlr_cov = fit_cmlr(
            color_a=csp_tab['C{}{}'.format(cb1, cb2)],
            log_ml_a=np.log10(csp_tab['ML{}'.format(mlb)]))
        self._makefig_axs()

    def _makefig_axs(self):
        self.fig = plt.figure(figsize =(7, 4), dpi=200)
        self.cmlr_ax = self.fig.add_subplot(1, 2, 1, projection=self.projection1)
        self.paramspace_ax = self.fig.add_subplot(1, 2, 2, projection=self.projection2)

        self.fig.subplots_adjust(
            left=0.075, right=0.95, bottom=0.125, top=0.9, wspace=.275, hspace=0.)

    def csp_cmlr_plot(self, cbar_name, cbar_label):
        self.cmlr_pts = self.cmlr_ax.scatter(
            self.csp_tab['C{}{}'.format(self.cb1, self.cb2)],
            np.log10(self.csp_tab['ML{}'.format(self.mlb)]),
            c=self.csp_tab[cbar_name], label='CSPs', edgecolor='None', s=1.)

        self.cmlr_ax_cb = plt.colorbar(
            self.cmlr_pts, ax=self.cmlr_ax, orientation='vertical', pad=0.)
        self.cmlr_ax_cb.set_label(cbar_label)
        self.cmlr_ax_cb.ax.tick_params(labelsize='x-small')

        self.cmlr_ax.set_xlabel(r'${} - {}$'.format(self.cb1, self.cb2))
        self.cmlr_ax.set_ylabel(r'$\log \Upsilon^*_{}$'.format(self.mlb))

        self.cmlr_ax.set_xlim([-0.15, 1.7])
        self.cmlr_ax.set_ylim([-1.1, 2.1])

    def paramspace_panel(self, p1name, p2name, p1label, p2label,
                         dlogML_fn, fn_label, fn_TeX, bins=15):
        self.p1name, self.p2name = p1name, p2name
        self.fn_label = fn_label

        pred_logML = np.polyval(self.cmlr, self.csp_tab['C{}{}'.format(self.cb1, self.cb2)])

        dlogML = np.log10(np.array(self.csp_tab['ML{}'.format(self.mlb)])) - pred_logML

        # find which models are closest to which nodes in parameter space
        if in_bin_format(*bins, naxes=2):
            param_edgegrid = bins
        else:
            param_edgegrid = binlims([self.csp_tab[p1name], self.csp_tab[p2name]], bins)

        param_ctrgrid = [0.5 * (eg[1:] + eg[:-1]) for eg in param_edgegrid]
        gridshape = tuple(len(cg) for cg in param_ctrgrid)
        param_ctrfullgrid = np.meshgrid(*param_ctrgrid, indexing='ij')
        param_edgefullgrid = np.meshgrid(*param_edgegrid, indexing='ij')
        binctr_coords = np.column_stack([fg.flatten() for fg in param_ctrfullgrid])
        dlogML_neighbors = dlogML[
            find_knn(
                np.column_stack([np.array(self.csp_tab[p1name]), 
                                 np.array(self.csp_tab[p2name])]),
                binctr_coords)]

        # reduce dlogML measurements by applying the passed function `dlogML_fn`
        fn_at_nodes = dlogML_fn(dlogML_neighbors)
        fn_on_grid = fn_at_nodes.reshape(gridshape)
        self.fn_im = self.paramspace_ax.pcolormesh(*param_edgegrid, fn_on_grid.T, shading='flat')
        self.fn_im_cb = plt.colorbar(
            self.fn_im, ax=self.paramspace_ax, orientation='vertical', pad=0.)
        self.fn_im_cb.set_label(fn_TeX)
        self.fn_im_cb.ax.tick_params(labelsize='x-small')

        # overplot histogram of number of models
        self.kde2d = NdKDE(
            data=np.column_stack([np.array(self.csp_tab[p1name]),
                                  np.array(self.csp_tab[p2name])]),
            bins=param_edgegrid)
        quantiles = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        self.kde_contours = self.kde2d.plot_kde_contours(
            self.paramspace_ax, quantiles=quantiles, colors='k', linewidths=0.5)
        self.paramspace_ax.clabel(
            self.kde_contours, fontsize='x-small',
            fmt={l: str(q) for l, q in zip(self.kde_contours.levels, quantiles)})

        self.paramspace_ax.set_xlabel(p1label)
        self.paramspace_ax.set_ylabel(p2label)

    def save(self):
        self.fig.suptitle(r'${} - {}$ vs. $\log \Upsilon^*_{}$'.format(self.cb1, self.cb2, self.mlb))
        self.cmlr_ax.legend(loc='best', prop={'size': 'x-small'})
        self.fig.savefig(
            os.path.join(
                basedir, 'CMLRDiag_C{}{}ML{}_{}-{}-dev{}.png'.format(
                    self.cb1, self.cb2, self.mlb, self.p1name,
                    self.p2name, self.fn_label)).replace(' ', '__'),
            dpi=self.fig.dpi)

class CMLR_Diag_sd(CMLR_Diag):
    '''
    CMLR diagnostic figure creator, but with scatter-density plots
    '''
    projection1 = projection2 = 'scatter_density'

    def csp_cmlr_plot(self, cbar_name, cbar_label):
        self.cmlr_pts = self.cmlr_ax.scatter_density(
            self.csp_tab['C{}{}'.format(self.cb1, self.cb2)],
            np.log10(self.csp_tab['ML{}'.format(self.mlb)]),
            c=self.csp_tab[cbar_name], label='CSPs')

        self.cmlr_ax_cb = plt.colorbar(
            self.cmlr_pts, ax=self.cmlr_ax, orientation='vertical', pad=0.)
        self.cmlr_ax_cb.set_label(cbar_label)
        self.cmlr_ax_cb.ax.tick_params(labelsize='x-small')

        self.cmlr_ax.set_xlabel(r'${} - {}$'.format(self.cb1, self.cb2))
        self.cmlr_ax.set_ylabel(r'$\log \Upsilon^*_{}$'.format(self.mlb))
        self.cmlr_ax.set_xlim([-0.15, 1.7])
        self.cmlr_ax.set_ylim([-1.1, 2.1])

    def paramspace_panel(self, p1name, p2name, p1label, p2label,
                         dlogML_fn, fn_label, fn_TeX, bins=15):
        self.p1name, self.p2name = p1name, p2name
        self.fn_label = fn_label

        pred_logML = np.polyval(self.cmlr, self.csp_tab['C{}{}'.format(self.cb1, self.cb2)])

        dlogML = np.log10(np.array(self.csp_tab['ML{}'.format(self.mlb)])) - pred_logML

        # find which models are closest to which nodes in parameter space
        if in_bin_format(*bins, naxes=2):
            param_edgegrid = bins
        else:
            param_edgegrid = binlims([self.csp_tab[p1name], self.csp_tab[p2name]], bins)

        param_ctrgrid = [0.5 * (eg[1:] + eg[:-1]) for eg in param_edgegrid]
        gridshape = tuple(len(cg) for cg in param_ctrgrid)
        param_ctrfullgrid = np.meshgrid(*param_ctrgrid, indexing='ij')
        param_edgefullgrid = np.meshgrid(*param_edgegrid, indexing='ij')
        
        self.fn_im = self.paramspace_ax.scatter_density(
            self.csp_tab[p1name], self.csp_tab[p2name], c=dlogML_fn(dlogML), dpi=50,
            cmap='Greens')
        self.fn_im_cb = plt.colorbar(
            self.fn_im, ax=self.paramspace_ax, orientation='vertical', pad=0.)
        self.fn_im_cb.set_label(fn_TeX)
        self.fn_im_cb.ax.tick_params(labelsize='x-small')

        # overplot histogram of number of models
        self.kde2d = NdKDE(
            data=np.column_stack([np.array(self.csp_tab[p1name]),
                                  np.array(self.csp_tab[p2name])]),
            bins=param_edgegrid)
        quantiles = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        self.kde_contours = self.kde2d.plot_kde_contours(
            self.paramspace_ax, quantiles=quantiles, colors='gray', linewidths=0.5)
        self.paramspace_ax.clabel(
            self.kde_contours, fontsize='x-small',
            fmt={l: str(q) for l, q in zip(self.kde_contours.levels, quantiles)})

        self.paramspace_ax.set_xlabel(p1label)
        self.paramspace_ax.set_ylabel(p2label)

        self.paramspace_ax.set_xlabel(p1label)
        self.paramspace_ax.set_ylabel(p2label)

if __name__ == '__main__':
    from glob import glob
    csp_tab = t.vstack(
        [t.Table.read(fn) for fn in glob(os.path.join(basedir, r'CSPs_*.fits'))])
    csp_tab['tau_V mu'] = csp_tab['tau_V'] * csp_tab['mu']
    csp_tab['tau_V (1 - mu)'] = csp_tab['tau_V'] * (1. - csp_tab['mu'])

    cmlr_poly_taylor11_MLiCgi = np.array([0.7, -0.68])
    cmlr_poly_bell03_MLiCgr = np.array([0.864, -0.222])
    cmlr_poly_bell03_MLiCgi = np.array([0.518, -0.152])

    cmlr_diag_MLi_Cgr = CMLR_Diag_sd(csp_tab, mlb='i', cb1='g', cb2='r')
    cmlr_diag_MLi_Cgr.csp_cmlr_plot(cbar_name='logzsol', cbar_label=r'${\rm [Z]}$')
    overplot_cmlr(poly=cmlr_diag_MLi_Cgr.cmlr, ax=cmlr_diag_MLi_Cgr.cmlr_ax, ycorr=0.,
                  linewidth=0.5, c='r', label='CSP CMLR')
    overplot_cmlr(poly=cmlr_poly_bell03_MLiCgr, ax=cmlr_diag_MLi_Cgr.cmlr_ax, ycorr=-.15,
                  linewidth=0.5, c='m', label='Bell et al. (2003) CMLR')
    cmlr_diag_MLi_Cgr.paramspace_panel(
        'logzsol', 'tau_V mu', r'${\rm [Z]}$', r'$\tau_V \mu$', bins=[50, 50],
        dlogML_fn=np.abs, fn_label='abs', 
        fn_TeX=r'$|\Delta\log \Upsilon^*_i|$')
    cmlr_diag_MLi_Cgr.paramspace_ax.set_ylim([0., 5.])
    cmlr_diag_MLi_Cgr.fig.subplots_adjust(left=.1)
    cmlr_diag_MLi_Cgr.save()

    #####

    cmlr_diag_MLi_Cgr = CMLR_Diag_sd(csp_tab, 'i', 'g', 'r')
    cmlr_diag_MLi_Cgr.csp_cmlr_plot(cbar_name='logzsol', cbar_label=r'${\rm [Z]}$')
    overplot_cmlr(poly=cmlr_diag_MLi_Cgr.cmlr, ax=cmlr_diag_MLi_Cgr.cmlr_ax, ycorr=0.,
                  linewidth=0.5, c='r', label='CSP CMLR')
    overplot_cmlr(poly=cmlr_poly_bell03_MLiCgr, ax=cmlr_diag_MLi_Cgr.cmlr_ax, ycorr=-.15,
                  linewidth=0.5, c='m', label='Bell et al. (2003) CMLR')
    cmlr_diag_MLi_Cgr.paramspace_panel(
        'logzsol', 'tau_V mu', r'${\rm [Z]}$', r'$\tau_V \mu$', bins=[50, 50],
        dlogML_fn=lambda x: x, fn_label='', 
        fn_TeX=r'$\Delta\log \Upsilon^*_i$')
    cmlr_diag_MLi_Cgr.paramspace_ax.set_ylim([0., 5.])
    cmlr_diag_MLi_Cgr.fig.subplots_adjust(left=.1)
    cmlr_diag_MLi_Cgr.save()

    #####

    cmlr_diag_MLi_Cgr = CMLR_Diag_sd(csp_tab, 'i', 'g', 'r')
    cmlr_diag_MLi_Cgr.csp_cmlr_plot(cbar_name='logzsol', cbar_label=r'${\rm [Z]}$')
    overplot_cmlr(poly=cmlr_diag_MLi_Cgr.cmlr, ax=cmlr_diag_MLi_Cgr.cmlr_ax, ycorr=0.,
                  linewidth=0.5, c='r', label='CSP CMLR')
    overplot_cmlr(poly=cmlr_poly_bell03_MLiCgr, ax=cmlr_diag_MLi_Cgr.cmlr_ax, ycorr=-.15,
                  linewidth=0.5, c='m', label='Bell et al. (2003) CMLR')
    cmlr_diag_MLi_Cgr.paramspace_panel(
        'logzsol', 'tau_V (1 - mu)', r'${\rm [Z]}$', r'$\tau_V (1 - \mu)$', bins=[50, 50],
        dlogML_fn=np.abs, fn_label='abs', 
        fn_TeX=r'$|\Delta\log \Upsilon^*_i|$')
    cmlr_diag_MLi_Cgr.paramspace_ax.set_ylim([0., 5.])
    cmlr_diag_MLi_Cgr.fig.subplots_adjust(left=.1, wspace=.3)
    cmlr_diag_MLi_Cgr.save()

    #####

    cmlr_diag_MLi_Cgr = CMLR_Diag_sd(csp_tab, 'i', 'g', 'r')
    cmlr_diag_MLi_Cgr.csp_cmlr_plot(cbar_name='logzsol', cbar_label=r'${\rm [Z]}$')
    overplot_cmlr(poly=cmlr_diag_MLi_Cgr.cmlr, ax=cmlr_diag_MLi_Cgr.cmlr_ax, ycorr=0.,
                  linewidth=0.5, c='r', label='CSP CMLR')
    overplot_cmlr(poly=cmlr_poly_bell03_MLiCgr, ax=cmlr_diag_MLi_Cgr.cmlr_ax, ycorr=-.15,
                  linewidth=0.5, c='m', label='Bell et al. (2003) CMLR')
    cmlr_diag_MLi_Cgr.paramspace_panel(
        'logzsol', 'tau_V (1 - mu)', r'${\rm [Z]}$', r'$\tau_V (1 - \mu)$', bins=[50, 50],
        dlogML_fn=lambda x: x, fn_label='', 
        fn_TeX=r'$\Delta\log \Upsilon^*_i$')
    cmlr_diag_MLi_Cgr.paramspace_ax.set_ylim([0., 5.])
    cmlr_diag_MLi_Cgr.fig.subplots_adjust(left=.1, wspace=.3)
    cmlr_diag_MLi_Cgr.save()

    #####

    cmlr_diag_MLi_Cgr = CMLR_Diag_sd(csp_tab, 'i', 'g', 'r')
    cmlr_diag_MLi_Cgr.csp_cmlr_plot(cbar_name='sbss', cbar_label='SBSS')
    overplot_cmlr(poly=cmlr_diag_MLi_Cgr.cmlr, ax=cmlr_diag_MLi_Cgr.cmlr_ax, ycorr=0.,
                  linewidth=0.5, c='r', label='CSP CMLR')
    overplot_cmlr(poly=cmlr_poly_bell03_MLiCgr, ax=cmlr_diag_MLi_Cgr.cmlr_ax, ycorr=-.15,
                  linewidth=0.5, c='m', label='Bell et al. (2003) CMLR')
    cmlr_diag_MLi_Cgr.paramspace_panel(
        'sbss', 'fbhb', 'SBSS', 'FBHB', bins=[50, 50],
        dlogML_fn=lambda x: x, fn_label='', 
        fn_TeX=r'$\Delta\log \Upsilon^*_i$')
    cmlr_diag_MLi_Cgr.fig.subplots_adjust(left=.1)
    cmlr_diag_MLi_Cgr.save()

    #####

    cmlr_diag_MLi_Cgi = CMLR_Diag_sd(csp_tab, mlb='i', cb1='g', cb2='i')
    cmlr_diag_MLi_Cgi.csp_cmlr_plot(cbar_name='logzsol', cbar_label=r'${\rm [Z]}$')
    overplot_cmlr(poly=cmlr_diag_MLi_Cgi.cmlr, ax=cmlr_diag_MLi_Cgi.cmlr_ax, ycorr=0.,
                  linewidth=0.5, c='r', label='CSP CMLR')
    overplot_cmlr(poly=cmlr_poly_bell03_MLiCgi, ax=cmlr_diag_MLi_Cgi.cmlr_ax, ycorr=-.15,
                  linewidth=0.5, c='m', label='Bell et al. (2003) CMLR')
    overplot_cmlr(poly=cmlr_poly_taylor11_MLiCgi, ax=cmlr_diag_MLi_Cgi.cmlr_ax, ycorr=.05,
                  linewidth=0.5, c='b', label='Taylor et al. (2011) CMLR')
    cmlr_diag_MLi_Cgi.paramspace_panel(
        'logzsol', 'tau_V mu', r'${\rm [Z]}$', r'$\tau_V \mu$', bins=[50, 50],
        dlogML_fn=lambda x: x, fn_label='',
        fn_TeX=r'$\Delta\log \Upsilon^*_i$')
    cmlr_diag_MLi_Cgi.paramspace_ax.set_ylim([0., 5.])
    cmlr_diag_MLi_Cgi.fig.subplots_adjust(left=.1)
    cmlr_diag_MLi_Cgi.save()

    #####