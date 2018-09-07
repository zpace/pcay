import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colors
import mpl_scatter_density
from corner import corner

from glob import glob
import os

from importer import *
import manga_tools as m

import read_results

def results_list(basedir):
    return glob(os.path.join(basedir, '*/*_res.fits'))

def concatenate_zipped(listsofarrays):
    return [np.ma.concatenate(onearraytype)
            for onearraytype in zip(*listsofarrays)]

class ResultsHarvester(object):
    def __init__(self, results_fname, pca_system, mock=False):
        if mock:
            cls = read_results.MocksPCAOutput
        else:
            cls = read_results.PCAOutput

        self.mock = mock

        self.results = cls.from_fname(results_fname)
        self.results.setup_photometry(pca_system)

    def datatype_name_parser(self, datatype, name):
        if datatype == 'color':
            fn = self.results.get_color
            args = tuple(name)
        elif datatype == 'map':
            fn = self.results.getdata
            args = (name, )
        elif datatype == 'param-med':
            # median of param PDF
            fn = self.results.param_dist_med
            args = (name, )
        elif datatype == 'param-wid':
            # width of param PDF (16-84 pctl)
            fn = self.results.param_dist_wid
            args = (name, )
        elif self.mock:
            if datatype == 'truth':
                fn = self.results.truth
                args = (name, )
            if datatype == 'dev':
                # deviation from truth
                fn = self.results.dev_from_truth
                args = (name, )
            elif datatype == 'dev-wid':
                # deviation from truth divided by half distribution width
                fn = self.results.dev_from_truth_div_distwid
                args = (name, )
        else:
            raise ValueError('unrecognized datatype/name: {}/{}'.format(datatype, name))

        return fn, args

    def harvest(self, datatype, name):
        fn, args = self.datatype_name_parser(datatype, name)
        return fn(*args)

def harvest_from_result(results_fname, pca_system, types, names, mock=False):
    harvester = ResultsHarvester(results_fname, pca_system, mock=mock)

    try:
        harvested_data = list(map(harvester.harvest, types, names))
        mask = np.logical_or.reduce((
            harvester.results.mask, harvester.results.badPDF(),
            harvester.results.getdata('SNRMED') < .1))
        harvested_data = list(map(lambda a: np.ma.array(a, mask=mask).flatten(), harvested_data))
    finally:
        harvester.results.close()

    return harvested_data

def make_binlabel(bin_bds, i, varname):
    if i == 0:
        # left-most bin
        label = r'{} $ < {}$'.format(varname, bin_bds[0])
    elif i == len(bin_bds):
        # right-most bin
        label = r'${} \le $ {}'.format(bin_bds[-1], varname)
    else:
        label = r'${} \le $ {} $ < {}$'.format(
            bin_bds[i - 1], varname, bin_bds[i])

    return label

def make_binned_hist_fig(results_fnames, pca_system,
                         histdatatype, histname, histlabel, histlims,
                         bin1datatype, bin1name, bin1label, bin1bds,
                         bin2datatype, bin2name, bin2label, bin2bds, bin2colors,
                         mock=False):
    '''
    make multipanel histogram figure where bin1bds defines how points are apportioned into
        subplots, and bin2bds defines how points are apportioned to histograms within
        a given subplot
    '''
    nbin1s = len(bin1bds) + 1
    nbin2s = len(bin2bds) + 1

    fig = plt.figure(figsize=(3, 4))
    gs = gridspec.GridSpec(nrows=nbin1s, ncols=1, hspace=0., top=0.92)
    axs = [None, ] * (len(bin1bds) + 1)
    for i in reversed(range(nbin1s)):
        axs[i] = plt.subplot(gs[i, 0], sharex=axs[-1])

    # retrieve color, SNR, and parameter of interest
    types = [bin1datatype, bin2datatype, histdatatype]
    names = [bin1name, bin2name, histname]
    bin1data, bin2data, histdata = concatenate_zipped(
        [harvest_from_result(fn, pca_system, types, names, mock)
         for fn in results_fnames])

    # assign each sample to a pair of bins
    bin_assignment_1 = np.digitize(bin1data, bin1bds)
    bin1_ax = dict(zip(range(nbin1s), reversed(axs)))

    bin_assignment_2 = np.digitize(bin2data, bin2bds)
    bin2_color = dict(zip(range(nbin2s), bin2colors))

    for b1_i, b1_ax in bin1_ax.items():

        # compose SNR label for axes (1 of m)
        b1label_i = make_binlabel(bin1bds, b1_i, bin1label)
        b1_ax.set_ylabel(b1label_i, size='x-small')

        for b2_i, b2_color in bin2_color.items():
            # compose color label for individual histogram (1 of n on axes)
            b2label_i = make_binlabel(bin2bds, b2_i, bin2label)

            in_2d_bin = np.logical_and(
                (bin_assignment_1 == b1_i), (bin_assignment_2 == b2_i))
            b1_ax.hist(histdata[in_2d_bin].compressed(), range=histlims, bins=40,
                       label=b2label_i, histtype='step', linewidth=.5, density=True,
                       color=bin2colors[b2_i])

            b1_ax.text(x=.05, y=np.linspace(.5, .9, nbin2s)[b2_i],
                       s=str(len(histdata[in_2d_bin].compressed())),
                       fontsize='xx-small', color=b2_color,
                       transform=b1_ax.transAxes)

        b1_ax.set_yticks([])

        if (b1_i % 2 == 1):
            b1_ax.yaxis.set_label_position('right')

        if b1_ax is axs[0]:
            b1_ax.legend(loc='best', prop={'size': 'xx-small'})

        if b1_ax is axs[-1]:
            b1_ax.tick_params(labelsize='xx-small')
            b1_ax.set_xlabel(histlabel, size='xx-small')
        else:
            b1_ax.set_xticks(axs[-1].get_xticks())
            b1_ax.set_xlim(axs[-1].get_xlim())
            b1_ax.tick_params(labelbottom=False)

    if mock:
        titletype = 'Mocks'
    else:
        titletype = 'Obs'

    fig.suptitle('{}: {} fit diagnostics'.format(titletype, histlabel),
                 size='x-small')
    fig.tight_layout()

    return fig

def make_fitdiag_fig(results_fnames, pca_system, xdatatype, xname, xlabel,
                     ydatatype, yname, ylabel, ptcolordatatype, ptcolorname, colorbarlabel,
                     xlims, ylims, clims, colorbar_log=False, mock=False):
    '''
    make a fit-diagnostic figure
    '''

    # munge data into correct form
    if (ptcolordatatype == None) or (ptcolorname == None):
        # no color
        colorbar = False
        types = [xdatatype, ydatatype]
        names = [xname, yname]

        xdata, ydata = concatenate_zipped(
            [harvest_from_result(fn, pca_system, types, names, mock) for fn in results_fnames])
        norm = None
        nfigcols = 2

    else:
        # color
        colorbar = True
        types = [xdatatype, ydatatype, ptcolordatatype]
        names = [xname, yname, ptcolorname]

        xdata, ydata, colordata = concatenate_zipped(
            [harvest_from_result(fn, pca_system, types, names, mock) for fn in results_fnames])

        norm = colors.LogNorm() if colorbar_log else colors.Normalize()
        nfigcols = 3

    # init figure
    fig = plt.figure(figsize=(3, 4))
    gs = gridspec.GridSpec(
        nrows=2, ncols=nfigcols, width_ratios=[1, 4, .5][:nfigcols], height_ratios=[4, 1],
        wspace=.05, hspace=.05, left=.175, right=.825, top=.95)

    if colorbar:
        # color ==> scatterplot + colorbar
        ax = plt.subplot(gs[0, 1])
        cax = plt.subplot(gs[:, 2])

        sc = ax.scatter(xdata, ydata, c=colordata, vmin=clims[0], vmax=clims[-1],
                        s=.25, edgecolor='None', norm=norm)
        cb = fig.colorbar(sc, cax=cax)

        cb.set_label(colorbarlabel, size='x-small')
        cb.ax.tick_params(labelsize='xx-small')

    else:
        # no color ==> scatter-density
        ax = fig.add_subplot(gs[0, 1], projection='scatter_density')
        sc = ax.scatter_density(xdata.compressed(), ydata.compressed(), color='k')

    xhist_ax = plt.subplot(gs[1, 1], sharex=ax)
    yhist_ax = plt.subplot(gs[0, 0], sharey=ax)

    ax.tick_params(labelbottom=False, labelleft=False)

    # place histograms on x and y axis
    xhist = xhist_ax.hist(
        xdata.compressed(), bins=40, range=xlims, density=True, histtype='step',
        linewidth=.5)

    yhist = yhist_ax.hist(
        ydata.compressed(), bins=40, range=ylims, density=True, histtype='step',
        orientation='horizontal', linewidth=.5)

    # place lines

    for v, ls in zip(np.percentile(xdata.compressed(), [16., 50, 84.]), ['--', '-', '--']):
        xhist_ax.axvline(v, linestyle=ls, c='k')

    for v, ls in zip(np.percentile(ydata.compressed(), [16., 50, 84.]), ['--', '-', '--']):
        yhist_ax.axhline(v, linestyle=ls, c='k')

    # suppress tick labels on primary axes

    xhist_ax.set_yticks([])
    xhist_ax.tick_params(labelsize='xx-small')

    yhist_ax.set_xticks([])
    yhist_ax.tick_params(labelsize='xx-small')

    # fix up the figure
    xhist_ax.set_xlabel(xlabel, size='x-small')
    yhist_ax.set_ylabel(ylabel, size='x-small')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    if mock:
        titletype = 'Mocks'
    else:
        titletype = 'Obs'

    fig.suptitle('{}: Fit diagnostics ({} galaxies, {} spaxels)'.format(
                     titletype, len(results_fnames), xdata.count()),
                 size='x-small')
    fig.tight_layout()

    return fig

def get_res_PC_rep(results_fname, mock=False):
    '''
    get PC representation of spectrum
    '''

    if mock:
        results = read_results.MocksPCAOutput.from_fname(results_fname)
    else:
        results = read_results.PCAOutput.from_fname(results_fname)

    calpha = np.column_stack(
        [results.flattenedcubechannel('CALPHA', i)
         for i in range(results['CALPHA'].data.shape[0])])
    mask = np.logical_or.reduce((
        results.mask.flatten(), results.badPDF().flatten(),
        results.getdata('SNRMED').flatten() < .1))

    broadcast_mask = np.tile(mask[None, :], [calpha.shape[1], 1])

    results.close()

    return np.ma.array(calpha, mask=broadcast_mask)

def make_PC_rep_fig(basedir):
    mocks_res_fnames = glob(os.path.join(basedir, 'fakedata/results/*-*/*-*_res.fits'))
    obs_res_fnames = glob(os.path.join(basedir, 'results/*-*/*-*_res.fits'))

    mocks_pc_reps = np.ma.concatenate(
        [get_res_PC_rep(fn, mock=True) for fn in mocks_res_fnames], axis=0)
    obs_pc_reps = np.ma.concatenate(
        [get_res_PC_rep(fn, mock=False) for fn in obs_res_fnames], axis=0)

    pclabels = ['PC{}'.format(i + 1) for i in range(mocks_pc_reps.shape[1])]
    pcranges = .995 * np.ones(len(pclabels))

    filtered_obs = obs_pc_reps.data[~obs_pc_reps.mask[:, 0]]
    filtered_mocks = mocks_pc_reps.data[~mocks_pc_reps.mask[:, 0]]

    fig = corner(
        filtered_mocks, color='r', labels=pclabels,
        range=pcranges, levels=[.68, .95],
        plot_datapoints=False, fill_contours=True, bins=50)
    corner(
        filtered_obs, color='k', labels=pclabels, fig=fig,
        range=pcranges, levels=[.68, .95],
        plot_datapoints=False, fill_contours=True, bins=50)

    return fig

def make_paper_libdiag_hists(basedir, pca_system, lib_diags_subdir):
    # search for all results files
    mocks_res = glob(os.path.join(basedir, 'fakedata/results/*/*_res.fits'))
    obs_res = glob(os.path.join(basedir, 'results/*/*_res.fits'))

    # COLOR
    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='dev-wid', histname='MLi', histlims=[-4., 4.],
        histlabel=r'$\frac{\Delta \log \Upsilon^*_i}{\sigma_{\log \Upsilon^*_i}}$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=[2., 10., 20.], bin1label=r'${\rm SNR}$',
        bin2datatype='color', bin2name='gr', bin2bds=[0.35, 0.7],
        bin2label=r'$g - r$', bin2colors=['b', 'g', 'r'], mock=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_color_hist_devwidMLi.png'))

    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='param-wid', histname='MLi', histlims=[-.02, .6],
        histlabel=r'$\sigma_{\log \Upsilon^*_i}$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=[2., 10., 20.], bin1label=r'${\rm SNR}$',
        bin2datatype='color', bin2name='gr', bin2bds=[0.35, 0.7],
        bin2label=r'$g - r$', bin2colors=['b', 'g', 'r'], mock=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_color_hist_widMLi.png'))

    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='dev', histname='MLi', histlims=[-.6, .6],
        histlabel=r'$\Delta \log \Upsilon^*_i$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=[2., 10., 20.], bin1label=r'${\rm SNR}$',
        bin2datatype='color', bin2name='gr', bin2bds=[0.35, 0.7],
        bin2label=r'$g - r$', bin2colors=['b', 'g', 'r'], mock=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_color_hist_devMLi.png'))

    plt.close('all')
    fig = make_binned_hist_fig(
        obs_res, pca_system,
        histdatatype='param-wid', histname='MLi', histlims=[-.02, .6],
        histlabel=r'$\sigma_{\log \Upsilon^*_i}$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=[2., 10., 20.], bin1label=r'${\rm SNR}$',
        bin2datatype='color', bin2name='gr', bin2bds=[0.35, 0.7],
        bin2label=r'$g - r$', bin2colors=['b', 'g', 'r'], mock=False)
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'obs_snr_color_hist_widMLi.png'))

    # METALLICITY
    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='dev-wid', histname='MLi', histlims=[-4., 4.],
        histlabel=r'$\frac{\Delta \log \Upsilon^*_i}{\sigma_{\log \Upsilon^*_i}}$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=[2., 10., 20.], bin1label=r'${\rm SNR}$',
        bin2datatype='truth', bin2name='logzsol', bin2bds=[-.5, 0.],
        bin2label=r'${\rm [Z]}_0$', bin2colors=['b', 'g', 'r'], mock=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_Z_hist_devwidMLi.png'))

    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='param-wid', histname='MLi', histlims=[-.02, .6],
        histlabel=r'$\sigma_{\log \Upsilon^*_i}$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=[2., 10., 20.], bin1label=r'${\rm SNR}$',
        bin2datatype='truth', bin2name='logzsol', bin2bds=[-.5, 0.],
        bin2label=r'${\rm [Z]}_0$', bin2colors=['b', 'g', 'r'], mock=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_Z_hist_widMLi.png'))

    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='dev', histname='MLi', histlims=[-.6, .6],
        histlabel=r'$\Delta \log \Upsilon^*_i$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=[2., 10., 20.], bin1label=r'${\rm SNR}$',
        bin2datatype='truth', bin2name='logzsol', bin2bds=[-.5, 0.],
        bin2label=r'${\rm [Z]}_0$', bin2colors=['b', 'g', 'r'], mock=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_Z_hist_devMLi.png'))

    # tau_V
    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='dev-wid', histname='MLi', histlims=[-4., 4.],
        histlabel=r'$\frac{\Delta \log \Upsilon^*_i}{\sigma_{\log \Upsilon^*_i}}$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=[2., 10., 20.], bin1label=r'${\rm SNR}$',
        bin2datatype='truth', bin2name='tau_V', bin2bds=[1., 2.5],
        bin2label=r'$(\tau_V)_0$', bin2colors=['b', 'g', 'r'], mock=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_tauV_hist_devwidMLi.png'))

    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='param-wid', histname='MLi', histlims=[-.02, .6],
        histlabel=r'$\sigma_{\log \Upsilon^*_i}$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=[2., 10., 20.], bin1label=r'${\rm SNR}$',
        bin2datatype='truth', bin2name='tau_V', bin2bds=[1., 2.5],
        bin2label=r'$(\tau_V)_0$', bin2colors=['b', 'g', 'r'], mock=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_tauV_hist_widMLi.png'))

    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='dev', histname='MLi', histlims=[-.6, .6],
        histlabel=r'$\Delta \log \Upsilon^*_i$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=[2., 10., 20.], bin1label=r'${\rm SNR}$',
        bin2datatype='truth', bin2name='tau_V', bin2bds=[1., 2.5],
        bin2label=r'$(\tau_V)_0$', bin2colors=['b', 'g', 'r'], mock=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_tauV_hist_devMLi.png'))

def make_paper_lib_diags(basedir, pca_system, lib_diags_subdir):

    # search for all results files
    mocks_res = glob(os.path.join(basedir, 'fakedata/results/*/*_res.fits'))
    obs_res = glob(os.path.join(basedir, 'results/*/*_res.fits'))
    # MOCKS
    ## wid
    plt.close()
    fig = make_fitdiag_fig(
        mocks_res, pca_system,
        ydatatype='param-wid', yname='MLi', ylabel=r'$\sigma_{\log \Upsilon^*_i}$',
        xdatatype='color', xname='gr', xlabel=r'$C_{gr}$',
        ptcolordatatype='map', ptcolorname='SNRMED', colorbarlabel=r'SNR',
        xlims=[-.1, 1.5], ylims=[0., .6], clims=[.1, 30.],
        mock=True, colorbar_log=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir, 'mocks_Cgr_widMLi_SNR.png'))
    ## dev
    ### Cgr-dev MLi-dev SNR
    plt.close()
    fig = make_fitdiag_fig(
        mocks_res, pca_system,
        ydatatype='dev', yname='MLi', ylabel=r'$\Delta \log \Upsilon^*_i$',
        xdatatype='color', xname='gr', xlabel=r'$C_{gr}$',
        ptcolordatatype='map', ptcolorname='SNRMED', colorbarlabel=r'SNR',
        xlims=[-.1, 1.5], ylims=[-.5, .5], clims=[.1, 30.],
        mock=True, colorbar_log=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir, 'mocks_Cgr_devMLi_SNR.png'))
    ### tau_V*mu-dev MLi-dev SNR
    plt.close()
    fig = make_fitdiag_fig(
        mocks_res, pca_system,
        ydatatype='dev', yname='MLi', ylabel=r'$\Delta \log \Upsilon^*_i$',
        xdatatype='dev', xname='tau_V mu', xlabel=r'$\Delta (\tau_V \mu)$',
        ptcolordatatype='map', ptcolorname='SNRMED', colorbarlabel=r'SNR',
        xlims=[-2.5, 2.5], ylims=[-.5, .5], clims=[.1, 30.],
        mock=True, colorbar_log=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir, 'mocks_devtauVmu_devMLi_SNR.png'))
    ### logzsol-dev MLi-dev SNR
    plt.close()
    fig = make_fitdiag_fig(
        mocks_res, pca_system,
        ydatatype='dev', yname='MLi', ylabel=r'$\Delta \log \Upsilon^*_i$',
        xdatatype='dev', xname='logzsol', xlabel=r'$\Delta \rm [Z]$',
        ptcolordatatype='map', ptcolorname='SNRMED', colorbarlabel=r'SNR',
        xlims=[-1.5, 1.5], ylims=[-.5, .5], clims=[.1, 30.],
        mock=True, colorbar_log=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir, 'mocks_devZ_devMLi_SNR.png'))
    ## devwid
    plt.close()
    fig = make_fitdiag_fig(
        mocks_res, pca_system,
        ydatatype='dev-wid', yname='MLi',
        ylabel=r'$\frac{\Delta \log \Upsilon^*_i}{\sigma_{\log \Upsilon^*_i}}$',
        xdatatype='color', xname='gr', xlabel=r'$C_{gr}$',
        ptcolordatatype='map', ptcolorname='SNRMED', colorbarlabel=r'SNR',
        xlims=[-.1, 1.5], ylims=[-4., 4.], clims=[.1, 30.],
        mock=True, colorbar_log=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir, 'mocks_Cgr_devwidMLi_SNR.png'))
    ### tau_V*mu-dev MLi-dev SNR
    plt.close()
    fig = make_fitdiag_fig(
        mocks_res, pca_system,
        ydatatype='dev-wid', yname='MLi',
        ylabel=r'$\frac{\Delta \log \Upsilon^*_i}{\sigma_{\log \Upsilon^*_i}}$',
        xdatatype='dev', xname='tau_V mu',
        xlabel=r'$\frac{\Delta (\tau_V \mu)}{\sigma_{\tau_V \mu}}$',
        ptcolordatatype='map', ptcolorname='SNRMED', colorbarlabel=r'SNR',
        xlims=[-1.75, 1.75], ylims=[-4., 4.], clims=[.1, 30.],
        mock=True, colorbar_log=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir, 'mocks_devwidtauVmu_devwidMLi_SNR.png'))
    ### logzsol-dev MLi-dev SNR
    plt.close()
    fig = make_fitdiag_fig(
        mocks_res, pca_system,
        ydatatype='dev-wid', yname='MLi',
        ylabel=r'$\frac{\Delta \log \Upsilon^*_i}{\sigma_{\log \Upsilon^*_i}}$',
        xdatatype='dev', xname='logzsol',
        xlabel=r'$\frac{\Delta \rm [Z]}{\sigma_{\rm [Z]}}$',
        ptcolordatatype='map', ptcolorname='SNRMED', colorbarlabel=r'SNR',
        xlims=[-1.75, 1.75], ylims=[-4., 4.], clims=[.1, 30.],
        mock=True, colorbar_log=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir, 'mocks_devwidZ_devwidMLi_SNR.png'))

    # OBS
    ## wid
    ### Cgr-wid MLi-wid SNR
    plt.close()
    fig = make_fitdiag_fig(
        obs_res, pca_system,
        ydatatype='param-wid', yname='MLi', ylabel=r'$\sigma_{\log \Upsilon^*_i}$',
        xdatatype='color', xname='gr', xlabel=r'$C_{gr}$',
        ptcolordatatype='map', ptcolorname='SNRMED', colorbarlabel=r'SNR',
        xlims=[-.1, 1.6], ylims=[0., 1.], clims=[.1, 30.],
        mock=False, colorbar_log=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir, 'obs_Cgr_widMLi_SNR.png'))
    ### tau_V*mu-wid MLi-wid SNR
    plt.close()
    fig = make_fitdiag_fig(
        obs_res, pca_system,
        ydatatype='param-wid', yname='MLi', ylabel=r'$\sigma_{\log \Upsilon^*_i}$',
        xdatatype='param-wid', xname='tau_V mu', xlabel=r'$\sigma_{\tau_V \mu}$',
        ptcolordatatype='map', ptcolorname='SNRMED', colorbarlabel=r'SNR',
        xlims=[-.05, 2.5], ylims=[0., 1.], clims=[.1, 30.],
        mock=False, colorbar_log=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir, 'obs_widtauVmu_widMLi_SNR.png'))
    ### logzsol-wid MLi-wid SNR
    plt.close()
    fig = make_fitdiag_fig(
        obs_res, pca_system,
        ydatatype='param-wid', yname='MLi', ylabel=r'$\sigma_{\log \Upsilon^*_i}$',
        xdatatype='param-wid', xname='logzsol', xlabel=r'$\sigma_{\rm [Z]}$',
        ptcolordatatype='map', ptcolorname='SNRMED', colorbarlabel=r'SNR',
        xlims=[-.05, 1.5], ylims=[0., 1.], clims=[.1, 30.],
        mock=False, colorbar_log=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir, 'obs_widlogzsol_widMLi_SNR.png'))

def example():
    fig = make_fitdiag_fig(
        results_files[:150], pca_system, ydatatype='dev', yname='MLi',
        xdatatype='color', xname='gr', ptcolordatatype='color', ptcolorname='ri',
        ylabel=r'$\Delta \log \Upsilon^*_i$', xlabel=r'$C_{gr}$', colorbarlabel=r'$C_{ri}$',
        xlims=[-.1, 2.2], ylims=[-.5, .5], clims=[-.1, 1.5], mock=True)
    return fig
