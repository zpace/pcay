import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colors
import mpl_scatter_density
from corner import corner

from glob import glob
import os

from scipy import stats

from importer import *
import manga_tools as m

import read_results

import warnings

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

def make_binlimtext(bin_bds, i):
    if i == 0:
        # left-most bin
        binlimtext = [r'$-\infty$', bin_bds[0]]
    elif i == len(bin_bds):
        # right-most bin
        binlimtext = [bin_bds[-1], r'$\infty$']
    else:
        binlimtext = [bin_bds[i - 1], bin_bds[i]]

    return binlimtext

def sgn_fr_exp10(x):
    exp = int(np.floor(np.log10(abs(x))))
    if np.sign(x) < 0:
        sgn = '-'
    else:
        sgn = ''
    return sgn, x / 10**exp, exp

def latex_sgn_fr_exp10_format(x):
    if not np.isfinite(x):
        tex = r'---'
    elif x == 0:
        tex = r'$0$'
    else:
        tex = r'${}{:.2f} \times 10^{{{:d}}}$'.format(*sgn_fr_exp10(x))
    
    return tex

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
    print(bin1datatype, bin1name)
    print(bin2datatype, bin2name)
    print(histdatatype, histname)
    print('Is Mock?', mock)
    print('|  Bin 1 lims  |  Bin 2 lims  |  P50  |  P50 - P16  |  P84 - P50  |')

    nbin1s = len(bin1bds) + 1
    nbin2s = len(bin2bds) + 1

    fig = plt.figure(figsize=(3, 4), dpi=300)
    gs = gridspec.GridSpec(nrows=nbin1s, ncols=1, hspace=0., top=0.85)
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
            '''
            print('\t', b1label_i, b2label_i)
            _, _, mean, var, skew, kurt = stats.describe(
                histdata[in_2d_bin].compressed(), nan_policy='omit')
            print('\t', 'mean = {:.2e}'.format(mean))
            print('\t', 'var  = {:.2e}'.format(var))
            print('\t', 'skew = {:.2e}'.format(skew))
            print('\t', 'kurt = {:.2e}'.format(kurt))\
            '''
            try:
                pctl_16, pctl_50, pctl_84 = np.percentile(
                    a=histdata[in_2d_bin].compressed(), q=[16., 50., 84.])
            except IndexError:
                pctl_16 = pctl_50 = pctl_84 = np.nan

            print(
                r'[{}, {}] & [{}, {}] & {} & {} & {} \\ \hline'.format(
                    *make_binlimtext(bin1bds, b1_i), *make_binlimtext(bin2bds, b2_i),
                    latex_sgn_fr_exp10_format(pctl_50), 
                    latex_sgn_fr_exp10_format(pctl_50 - pctl_16), 
                    latex_sgn_fr_exp10_format(pctl_84 - pctl_50)))

        b1_ax.set_yticks([])

        if (b1_i % 2 == 1):
            b1_ax.yaxis.set_label_position('right')

        if b1_ax is axs[0]:
            b1_ax.legend(loc='lower left', bbox_to_anchor=(-.05, 1.01),
                         prop={'size': 'xx-small'}, ncol=2, borderaxespad=0,
                         frameon=False)

        if b1_ax is axs[-1]:
            b1_ax.tick_params(labelsize='xx-small')
            b1_ax.set_xlabel(histlabel, size='xx-small')
        else:
            b1_ax.set_xticks(axs[-1].get_xticks())
            b1_ax.set_xlim(axs[-1].get_xlim())
            b1_ax.tick_params(labelbottom=False)

    print('-----')

    if mock:
        titletype = 'Mocks'
    else:
        titletype = 'Obs'

    fig.suptitle('{}: {} fit diagnostics'.format(titletype, histlabel),
                 size='x-small')
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

    colorbins = [0.35, 0.7]
    snrbins = [2., 10., 20.]
    zbins = [-.5, 0.]
    taubins = [1., 2.5]

    # COLOR
    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='dev-wid', histname='MLi', histlims=[-4., 4.],
        histlabel=r'$\frac{\Delta \log \Upsilon^*_i}{\sigma_{\log \Upsilon^*_i}}$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=snrbins, bin1label=r'${\rm SNR}$',
        bin2datatype='color', bin2name='gr', bin2bds=colorbins,
        bin2label=r'$g - r$', bin2colors=['b', 'g', 'r'], mock=True)
    for ax in fig.axes:
        ax_ = ax.twinx()
        xs = np.linspace(-5., 5., 201)
        gau = lambda x: np.exp(-0.5 * x**2.)
        ax_.plot(xs, gau(xs), linestyle='--')
        ax_.set_yticks([])
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_color_hist_devwidMLi.png'))

    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='param-wid', histname='MLi', histlims=[-.02, .6],
        histlabel=r'$\sigma_{\log \Upsilon^*_i}$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=snrbins, bin1label=r'${\rm SNR}$',
        bin2datatype='color', bin2name='gr', bin2bds=colorbins,
        bin2label=r'$g - r$', bin2colors=['b', 'g', 'r'], mock=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_color_hist_widMLi.png'))

    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='dev', histname='MLi', histlims=[-.6, .6],
        histlabel=r'$\Delta \log \Upsilon^*_i$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=snrbins, bin1label=r'${\rm SNR}$',
        bin2datatype='color', bin2name='gr', bin2bds=colorbins,
        bin2label=r'$g - r$', bin2colors=['b', 'g', 'r'], mock=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_color_hist_devMLi.png'))

    plt.close('all')
    fig = make_binned_hist_fig(
        obs_res, pca_system,
        histdatatype='param-wid', histname='MLi', histlims=[-.02, .6],
        histlabel=r'$\sigma_{\log \Upsilon^*_i}$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=snrbins, bin1label=r'${\rm SNR}$',
        bin2datatype='color', bin2name='gr', bin2bds=colorbins,
        bin2label=r'$g - r$', bin2colors=['b', 'g', 'r'], mock=False)
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'obs_snr_color_hist_widMLi.png'))

    # METALLICITY
    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='dev-wid', histname='MLi', histlims=[-4., 4.],
        histlabel=r'$\frac{\Delta \log \Upsilon^*_i}{\sigma_{\log \Upsilon^*_i}}$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=snrbins, bin1label=r'${\rm SNR}$',
        bin2datatype='truth', bin2name='logzsol', bin2bds=zbins,
        bin2label=r'${\rm [Z]}_0$', bin2colors=['b', 'g', 'r'], mock=True)
    for ax in fig.axes:
        ax_ = ax.twinx()
        xs = np.linspace(-5., 5., 201)
        gau = lambda x: np.exp(-0.5 * x**2.)
        ax_.plot(xs, gau(xs), linestyle='--')
        ax_.set_yticks([])
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_Z_hist_devwidMLi.png'))

    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='param-wid', histname='MLi', histlims=[-.02, .6],
        histlabel=r'$\sigma_{\log \Upsilon^*_i}$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=snrbins, bin1label=r'${\rm SNR}$',
        bin2datatype='truth', bin2name='logzsol', bin2bds=zbins,
        bin2label=r'${\rm [Z]}_0$', bin2colors=['b', 'g', 'r'], mock=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_Z_hist_widMLi.png'))

    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='dev', histname='MLi', histlims=[-.6, .6],
        histlabel=r'$\Delta \log \Upsilon^*_i$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=snrbins, bin1label=r'${\rm SNR}$',
        bin2datatype='truth', bin2name='logzsol', bin2bds=zbins,
        bin2label=r'${\rm [Z]}_0$', bin2colors=['b', 'g', 'r'], mock=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_Z_hist_devMLi.png'))

    # tau_V
    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='dev-wid', histname='MLi', histlims=[-4., 4.],
        histlabel=r'$\frac{\Delta \log \Upsilon^*_i}{\sigma_{\log \Upsilon^*_i}}$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=snrbins, bin1label=r'${\rm SNR}$',
        bin2datatype='truth', bin2name='tau_V', bin2bds=taubins,
        bin2label=r'$(\tau_V)_0$', bin2colors=['b', 'g', 'r'], mock=True)
    for ax in fig.axes:
        ax_ = ax.twinx()
        xs = np.linspace(-5., 5., 201)
        gau = lambda x: np.exp(-0.5 * x**2.)
        ax_.plot(xs, gau(xs), linestyle='--')
        ax_.set_yticks([])
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_tauV_hist_devwidMLi.png'))

    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='param-wid', histname='MLi', histlims=[-.02, .6],
        histlabel=r'$\sigma_{\log \Upsilon^*_i}$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=snrbins, bin1label=r'${\rm SNR}$',
        bin2datatype='truth', bin2name='tau_V', bin2bds=taubins,
        bin2label=r'$(\tau_V)_0$', bin2colors=['b', 'g', 'r'], mock=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_tauV_hist_widMLi.png'))

    plt.close('all')
    fig = make_binned_hist_fig(
        mocks_res, pca_system,
        histdatatype='dev', histname='MLi', histlims=[-.6, .6],
        histlabel=r'$\Delta \log \Upsilon^*_i$',
        bin1datatype='map', bin1name='SNRMED', bin1bds=snrbins, bin1label=r'${\rm SNR}$',
        bin2datatype='truth', bin2name='tau_V', bin2bds=taubins,
        bin2label=r'$(\tau_V)_0$', bin2colors=['b', 'g', 'r'], mock=True)
    fig.savefig(os.path.join(basedir, lib_diags_subdir,
                'mocks_snr_tauV_hist_devMLi.png'))

if __name__ == '__main__':
    pca_system = read_results.PCASystem.fromfile(os.path.join(basedir, 'pc_vecs.fits'))
    lib_diags_subdir = 'lib_diags/'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        make_paper_libdiag_hists(basedir, pca_system, lib_diags_subdir)