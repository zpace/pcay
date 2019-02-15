import totalmass
import read_results
import os
from glob import glob
from importer import *
import manga_tools as m
from astropy import table as t
from astropy import units as u
from astropy.cosmology import WMAP9
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from statsmodels.nonparametric.smoothers_lowess import lowess

drpall = m.load_drpall(mpl_v, index='plateifu')

dapall = m.load_dapall(mpl_v)
dapall = dapall[dapall['DAPDONE'] * (dapall['DAPTYPE'] == 'SPX-GAU-MILESHC')]
dapall.add_index('PLATEIFU')

pca_system = read_results.PCASystem.fromfile(os.path.join(basedir, 'pc_vecs.fits'))

jhumpa = t.Table.read('/usr/data/minhas/zpace/stellarmass_pca/jhu_mpa_mpl7.fits')
jhumpa['plateifu'] = [plateifu.strip(' ') for plateifu in jhumpa['PLATEIFU']]
jhumpa = jhumpa['plateifu', 'LOG_MSTAR']

sfrsd_tab = t.Table.read('/usr/data/minhas/zpace/stellarmass_pca/sigma_sfr.fits')
sfrsd_tab['plateifu'] = sfrsd_tab['names']
del sfrsd_tab['names']
sfrsd_tab.add_index('plateifu')

cmlr_kwargs = {
    'cb1': 'g', 'cb2': 'r',
    'cmlr_poly': np.array([ 1.15614812, -0.48479653])}

def update_mass_table(drpall, old_mass_table=None, limit=None):
    new_mass_table = t.Table(
        data=[[0], [0], ['00000-00000'], [0.],
              [0.], [0.], [0.], [0.], [0.],
              [0.], [0.]],
        names=['plate', 'ifu', 'plateifu', 'mass_in_ifu',
               'fluxwt_ml', 'inner_lum', 'outer_ml_ring', 'outer_ml_cmlr', 'outer_lum',
               'mean_atten_mwtd', 'std_atten_mwtd'],
        dtype = [int, int, str, float,
                 float, float, float, float, float,
                 float, float])

    res_fnames = glob(os.path.join(basedir, 'results/*-*/*-*_res.fits'))[:limit]

    if old_mass_table is None:
        already_aggregated = [False for _ in range(len(res_fnames))]
    else:
        already_aggregated = [os.path.split(fn)[1].split('_')[0] in old_mass_table['plateifu']
                              for fn in res_fnames]

    for res_fname, done in zip(res_fnames, already_aggregated):
        if done:
            continue

        res = read_results.PCAOutput.from_fname(res_fname)
        plateifu = res[0].header['PLATEIFU']
        plate, ifu = plateifu.split('-')
        drp = res.get_drp_logcube('MPL-6')
        dap = res.get_dap_maps('MPL-6', 'SPX-GAU-MILESHC')
        totalmass_res = totalmass.estimate_total_stellar_mass(
            res, pca_system, drp, dap,
            drpall.loc[plateifu], dapall.loc[plateifu], WMAP9,
            missing_mass_kwargs=cmlr_kwargs)

        mass_in_ifu, fluxwt_ml, inner_lum, outer_ml_ring, outer_ml_cmlr, outer_lum = totalmass_res
        mstar_map = read_results.bandpass_mass(
            res, pca_system, WMAP9, 'i',
            drpall.loc[plateifu]['nsa_zdist'])
        mean_atten_mwtd = np.average(
            res.param_dist_med('tau_V'),
            weights=(mstar_map * ~res.mask))
        std_atten_mwtd = np.sqrt(np.average(
            (res.param_dist_med('tau_V') - mean_atten_mwtd)**2.,
            weights=(mstar_map * ~res.mask)))
        data = [plate, ifu, plateifu, mass_in_ifu,
                fluxwt_ml, inner_lum, outer_ml_ring, outer_ml_cmlr, outer_lum,
                mean_atten_mwtd, std_atten_mwtd]
        new_mass_table.add_row(data)

        drp.close()
        dap.close()
        res.close()

    new_mass_table = new_mass_table[1:]
    new_mass_table['outer_mass_ring'] = (new_mass_table['outer_ml_ring'] * \
                                         new_mass_table['outer_lum'])
    new_mass_table['outer_mass_cmlr'] = (new_mass_table['outer_ml_cmlr'] * \
                                         new_mass_table['outer_lum'])
    new_mass_table['inner_mass_fluxwt'] = (new_mass_table['fluxwt_ml'] * \
                                           new_mass_table['inner_lum'])

    if old_mass_table is None:
        mass_table = new_mass_table
    else:
        mass_table = t.vstack([old_mass_table, new_mass_table])

    full_table = t.join(mass_table, drpall, 'plateifu')

    return mass_table, full_table

mass_table, full_table = update_mass_table(drpall, old_mass_table=None, limit=None)
mass_deficit_order = np.argsort(
    full_table['mass_in_ifu'] - (full_table['fluxwt_ml'] * full_table['inner_lum']))[::-1]

def compare_outerml_ring_cmlr(full_table):
    primarysample = m.mask_from_maskbits(full_table['mngtarg1'], [10])
    secondarysample = m.mask_from_maskbits(full_table['mngtarg1'], [11])

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)

    ax.scatter(
        x=(full_table['nsa_elpetro_absmag'][:, 3] - \
           full_table['nsa_elpetro_absmag'][:, 4])[primarysample],
        y=np.log10(full_table['outer_ml_cmlr'] / \
                   full_table['outer_ml_ring'])[primarysample],
        c='r', marker='o',
        s=8. * (full_table['outer_lum'] / (full_table['inner_lum'] + \
                full_table['outer_lum']))[primarysample],
        label='Primary')
    ax.scatter(
        x=(full_table['nsa_elpetro_absmag'][:, 3] - \
           full_table['nsa_elpetro_absmag'][:, 4])[secondarysample],
        y=np.log10(full_table['outer_ml_cmlr'] / \
                   full_table['outer_ml_ring'])[secondarysample],
        c='b', marker='D',
        s=8. * (full_table['outer_lum'] / (full_table['inner_lum'] + \
                full_table['outer_lum']))[secondarysample],
        label='Secondary')

    ax.legend(loc='best', prop={'size': 'xx-small'})
    ax.tick_params(labelsize='xx-small')
    ax.set_xlabel(r'$g-r$', size='x-small')
    ax.set_ylabel(r'$\log{\frac{\Upsilon^*_{\rm CMLR}}{\Upsilon^*_{\rm ring}}}$',
                  size='x-small')
    fig.tight_layout()
    fig.subplots_adjust(top=.95, left=.21, right=.98)
    fig.savefig(os.path.join(basedir, 'lib_diags/', 'outer_ml.png'))

def make_missing_mass_fig(full_table, mltype='ring'):
    primarysample = m.mask_from_maskbits(full_table['mngtarg1'], [10])
    secondarysample = m.mask_from_maskbits(full_table['mngtarg1'], [11])

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
    ax.scatter(
        x=(full_table['nsa_elpetro_absmag'][:, 3] - \
           full_table['nsa_elpetro_absmag'][:, 4])[primarysample],
        y=(full_table['outer_mass_{}'.format(mltype)] / (full_table['mass_in_ifu'] + \
           full_table['outer_mass_{}'.format(mltype)]))[primarysample],
        c='r', s=5., marker='o', label='Primary+')
    ax.scatter(
        x=(full_table['nsa_elpetro_absmag'][:, 3] - \
           full_table['nsa_elpetro_absmag'][:, 4])[secondarysample],
        y=(full_table['outer_mass_{}'.format(mltype)] / (full_table['mass_in_ifu'] + \
           full_table['outer_mass_{}'.format(mltype)]))[secondarysample],
        c='b', s=5., marker='D', label='Secondary')

    ax.legend(loc='best', prop={'size': 'xx-small'})
    ax.tick_params(labelsize='xx-small')
    ax.set_xlabel(r'$g-r$', size='x-small')
    ax.set_ylabel('Stellar-mass fraction outside IFU', size='x-small')
    fig.tight_layout()
    fig.subplots_adjust(top=.95, left=.21, right=.98)
    fig.savefig(os.path.join(basedir, 'lib_diags/', 'mass_outside_ifu_{}.png'.format(mltype)))

def make_missing_flux_fig(full_table):
    primarysample = m.mask_from_maskbits(full_table['mngtarg1'], [10])
    secondarysample = m.mask_from_maskbits(full_table['mngtarg1'], [11])

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
    ax.scatter(
        x=(full_table['nsa_elpetro_absmag'][:, 3] - \
           full_table['nsa_elpetro_absmag'][:, 4])[primarysample],
        y=(full_table['outer_lum'] / (full_table['inner_lum'] + \
           full_table['outer_lum']))[primarysample],
        c='r', s=5., marker='o', label='Primary+')
    ax.scatter(
        x=(full_table['nsa_elpetro_absmag'][:, 3] - \
           full_table['nsa_elpetro_absmag'][:, 4])[secondarysample],
        y=(full_table['outer_lum'] / (full_table['inner_lum'] + \
           full_table['outer_lum']))[secondarysample],
        c='b', s=5., marker='D', label='Secondary')

    ax.legend(loc='best', prop={'size': 'xx-small'})
    ax.tick_params(labelsize='xx-small')
    ax.set_xlabel(r'$g-r$', size='x-small')
    ax.set_ylabel('Flux fraction outside IFU', size='x-small')
    fig.tight_layout()
    fig.subplots_adjust(top=.95, left=.21, right=.98)
    fig.savefig(os.path.join(basedir, 'lib_diags/', 'flux_outside_ifu.png'))

def compare_missing_mass(full_table):
    primarysample = m.mask_from_maskbits(full_table['mngtarg1'], [10])
    secondarysample = m.mask_from_maskbits(full_table['mngtarg1'], [11])

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
    ax.scatter(
        x=(full_table['nsa_elpetro_absmag'][:, 3] - \
           full_table['nsa_elpetro_absmag'][:, 4])[primarysample],
        y=np.log10(
              (full_table['mass_in_ifu'] + full_table['outer_mass_cmlr']) / \
              (full_table['mass_in_ifu'] + full_table['outer_mass_ring']))[primarysample],
        c='r', s=5., marker='o', label='Primary+')
    ax.scatter(
        x=(full_table['nsa_elpetro_absmag'][:, 3] - \
           full_table['nsa_elpetro_absmag'][:, 4])[secondarysample],
        y=np.log10(
              (full_table['mass_in_ifu'] + full_table['outer_mass_cmlr']) / \
              (full_table['mass_in_ifu'] + full_table['outer_mass_ring']))[secondarysample],
        c='b', s=5., marker='D', label='Secondary')

    ax.legend(loc='best', prop={'size': 'xx-small'})
    ax.tick_params(labelsize='xx-small')
    ax.set_xlabel(r'$g-r$', size='x-small')
    ax.set_ylabel(r'$M^{\rm tot}_{\rm CMLR} - M^{\rm tot}_{\rm ring} {\rm [dex]}$',
                  size='x-small')
    fig.tight_layout()
    fig.subplots_adjust(top=.95, left=.21, right=.98)
    fig.savefig(os.path.join(basedir, 'lib_diags/', 'mtot_compare_cmlr_ring.png'))

def compare_mtot_pca_nsa(full_table, jhu_mpa, mltype='ring', nsa_phototype='elpetro'):
    tab = full_table['plateifu', 'mass_in_ifu', 'outer_mass_ring', 'outer_mass_cmlr',
                     'inner_lum', 'outer_lum', 'nsa_elpetro_absmag', 'nsa_elpetro_mass',
                     'nsa_sersic_mass', 'nsa_sersic_absmag']
    jointab = t.join(tab, jhu_mpa, 'plateifu')

    Cgr = jointab['nsa_{}_absmag'.format(nsa_phototype)][:, 3] - \
          jointab['nsa_{}_absmag'.format(nsa_phototype)][:, 4]
    mass_pca = jointab['mass_in_ifu'] + jointab['outer_mass_{}'.format(mltype)]

    chabrier_to_kroupa = 10.**.05

    nsa_h = 1.
    masscorr_nsa = (WMAP9.h / nsa_h)**-2.
    mass_nsa = jointab['nsa_{}_mass'.format(nsa_phototype)] * masscorr_nsa * chabrier_to_kroupa

    mpajhu_h = .7
    masscorr_jhumpa = (WMAP9.h / mpajhu_h)**-2.
    mass_jhumpa = (10.**jointab['LOG_MSTAR']) * masscorr_jhumpa * chabrier_to_kroupa

    lowess_pca_nsa = lowess(endog=np.log10(mass_pca / mass_nsa), exog=Cgr,
                            is_sorted=False, delta=.005, it=15, frac=.2,
                            return_sorted=True)
    lowess_pca_jhumpa = lowess(endog=np.log10(mass_pca / mass_jhumpa), exog=Cgr,
                               is_sorted=False, delta=.005, it=15, frac=.2,
                               return_sorted=True)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)

    ax.scatter(Cgr, np.log10(mass_pca / mass_nsa),
               s=3., edgecolor='None', c='C0', label='NSA')
    ax.plot(lowess_pca_nsa[:, 0], lowess_pca_nsa[:, 1], linewidth=0.5, c='C0')

    ax.scatter(Cgr, np.log10(mass_pca / mass_jhumpa),
               s=3., edgecolor='None', c='C1', label='JHU-MPA')
    ax.plot(lowess_pca_jhumpa[:, 0], lowess_pca_jhumpa[:, 1], linewidth=0.5, c='C1')

    ax.set_ylim([-.2, .5]);
    ax.set_xlim([-.1, 1.])

    ax.legend(loc='best', prop={'size': 'xx-small'})
    ax.tick_params(labelsize='xx-small')
    ax.set_xlabel(r'$g-r$', size='x-small')
    ax.set_ylabel(r'$M^*_{\rm PCA} - M^*_{\rm catalog} {\rm [dex]}$',
                  size='x-small')
    fig.tight_layout()
    fig.subplots_adjust(top=.95, left=.21, right=.97)

    fig.savefig(os.path.join(basedir, 'lib_diags/', 'dMasses.png'), dpi=fig.dpi)

def make_stdtauV_vs_dMass_fig(full_table):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)

    logmass_in_ifu = np.log10(full_table['mass_in_ifu'])
    logmass_in_ifu_lw = np.log10(full_table['fluxwt_ml'] * full_table['inner_lum'])
    std_atten_mwtd = full_table['std_atten_mwtd']
    mean_atten_mwtd = full_table['mean_atten_mwtd']

    sc = ax.scatter(
        x=std_atten_mwtd, y=(logmass_in_ifu - logmass_in_ifu_lw), c=mean_atten_mwtd,
        edgecolor='k', linewidths=0.25, s=3., cmap='viridis_r', norm=mcolors.LogNorm())

    cb = fig.colorbar(sc, ax=ax, format='%.0f')
    cb.set_label(r'$\bar{\tau_V}$', size='x-small')
    cb.ax.tick_params(labelsize='xx-small')

    ax.tick_params(which='major', labelsize='xx-small')
    ax.tick_params(which='minor', labelbottom=False, labelleft=False)
    ax.set_xscale('log')

    ax.set_xlabel(r'$\sigma_{\tau_V}$', size='x-small')
    ax.set_ylabel(r'$\log{ \frac{M^*}{M^*_{\rm LW}} ~ {\rm [dex]} }$', size='x-small')

    hist_ax = ax.twiny()
    hist_ax.hist((logmass_in_ifu - logmass_in_ifu_lw), bins='auto', histtype='step',
                 orientation='horizontal', linewidth=.5, density=True, color='k')

    for yloc, lw, ls in zip(
        np.percentile((logmass_in_ifu - logmass_in_ifu_lw), [16., 50., 84.]),
        [.5, 1., .5], ['--', '-', '--']):

        hist_ax.axhline(yloc, linestyle=ls, linewidth=lw, color='k')

    hist_ax.tick_params(axis='both', which='both', length=0.,
                        labelright=False, labeltop=False)
    hist_ax.set_xlim([0., hist_ax.get_xlim()[1] * 5.])

    fig.tight_layout()
    fig.suptitle('Mass excess from luminosity-weighting', size='x-small')
    fig.subplots_adjust(left=.175, bottom=.1, right=.85, top=.925)

    fig.savefig(
        os.path.join(basedir, 'lib_diags/', 'stdtauV_dMglobloc_meantauV.png'),
        dpi=fig.dpi)

def make_stdtauV_vs_dMass_ba_fig(full_table):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)

    logmass_in_ifu = np.log10(full_table['mass_in_ifu'])
    logmass_in_ifu_lw = np.log10(full_table['fluxwt_ml'] * full_table['inner_lum'])
    std_atten_mwtd = full_table['std_atten_mwtd']
    ba = full_table['nsa_elpetro_ba']

    sc = ax.scatter(
        x=std_atten_mwtd, y=(logmass_in_ifu - logmass_in_ifu_lw), c=ba,
        edgecolor='k', linewidths=0.25, s=3., cmap='viridis_r',
        vmin=0.1, vmax=.9)

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r'$(\frac{b}{a})$', size='xx-small')
    cb.ax.tick_params(labelsize='xx-small')

    ax.tick_params(which='major', labelsize='xx-small')
    ax.tick_params(which='minor', labelbottom=False, labelleft=False)
    ax.set_xscale('log')

    ax.set_xlabel(r'$\sigma_{\tau_V}$', size='x-small')
    ax.set_ylabel(r'$\log{ \frac{M^*}{M^*_{\rm LW}} ~ {\rm [dex]} }$', size='x-small')

    hist_ax = ax.twiny()
    hist_ax.hist((logmass_in_ifu - logmass_in_ifu_lw), bins='auto', histtype='step',
                 orientation='horizontal', linewidth=.5, density=True, color='k')

    for yloc, lw, ls in zip(
        np.percentile((logmass_in_ifu - logmass_in_ifu_lw), [16., 50., 84.]),
        [.5, 1., .5], ['--', '-', '--']):

        hist_ax.axhline(yloc, linestyle=ls, linewidth=lw, color='k')

    hist_ax.tick_params(axis='both', which='both', length=0.,
                        labelright=False, labeltop=False)
    hist_ax.set_xlim([0., hist_ax.get_xlim()[1] * 5.])

    fig.tight_layout()
    fig.suptitle('Mass excess from luminosity-weighting', size='x-small')
    fig.subplots_adjust(left=.175, bottom=.1, right=.925, top=.925)

    fig.savefig(
        os.path.join(basedir, 'lib_diags/', 'stdtauV_dMglobloc_ba.png'),
        dpi=fig.dpi)

def make_stdtauV_vs_dMass_ssfrsd_fig(full_table, sfrsd_tab, mltype='ring'):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)

    merge_table = t.join(full_table, sfrsd_tab, 'plateifu')

    std_atten_mwtd = merge_table['std_atten_mwtd']
    logmass_in_ifu = np.log10(merge_table['mass_in_ifu'])
    logmass_in_ifu_lw = np.log10(merge_table['fluxwt_ml'] * merge_table['inner_lum'])
    sfrsd = merge_table['sigma_sfr']

    mass_pca = merge_table['mass_in_ifu'] + merge_table['outer_mass_{}'.format(mltype)]

    sc = ax.scatter(
        x=std_atten_mwtd, y=(logmass_in_ifu - logmass_in_ifu_lw),
        c=np.log10(sfrsd / mass_pca),
        edgecolor='k', linewidths=0.25, s=3., cmap='viridis_r')

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r'$\log \frac{{\Sigma}^{\rm SFR}_{R<R_e}}{M^*_{\rm tot}}$', size='xx-small')
    cb.ax.tick_params(labelsize='xx-small')

    ax.tick_params(which='major', labelsize='xx-small')
    ax.tick_params(which='minor', labelbottom=False, labelleft=False)
    ax.set_xscale('log')

    ax.set_xlabel(r'$\sigma_{\tau_V}$', size='x-small')
    ax.set_ylabel(r'$\log{ \frac{M^*}{M^*_{\rm LW}} ~ {\rm [dex]} }$', size='x-small')

    hist_ax = ax.twiny()
    hist_ax.hist((logmass_in_ifu - logmass_in_ifu_lw), bins='auto', histtype='step',
                 orientation='horizontal', linewidth=.5, density=True, color='k')

    for yloc, lw, ls in zip(
        np.percentile((logmass_in_ifu - logmass_in_ifu_lw), [16., 50., 84.]),
        [.5, 1., .5], ['--', '-', '--']):

        hist_ax.axhline(yloc, linestyle=ls, linewidth=lw, color='k')

    hist_ax.tick_params(axis='both', which='both', length=0.,
                        labelright=False, labeltop=False)
    hist_ax.set_xlim([0., hist_ax.get_xlim()[1] * 5.])

    fig.tight_layout()
    fig.suptitle('Mass excess from luminosity-weighting', size='x-small')
    fig.subplots_adjust(left=.175, bottom=.1, right=.9, top=.925)

    fig.savefig(
        os.path.join(basedir, 'lib_diags/', 'stdtauV_dMglobloc_ssfrsd.png'),
        dpi=fig.dpi)

def make_stdtauV_vs_ssfrsd_dMass_fig(full_table, sfrsd_tab, mltype='ring'):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)

    std_atten_mwtd = full_table['std_atten_mwtd']
    logmass_in_ifu = np.log10(full_table['mass_in_ifu'])
    logmass_in_ifu_lw = np.log10(full_table['fluxwt_ml'] * full_table['inner_lum'])
    sfrsd = np.array([sfrsd_tab.loc[plateifu]['sigma_sfr']
                      for plateifu in full_table['plateifu']])

    mass_pca = full_table['mass_in_ifu'] + full_table['outer_mass_{}'.format(mltype)]

    sc = ax.scatter(
        x=np.log10(std_atten_mwtd), c=(logmass_in_ifu - logmass_in_ifu_lw),
        y=np.log10(sfrsd / mass_pca),
        edgecolor='k', linewidths=0.25, s=3., cmap='viridis_r')

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r'$\log{ \frac{M^*}{M^*_{\rm LW}} ~ {\rm [dex]} }$', size='xx-small')
    cb.ax.tick_params(labelsize='xx-small')

    ax.tick_params(which='major', labelsize='xx-small')
    ax.tick_params(which='minor', labelbottom=False, labelleft=False)

    ax.set_xlabel(r'$\log \sigma_{\tau_V}$', size='x-small')
    ax.set_ylabel(r'$\log \frac{{\Sigma}^{\rm SFR}_{R<R_e}}{M^*_{\rm tot}}$', size='x-small')

    fig.tight_layout()
    fig.suptitle('Mass excess from luminosity-weighting', size='x-small')
    fig.subplots_adjust(left=.2, bottom=.1, right=.9, top=.925)

    fig.savefig(
        os.path.join(basedir, 'lib_diags/', 'stdtauV_ssfrsd_dMglobloc.png'),
        dpi=fig.dpi)

def make_meantauV_vs_ba_fig(full_table):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)

    mean_atten_mwtd = full_table['mean_atten_mwtd']
    ba = full_table['nsa_elpetro_ba']
    mask = np.logical_or(~np.isfinite(full_table['nsa_elpetro_ba']),
                         ~np.isfinite(full_table['mean_atten_mwtd']))

    sc = ax.scatter(
        x=np.log10(ba[~mask]), y=np.log10(mean_atten_mwtd[~mask]),
        edgecolor='k', linewidths=0.25, s=3., label='PCA fits')
    pv = np.polyfit(
        np.log10(ba[~mask]), np.log10(mean_atten_mwtd[~mask]),
        deg=1, cov=False, full=False)
    pv_resid = np.polyval(pv, np.log10(ba[~mask])) - np.log10(mean_atten_mwtd[~mask])
    pv_resid_rms = pv_resid.std()

    xg = np.array([np.log10(ba[~mask]).min(), np.log10(ba[~mask]).max()])
    fitlabel = r'$\log (\bar{{\tau_V}}) = {:.2f} \times \log \frac{{b}}{{a}} + {:.2f}$'.format(
        *pv)
    ax.plot(xg, np.polyval(pv, xg), c='k', linewidth=.75, label=fitlabel)
    ax.fill_between(
        xg, np.polyval(pv, xg) - pv_resid_rms, np.polyval(pv, xg) + pv_resid_rms,
        color='gray', alpha=0.5, label=r'${{\rm RMS}} = {:.2f}$'.format(pv_resid_rms))

    ax.set_ylabel(r'$\log \bar{\tau_V}$', size='x-small')
    ax.set_xlabel(r'$\log \frac{b}{a}$', size='x-small')
    ax.tick_params(axis='both', which='both', labelsize='xx-small')
    ax.legend(loc='best', prop={'size': 'xx-small'})

    fig.suptitle('Effect of axis ratio on inferred dust properties', size='x-small')
    fig.tight_layout()
    fig.subplots_adjust(left=.175, bottom=.125, right=.95, top=.925)

    fig.savefig(
        os.path.join(basedir, 'lib_diags/', 'meantauV_ba.png'),
        dpi=fig.dpi)

