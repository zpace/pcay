#!/usr/bin/env python3

import os
from glob import glob
from warnings import warn, filterwarnings, catch_warnings, simplefilter
from functools import partial

from importer import *
import manga_tools as m
import totalmass
import read_results

from astropy import table as t
from astropy import units as u
from astropy.cosmology import WMAP9
from astropy.utils.console import ProgressBar

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors as mcolors
from matplotlib import gridspec

from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant as sm_add_constant

drpall = m.load_drpall(mpl_v, index='plateifu')

dapall = m.load_dapall(mpl_v)
dapall = dapall[dapall['DAPDONE'] * (dapall['DAPTYPE'] == daptype)]
dapall.add_index('PLATEIFU')

pca_system = read_results.PCASystem.fromfile(os.path.join(basedir, 'pc_vecs.fits'))

jhumpa = t.Table.read('/usr/data/minhas/zpace/stellarmass_pca/jhu_mpa_{}.fits'.format(
    mpl_v.replace('-', '').lower()))
jhumpa['plateifu'] = [plateifu.strip(' ') for plateifu in jhumpa['PLATEIFU']]
jhumpa = jhumpa['plateifu', 'LOG_MSTAR']
jhumpa = jhumpa[jhumpa['LOG_MSTAR'] > 0.]

sfrsd_tab = t.Table.read('/usr/data/minhas/zpace/stellarmass_pca/sigma_sfr.fits')
sfrsd_tab['plateifu'] = sfrsd_tab['names']
del sfrsd_tab['names']
sfrsd_tab.add_index('plateifu')

def mass_agg_onegal(res_fname, mlband):
    res = read_results.PCAOutput.from_fname(res_fname)
    plateifu = res[0].header['PLATEIFU']
    plate, ifu = plateifu.split('-')
    drp = res.get_drp_logcube(mpl_v)
    dap = res.get_dap_maps(mpl_v, daptype)

    stellarmass = totalmass.StellarMass(
        res, pca_system, drp, dap, drpall.loc[plateifu],
        WMAP9, mlband=mlband)

    with catch_warnings():
        simplefilter('error')
        mass_table_new_entry = stellarmass.to_table()

    mstar_map = stellarmass.mstar[stellarmass.bands_ixs[mlband], ...]

    mean_atten_mwtd = np.average(
        res.param_dist_med('tau_V'),
        weights=(mstar_map * ~res.mask))
    std_atten_mwtd = np.sqrt(np.average(
        (res.param_dist_med('tau_V') - mean_atten_mwtd)**2.,
        weights=(mstar_map * ~res.mask)))

    mass_table_new_entry['mean_atten_mwtd'] = [mean_atten_mwtd]
    mass_table_new_entry['std_atten_mwtd'] = [std_atten_mwtd]

    drp.close()
    dap.close()
    res.close()

    return mass_table_new_entry


def update_mass_table(drpall, mass_table_old=None, limit=None, mlband='i'):
    '''
    '''
    
    # what galaxies are available to aggregate?
    res_fnames = glob(os.path.join(csp_basedir, 'results/*-*/*-*_res.fits'))[:limit]

    # filter out whose that have not been done
    if mass_table_old is None:
        already_aggregated = [False for _ in range(len(res_fnames))]
    else:
        already_aggregated = [os.path.split(fn)[1].split('_')[0] in old_mass_table['plateifu']
                              for fn in res_fnames]
    res_fnames = [fn for done, fn in zip(already_aggregated, res_fnames)]

    # aggregate individual galaxies, and stack them 
    mass_tables_new = list(ProgressBar.map(
        partial(mass_agg_onegal, mlband=mlband), res_fnames, multiprocess=False, step=5))
    mass_table_new = t.vstack(mass_tables_new)

    # if there was an old mass table, stack it with the new one
    if mass_table_old is None:
        mass_table = mass_table_new
    else:
        mass_table = t.vstack([mass_table_old, mass_table_new], join_type='inner')

    cmlr = totalmass.cmlr_kwargs
    missing_flux =  (mass_table['nsa_absmag'].to(m.Mgy) - \
                     mass_table['ifu_absmag'].to(m.Mgy)).clip(
                        a_min=0.*m.Mgy, a_max=np.inf*m.Mgy)
    mag_missing_flux = missing_flux.to(u.ABmag)
    cb1, cb2 = cmlr['cb1'], cmlr['cb2']
    color_missing_flux = mag_missing_flux[:, totalmass.StellarMass.bands_ixs[cb1]] - \
                         mag_missing_flux[:, totalmass.StellarMass.bands_ixs[cb2]]
    color_missing_flux[~np.isfinite(color_missing_flux)] = np.inf
    mass_table['outer_ml_cmlr'] = np.polyval(cmlr['cmlr_poly'], color_missing_flux.value) * \
                                  u.dex(m.m_to_l_unit)
    mass_table['outer_lum'] = mag_missing_flux.to(
        u.dex(m.bandpass_sol_l_unit),
        totalmass.bandpass_flux_to_solarunits(totalmass.StellarMass.absmag_sun))

    mass_table['outer_mass_ring'] = \
        (mass_table['outer_lum'][:, totalmass.StellarMass.bands_ixs['i']] + \
         mass_table['outer_ml_ring']).to(u.Msun)
    mass_table['outer_mass_cmlr'] = \
        (mass_table['outer_lum'][:, totalmass.StellarMass.bands_ixs['i']] + \
         mass_table['outer_ml_cmlr']).to(u.Msun)

    return mass_table

def make_panel_hist(figsize=(3, 3), dpi=300, **kwargs):
    gs_dict = dict(nrows=1, ncols=2, bottom=.125, top=.85, left=.2, right=.95,
        width_ratios=[6, 1], hspace=0., wspace=0.)
    gs_dict.update(**kwargs)
    gs = gridspec.GridSpec(**gs_dict)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    main_ax = fig.add_subplot(gs[0, 0])
    main_ax.tick_params(labelsize='xx-small')

    hist_ax = fig.add_subplot(gs[0, 1], sharey=main_ax)
    hist_ax.tick_params(axis='both', which='both', labelsize='xx-small',
        left=True, labelleft=False, right=True, labelright=False,
        bottom=False, labelbottom=False, top=False, labeltop=False)

    return fig, main_ax, hist_ax

def compare_outerml_ring_cmlr(tab, mlb='i', cb1='g', cb2='r'):
    mlb_ix = totalmass.StellarMass.bands_ixs[mlb]
    cb1_ix = totalmass.StellarMass.bands_ixs[cb1]
    cb2_ix = totalmass.StellarMass.bands_ixs[cb2]
    absmag_sun_mlb = totalmass.StellarMass.absmag_sun[mlb_ix]

    broadband_color = (tab['nsa_absmag'][:, cb1_ix] - tab['nsa_absmag'][:, cb2_ix])
    ml_cmlr_ring = tab['outer_ml_cmlr'] - tab['outer_ml_ring']
    ifu_lum = tab['ifu_absmag'][:, mlb_ix].to(
        m.bandpass_sol_l_unit, totalmass.bandpass_flux_to_solarunits(absmag_sun_mlb))
    outer_lum = tab['outer_lum'][:, mlb_ix].to(m.bandpass_sol_l_unit)
    total_lum = tab['nsa_absmag'][:, mlb_ix].to(
        m.bandpass_sol_l_unit, totalmass.bandpass_flux_to_solarunits(absmag_sun_mlb))
    lum_frac_outer = outer_lum / total_lum

    valid = np.isfinite(tab['outer_ml_cmlr'])

    primarysample = m.mask_from_maskbits(tab['mngtarg1'], [10])
    secondarysample = m.mask_from_maskbits(tab['mngtarg1'], [11])

    fig, main_ax, hist_ax = make_panel_hist(top=0.875)

    for selection, label, marker, color in zip(
        [primarysample, secondarysample], ['Primary+', 'Secondary'],
        ['o', 'D'], ['r', 'b']):
        
        main_ax.scatter(x=broadband_color[selection * valid], y=ml_cmlr_ring[selection * valid],
                       c=color, marker=marker, s=8. * lum_frac_outer[selection * valid],
                       edgecolor='None', label=label)

        hist_ax.hist(ml_cmlr_ring[selection * valid], color=color, density=True, bins='auto',
                     histtype='step', orientation='horizontal', linewidth=0.75)

    main_ax.legend(loc='best', prop={'size': 'xx-small'})
    main_ax.tick_params(labelsize='xx-small')
    main_ax.set_xlabel(r'${}-{}$'.format(cb1, cb2), size='x-small')
    main_ax.set_ylabel(r'$\log{\frac{\Upsilon^*_{\rm CMLR}}{\Upsilon^*_{\rm ring}}}$',
                       size='x-small')
    main_ax.set_ylim(np.percentile(ml_cmlr_ring[valid], [1., 99.]))
    fig.suptitle(r'$\Upsilon^*_{\rm CMLR}$ vs $\Upsilon^*_{\rm ring}$', size='small')
    fig.savefig(os.path.join(basedir, 'lib_diags/', 'outer_ml.png'))

def make_missing_mass_fig(tab, mltype='ring', mlb='i', cb1='g', cb2='r'):
    mlb_ix = totalmass.StellarMass.bands_ixs[mlb]
    cb1_ix = totalmass.StellarMass.bands_ixs[cb1]
    cb2_ix = totalmass.StellarMass.bands_ixs[cb2]
    absmag_sun_mlb = totalmass.StellarMass.absmag_sun[mlb_ix]

    broadband_color = (tab['nsa_absmag'][:, cb1_ix] - tab['nsa_absmag'][:, cb2_ix])
    outermass_frac = tab['outer_mass_{}'.format(mltype)] / \
                     (tab['outer_mass_{}'.format(mltype)] + tab['mass_in_ifu'])

    valid = np.isfinite(tab['outer_ml_{}'.format(mltype)])

    primarysample = m.mask_from_maskbits(tab['mngtarg1'], [10])
    secondarysample = m.mask_from_maskbits(tab['mngtarg1'], [11])

    fig, main_ax, hist_ax = make_panel_hist(top=0.875)

    for selection, label, marker, color in zip(
        [primarysample, secondarysample], ['Primary+', 'Secondary'],
        ['o', 'D'], ['r', 'b']):
    
        main_ax.scatter(
            x=broadband_color[selection * valid], y=outermass_frac[selection * valid],
            c=color, edgecolor='None', s=5., marker=marker, label=label)

        hist_ax.hist(outermass_frac[selection * valid], color=color, density=True, bins='auto',
                     histtype='step', orientation='horizontal', linewidth=0.75)

    main_ax.set_xlim(np.percentile(broadband_color, [1., 99.]))

    main_ax.legend(loc='best', prop={'size': 'xx-small'})
    main_ax.tick_params(labelsize='xx-small')
    main_ax.set_xlabel(r'${}-{}$'.format(cb1, cb2), size='x-small')
    main_ax.set_ylabel('Stellar-mass fraction outside IFU', size='x-small')
    fig.suptitle('Inferred mass fraction outside IFU', size='small')
    fig.savefig(os.path.join(basedir, 'lib_diags/', 'mass_outside_ifu_{}.png'.format(mltype)))

def make_missing_flux_fig(tab, mlb='i', cb1='g', cb2='r'):
    mlb_ix = totalmass.StellarMass.bands_ixs[mlb]
    cb1_ix = totalmass.StellarMass.bands_ixs[cb1]
    cb2_ix = totalmass.StellarMass.bands_ixs[cb2]
    absmag_sun_mlb = totalmass.StellarMass.absmag_sun[mlb_ix]

    broadband_color = (tab['nsa_absmag'][:, cb1_ix] - tab['nsa_absmag'][:, cb2_ix])
    outerlum_frac = tab['outer_lum'][:, mlb_ix].to(m.bandpass_sol_l_unit) / \
                    tab['nsa_absmag'][:, mlb_ix].to(m.bandpass_sol_l_unit, 
                        totalmass.bandpass_flux_to_solarunits(absmag_sun_mlb))

    primarysample = m.mask_from_maskbits(tab['mngtarg1'], [10])
    secondarysample = m.mask_from_maskbits(tab['mngtarg1'], [11])

    fig, main_ax, hist_ax = make_panel_hist(top=0.875)

    for selection, label, marker, color in zip(
        [primarysample, secondarysample], ['Primary+', 'Secondary'],
        ['o', 'D'], ['r', 'b']):
    
        main_ax.scatter(
            x=broadband_color[selection], y=outerlum_frac[selection],
            c=color, edgecolor='None', s=5., marker=marker, label=label)

        hist_ax.hist(outerlum_frac[selection], color=color, density=True, bins='auto',
                     histtype='step', orientation='horizontal', linewidth=0.75)

    main_ax.set_xlim(np.percentile(broadband_color, [1., 99.]))

    main_ax.legend(loc='best', prop={'size': 'xx-small'})
    main_ax.tick_params(labelsize='xx-small')
    main_ax.set_xlabel(r'${}-{}$'.format(cb1, cb2), size='x-small')
    main_ax.set_ylabel('Flux fraction outside IFU', size='x-small')
    fig.suptitle('Flux fraction outside IFU', size='small')
    fig.savefig(os.path.join(basedir, 'lib_diags/', 'flux_outside_ifu.png'))

def compare_missing_mass(tab, mlb='i', cb1='g', cb2='r'):
    mlb_ix = totalmass.StellarMass.bands_ixs[mlb]
    cb1_ix = totalmass.StellarMass.bands_ixs[cb1]
    cb2_ix = totalmass.StellarMass.bands_ixs[cb2]
    absmag_sun_mlb = totalmass.StellarMass.absmag_sun[mlb_ix]

    broadband_color = (tab['nsa_absmag'][:, cb1_ix] - tab['nsa_absmag'][:, cb2_ix])
    outerlum_frac = tab['outer_lum'][:, mlb_ix].to(m.bandpass_sol_l_unit) / \
                    tab['nsa_absmag'][:, mlb_ix].to(m.bandpass_sol_l_unit, 
                        totalmass.bandpass_flux_to_solarunits(absmag_sun_mlb))
    dlogmass_cmlr_ring = ((tab['mass_in_ifu'] + tab['outer_mass_cmlr']).to(u.dex(u.Msun)) - \
                          (tab['mass_in_ifu'] + tab['outer_mass_ring']).to(u.dex(u.Msun)))

    primarysample = m.mask_from_maskbits(tab['mngtarg1'], [10])
    secondarysample = m.mask_from_maskbits(tab['mngtarg1'], [11])

    fig, main_ax, hist_ax = make_panel_hist(top=0.9, left=.225)

    valid = np.isfinite(dlogmass_cmlr_ring)

    for selection, label, marker, color in zip(
        [primarysample, secondarysample], ['Primary', 'Secondary'],
        ['o', 'D'], ['r', 'b']):
    
        main_ax.scatter(
            x=broadband_color[selection * valid], y=dlogmass_cmlr_ring[selection * valid],
            c=color, edgecolor='None', s=5., marker=marker, label=label)

        hist_ax.hist(dlogmass_cmlr_ring[selection * valid], color=color, density=True, bins='auto',
                     histtype='step', orientation='horizontal', linewidth=0.75)

    main_ax.set_xlim(np.percentile(broadband_color[valid], [1., 99.]))
    main_ax.set_ylim(np.percentile(dlogmass_cmlr_ring[valid], [1., 99.]))

    main_ax.legend(loc='best', prop={'size': 'xx-small'})
    main_ax.tick_params(labelsize='xx-small')
    main_ax.set_xlabel(r'${}-{}$'.format(cb1, cb2), size='x-small')
    main_ax.set_ylabel(r'$\log \frac{M^{\rm tot}_{\rm CMLR}}{M^{\rm tot}_{\rm ring}}$',
                  size='x-small')
    fig.suptitle(r'Impact of aperture-correction on $M^{\rm tot}$', size='small')
    fig.savefig(os.path.join(basedir, 'lib_diags/', 'mtot_compare_cmlr_ring.png'))

def smooth(x, y, xgrid, bw):
    '''
    '''
    good = np.isfinite(y)
    x, y = x[good], y[good]

    w = np.exp(-0.5 * (xgrid[None, ...] - x[..., None])**2. / bw**2.)
    w[w == 0] = np.min(w[w > 0])
    y_avg, sum_of_weights = np.average(
        np.tile(y[:, None], [1, len(xgrid)]), weights=w, axis=0, returned=True)

    return y_avg, sum_of_weights


def compare_mtot_pca_nsa(tab, jhu_mpa, mltype='ring', mlb='i', cb1='g', cb2='r'):
    jointab = t.join(tab, jhu_mpa, 'plateifu')

    mlb_ix = totalmass.StellarMass.bands_ixs[mlb]
    cb1_ix = totalmass.StellarMass.bands_ixs[cb1]
    cb2_ix = totalmass.StellarMass.bands_ixs[cb2]
    absmag_sun_mlb = totalmass.StellarMass.absmag_sun[mlb_ix]

    broadband_color = (jointab['nsa_absmag'][:, cb1_ix] - jointab['nsa_absmag'][:, cb2_ix])

    mass_pca = jointab['mass_in_ifu'] + jointab['outer_mass_{}'.format(mltype)]

    nsa_h = 1.
    mass_nsa = (jointab['nsa_elpetro_mass'] * u.Msun * (nsa_h * u.littleh)**-2).to(
        u.Msun, u.with_H0(cosmo.H0))

    jhumpa_h = 1. / .7
    chabrier_to_kroupa_dex = .05
    mass_jhumpa = (10.**(jointab['LOG_MSTAR'] + chabrier_to_kroupa_dex) * \
                   u.Msun * (jhumpa_h * u.littleh)**-2.).to(u.Msun, u.with_H0(cosmo.H0))

    lowess_grid = np.linspace(broadband_color.min(), broadband_color.max(), 100).value
    lowess_pca_nsa, swt_nsa = smooth(
        x=broadband_color.value, y=np.log10(mass_pca / mass_nsa).value,
        xgrid=lowess_grid, bw=.01)
    print(lowess_pca_nsa)
    print(swt_nsa)
    lowess_pca_jhumpa, swt_jhumpa = smooth(
        x=broadband_color.value, y=np.log10(mass_pca / mass_jhumpa).value,
        xgrid=lowess_grid, bw=.01)
    print(lowess_pca_jhumpa)
    print(swt_jhumpa)
    swt_th = .2 * swt_nsa.max()
    good_lowess_nsa = (swt_nsa >= swt_th)
    good_lowess_jhumpa = (swt_jhumpa >= swt_th)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)

    ax.scatter(broadband_color, np.log10(mass_pca / mass_nsa),
               s=2., edgecolor='None', c='C0', label='NSA')
    ax.plot(lowess_grid[good_lowess_nsa], lowess_pca_nsa[good_lowess_nsa], linewidth=0.5, c='k', linestyle='-')

    ax.scatter(broadband_color, np.log10(mass_pca / mass_jhumpa),
               s=2., edgecolor='None', c='C1', label='JHU-MPA')
    ax.plot(lowess_grid[good_lowess_jhumpa], lowess_pca_jhumpa[good_lowess_jhumpa],
            linewidth=0.5, c='k', linestyle='--')

    ax.set_ylim([-.2, .5]);
    ax.set_xlim([-.1, 1.])

    ax.legend(loc='best', prop={'size': 'xx-small'})
    ax.tick_params(labelsize='xx-small')
    ax.set_xlabel(r'${}-{}$'.format(cb1, cb2), size='x-small')
    ax.set_ylabel(r'$\log \frac{M^*_{\rm PCA}}{M^*_{\rm catalog}}$',
                  size='x-small')
    fig.tight_layout()
    fig.subplots_adjust(top=.95, left=.21, right=.97)

    fig.savefig(os.path.join(basedir, 'lib_diags/', 'dMasses.png'), dpi=fig.dpi)

def make_panel_hcb_hist(figsize=(3, 3), dpi=300, **kwargs):
    gs_dict = dict(nrows=2, ncols=2, bottom=.125, top=.85, left=.2, right=.95,
        width_ratios=[6, 1], height_ratios=[1, 12], hspace=0., wspace=0.)
    gs_dict.update(**kwargs)
    gs = gridspec.GridSpec(**gs_dict)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    main_ax = fig.add_subplot(gs[1, 0])
    main_ax.tick_params(labelsize='xx-small')
    main_ax.set_xscale('log')

    hist_ax = fig.add_subplot(gs[1, 1], sharey=main_ax)
    hist_ax.tick_params(axis='both', which='both', labelsize='xx-small',
        left=False, labelleft=False, right=False, labelright=False,
        bottom=False, labelbottom=False, top=False, labeltop=False)
    
    cb_ax = fig.add_subplot(gs[0, :])

    return fig, main_ax, cb_ax, hist_ax

def colorbartop(fig, sc_data, cax):
    cb = fig.colorbar(sc_data, cax=cax, orientation='horizontal', extend='both')
    cb.ax.tick_params(which='both', labelsize='xx-small')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    return cb

def make_stdtauV_vs_dMass_fig(tab, mlb='i'):
    mlb_ix = totalmass.StellarMass.bands_ixs[mlb]
    absmag_sun_mlb = totalmass.StellarMass.absmag_sun[mlb_ix]

    fig, ax, cax, hist_ax = make_panel_hcb_hist(figsize=(3, 3), dpi=300)

    logmass_in_ifu = tab['mass_in_ifu'].to(u.dex(u.Msun))
    logmass_in_ifu_lw = tab['ml_fluxwt'] + tab['ifu_absmag'][:, mlb_ix].to(
        u.dex(m.bandpass_sol_l_unit), totalmass.bandpass_flux_to_solarunits(absmag_sun_mlb))
    std_atten_mwtd = tab['std_atten_mwtd']
    mean_atten_mwtd = tab['mean_atten_mwtd']

    sc = ax.scatter(
        x=std_atten_mwtd, y=(logmass_in_ifu - logmass_in_ifu_lw), c=mean_atten_mwtd,
        edgecolor='k', linewidths=.125, s=2., cmap='viridis_r',
        norm=mcolors.LogNorm(), vmin=.5, vmax=5.)

    cb = colorbartop(fig, sc, cax)
    cb.set_label(r'$\bar{\tau_V}$', size='x-small', labelpad=0)

    ax.set_xlabel(r'$\sigma_{\tau_V}$', size='x-small')
    ax.set_ylabel(r'$\log{ \frac{M^*}{M^*_{\rm LW}} ~ {\rm [dex]} }$', size='x-small')

    hist_ax.hist(np.ma.masked_invalid(logmass_in_ifu - logmass_in_ifu_lw).compressed(),
                 bins='auto', histtype='step', orientation='horizontal', linewidth=.5,
                 density=True, color='k')

    for yloc, lw, ls, c in zip(
        np.percentile(np.ma.masked_invalid(logmass_in_ifu - logmass_in_ifu_lw).compressed(),
                      [16., 50., 84.]),
        [.5, 1., .5], ['--', '-', '--'], ['gray', 'k', 'gray']):

        hist_ax.axhline(yloc, linestyle=ls, linewidth=lw, color=c)

    fig.suptitle('Mass excess from luminosity-weighting', size='x-small')

    fig.savefig(
        os.path.join(basedir, 'lib_diags/', 'stdtauV_dMglobloc_meantauV.png'),
        dpi=fig.dpi)

def make_stdtauV_vs_dMass_ba_fig(tab, mlb='i'):
    mlb_ix = totalmass.StellarMass.bands_ixs[mlb]
    absmag_sun_mlb = totalmass.StellarMass.absmag_sun[mlb_ix]

    fig, ax, cax, hist_ax = make_panel_hcb_hist(figsize=(3, 3), dpi=300)

    logmass_in_ifu = tab['mass_in_ifu'].to(u.dex(u.Msun))
    logmass_in_ifu_lw = tab['ml_fluxwt'] + tab['ifu_absmag'][:, mlb_ix].to(
        u.dex(m.bandpass_sol_l_unit), totalmass.bandpass_flux_to_solarunits(absmag_sun_mlb))
    std_atten_mwtd = tab['std_atten_mwtd']
    mean_atten_mwtd = tab['mean_atten_mwtd']
    ba = tab['nsa_elpetro_ba']

    sc = ax.scatter(
        x=std_atten_mwtd, y=(logmass_in_ifu - logmass_in_ifu_lw), c=ba,
        edgecolor='k', linewidths=.125, s=2., cmap='viridis_r',
        vmin=.15, vmax=.8)

    cb = colorbartop(fig, sc, cax)
    cb.set_label(r'$\frac{b}{a}$', size='x-small', labelpad=0)

    ax.set_xlabel(r'$\sigma_{\tau_V}$', size='x-small')
    ax.set_ylabel(r'$\log{ \frac{M^*}{M^*_{\rm LW}} ~ {\rm [dex]} }$', size='x-small')

    hist_ax.hist(np.ma.masked_invalid(logmass_in_ifu - logmass_in_ifu_lw).compressed(),
                 bins='auto', histtype='step', orientation='horizontal', linewidth=.5,
                 density=True, color='k')

    for yloc, lw, ls, c in zip(
        np.percentile(np.ma.masked_invalid(logmass_in_ifu - logmass_in_ifu_lw).compressed(),
                      [16., 50., 84.]),
        [.5, 1., .5], ['--', '-', '--'], ['gray', 'k', 'gray']):

        hist_ax.axhline(yloc, linestyle=ls, linewidth=lw, color=c)

    fig.suptitle('Mass excess from luminosity-weighting', size='x-small')

    fig.savefig(
        os.path.join(basedir, 'lib_diags/', 'stdtauV_dMglobloc_ba.png'),
        dpi=fig.dpi)

def make_meanstdtauV_vs_dMass_fig(tab, mlb='i'):
    mlb_ix = totalmass.StellarMass.bands_ixs[mlb]
    absmag_sun_mlb = totalmass.StellarMass.absmag_sun[mlb_ix]

    fig, ax, cax, hist_ax = make_panel_hcb_hist(figsize=(3, 3), dpi=300)

    logmass_in_ifu = tab['mass_in_ifu'].to(u.dex(u.Msun))
    logmass_in_ifu_lw = tab['ml_fluxwt'] + tab['ifu_absmag'][:, mlb_ix].to(
        u.dex(m.bandpass_sol_l_unit), totalmass.bandpass_flux_to_solarunits(absmag_sun_mlb))
    std_atten_mwtd = tab['std_atten_mwtd']
    mean_atten_mwtd = tab['mean_atten_mwtd']

    sc = ax.scatter(
        y=std_atten_mwtd, x=mean_atten_mwtd, c=(logmass_in_ifu - logmass_in_ifu_lw),
        edgecolor='k', linewidths=.125, s=2., cmap='viridis_r',
        vmin=.005, vmax=.15)

    cb = colorbartop(fig, sc, cax)
    cb.set_label(r'$\log \frac{M^*}{M_{\rm LW}}$', size='x-small', labelpad=0)

    ax.set_ylabel(r'$\sigma_{\tau_V}$', size='x-small')
    ax.set_xlabel(r'$\bar{\tau_V}$', size='x-small')

    hist_ax.hist(std_atten_mwtd, bins='auto', histtype='step', range=[0.1, 1.5],
                 orientation='horizontal', linewidth=.5, density=True, color='k')

    for yloc, lw, ls, c in zip(
        np.percentile(std_atten_mwtd[np.isfinite(std_atten_mwtd)], [16., 50., 84.]),
        [.5, 1., .5], ['--', '-', '--'], ['gray', 'k', 'gray']):

        hist_ax.axhline(yloc, linestyle=ls, linewidth=lw, color=c)

    fig.suptitle('Mass excess from luminosity-weighting', size='x-small')

    fig.savefig(
        os.path.join(basedir, 'lib_diags/', 'mean+stdtauV_dMglobloc.png'),
        dpi=fig.dpi)

def make_stdtauV_vs_dMass_ssfrsd_fig(tab, sfrsd_tab, mltype='ring', mlb='i'):
    merge_tab = t.join(tab, sfrsd_tab, 'plateifu')

    mlb_ix = totalmass.StellarMass.bands_ixs[mlb]
    absmag_sun_mlb = totalmass.StellarMass.absmag_sun[mlb_ix]

    fig, ax, cax, hist_ax = make_panel_hcb_hist(figsize=(3, 3), dpi=300, top=.8)

    logmass_in_ifu = merge_tab['mass_in_ifu'].to(u.dex(u.Msun))
    logmass_in_ifu_lw = merge_tab['ml_fluxwt'] + merge_tab['ifu_absmag'][:, mlb_ix].to(
        u.dex(m.bandpass_sol_l_unit), totalmass.bandpass_flux_to_solarunits(absmag_sun_mlb))
    std_atten_mwtd = merge_tab['std_atten_mwtd']
    mean_atten_mwtd = merge_tab['mean_atten_mwtd']
    ha_corr = np.exp(merge_tab['mean_atten_mwtd'] * (6563 / 5500)**-1.3)
    sfrsd = merge_tab['sigma_sfr'] * ha_corr * u.Msun / u.yr / u.pc**2
    mass_pca = merge_tab['mass_in_ifu'] + merge_tab['outer_mass_{}'.format(mltype)]
    ssfrsd = sfrsd / mass_pca

    sc = ax.scatter(
        x=std_atten_mwtd, y=(logmass_in_ifu - logmass_in_ifu_lw),
        c=ssfrsd.to(u.dex(ssfrsd.unit)),
        edgecolor='k', linewidths=.125, s=2., cmap='viridis_r',
        vmin=-15., vmax=-10.)

    cb = colorbartop(fig, sc, cax)
    cb.set_label(r'$\log \frac{{\Sigma}^{\rm SFR}_{R<R_e}}{M^*_{\rm tot}}$', size='xx-small')

    ax.tick_params(which='major', labelsize='xx-small')
    ax.tick_params(which='minor', labelbottom=False, labelleft=False)
    ax.set_xscale('log')

    ax.set_xlabel(r'$\sigma_{\tau_V}$', size='x-small')
    ax.set_ylabel(r'$\log{ \frac{M^*}{M^*_{\rm LW}} ~ {\rm [dex]} }$', size='x-small')

    hist_ax.hist(np.ma.masked_invalid(logmass_in_ifu - logmass_in_ifu_lw).compressed(),
                 bins='auto', histtype='step', orientation='horizontal', linewidth=.5,
                 density=True, color='k')

    for yloc, lw, ls, c in zip(
        np.percentile(np.ma.masked_invalid(logmass_in_ifu - logmass_in_ifu_lw).compressed(),
                      [16., 50., 84.]),
        [.5, 1., .5], ['--', '-', '--'], ['gray', 'k', 'gray']):

        hist_ax.axhline(yloc, linestyle=ls, linewidth=lw, color=c)

    fig.suptitle('Mass excess from luminosity-weighting', size='x-small')

    fig.savefig(
        os.path.join(basedir, 'lib_diags/', 'stdtauV_dMglobloc_ssfrsd.png'),
        dpi=fig.dpi)

def make_stdtauV_vs_ssfrsd_dMass_fig(tab, sfrsd_tab, mltype='ring', mlb='i'):
    merge_tab = t.join(tab, sfrsd_tab, 'plateifu')

    mlb_ix = totalmass.StellarMass.bands_ixs[mlb]
    absmag_sun_mlb = totalmass.StellarMass.absmag_sun[mlb_ix]

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)

    logmass_in_ifu = merge_tab['mass_in_ifu'].to(u.dex(u.Msun))
    logmass_in_ifu_lw = merge_tab['ml_fluxwt'] + merge_tab['ifu_absmag'][:, mlb_ix].to(
        u.dex(m.bandpass_sol_l_unit), totalmass.bandpass_flux_to_solarunits(absmag_sun_mlb))
    std_atten_mwtd = merge_tab['std_atten_mwtd']
    mean_atten_mwtd = merge_tab['mean_atten_mwtd']
    ha_corr = np.exp(merge_tab['mean_atten_mwtd'] * (6563 / 5500)**-1.3)
    sfrsd = merge_tab['sigma_sfr'] * ha_corr * u.Msun / u.yr / u.pc**2
    mass_pca = merge_tab['mass_in_ifu'] + merge_tab['outer_mass_{}'.format(mltype)]
    ssfrsd = sfrsd / mass_pca

    sc = ax.scatter(
        x=np.log10(std_atten_mwtd), c=(logmass_in_ifu - logmass_in_ifu_lw),
        y=ssfrsd.to(u.dex(ssfrsd.unit)),
        edgecolor='k', linewidths=.125, s=2., cmap='viridis_r',
        vmin=.01, vmax=.12)

    cb = fig.colorbar(sc, ax=ax, extend='both')
    cb.set_label(r'$\log{ \frac{M^*}{M^*_{\rm LW}} ~ {\rm [dex]} }$', size='xx-small')
    cb.ax.tick_params(labelsize='xx-small')

    ax.tick_params(which='major', labelsize='xx-small')
    ax.tick_params(which='minor', labelbottom=False, labelleft=False)

    ax.set_xlabel(r'$\log \sigma_{\tau_V}$', size='x-small')
    ax.set_ylabel(r'$\log \frac{{\Sigma}^{\rm SFR}_{R<R_e}}{M^*_{\rm tot}}$', size='x-small')

    fig.tight_layout()
    fig.suptitle('Mass excess from luminosity-weighting', size='x-small')
    fig.subplots_adjust(left=.2, bottom=.125, right=.9, top=.925)

    fig.savefig(
        os.path.join(basedir, 'lib_diags/', 'stdtauV_ssfrsd_dMglobloc.png'),
        dpi=fig.dpi)

def make_meantauV_vs_ba_fig(tab):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)

    mean_atten_mwtd = tab['mean_atten_mwtd']
    ba = tab['nsa_elpetro_ba']
    mask = np.logical_or(~np.isfinite(tab['nsa_elpetro_ba']),
                         ~np.isfinite(tab['mean_atten_mwtd']))

    sc = ax.scatter(
        x=np.log10(ba[~mask]), y=np.log10(mean_atten_mwtd[~mask]),
        edgecolor='k', linewidths=.125, s=2., label='PCA fits')
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

def fit_dlogM_mw(tab, sfrsd_tab, mltype='ring', mlb='i'):
    merge_tab = t.join(tab, sfrsd_tab, 'plateifu')
    is_agn = m.mask_from_maskbits(merge_tab['mngtarg3'], [1, 2, 3, 4])

    mlb_ix = totalmass.StellarMass.bands_ixs[mlb]
    absmag_sun_mlb = totalmass.StellarMass.absmag_sun[mlb_ix]

    logmass_in_ifu = merge_tab['mass_in_ifu'].to(u.dex(u.Msun))
    logmass_in_ifu_lw = merge_tab['ml_fluxwt'] + merge_tab['ifu_absmag'][:, mlb_ix].to(
        u.dex(m.bandpass_sol_l_unit), totalmass.bandpass_flux_to_solarunits(absmag_sun_mlb))
    merge_tab['dlogmass_lw'] = logmass_in_ifu - logmass_in_ifu_lw
    ha_corr = np.exp(merge_tab['mean_atten_mwtd'] * (6563 / 5500)**-1.3)
    sfrsd = merge_tab['sigma_sfr'] * ha_corr * u.Msun / u.yr / u.pc**2
    mass_pca = merge_tab['mass_in_ifu'] + merge_tab['outer_mass_{}'.format(mltype)]
    ssfrsd = sfrsd / mass_pca
    merge_tab['log_ssfrsd'] = ssfrsd.to(u.dex(ssfrsd.unit))
    merge_tab['log_ssfrsd'][~np.isfinite(merge_tab['log_ssfrsd'])] = np.nan * merge_tab['log_ssfrsd'].unit

    ols = OLS(
        endog=np.array(merge_tab['dlogmass_lw'][~is_agn]),
        exog=sm_add_constant(
            t.Table(merge_tab['mean_atten_mwtd', 'std_atten_mwtd', 'log_ssfrsd'])[~is_agn].to_pandas(),
            prepend=False),
        hasconst=True, missing='drop')

    olsfit = ols.fit()

    return olsfit


if __name__ == '__main__':
    mlband = 'i'

    mass_table = update_mass_table(drpall, mass_table_old=None, limit=None, mlband=mlband)
    mass_table.write(os.path.join(manga_results_basedir, 'mass_table.fits'), format='fits')
    drpall.keep_columns(['plateifu', 'mangaid', 'objra', 'objdec', 'ebvgal', 
                         'mngtarg1', 'mngtarg2', 'mngtarg3', 'nsa_iauname', 'ifudesignsize',
                         'nsa_z', 'nsa_zdist', 'nsa_nsaid', 'nsa_elpetro_ba', 'nsa_elpetro_mass'])
    full_table = t.join(mass_table, drpall, 'plateifu')
    
    mlband_ix = totalmass.StellarMass.bands_ixs[mlband]
    mlband_absmag_sun = totalmass.StellarMass.absmag_sun[mlband_ix]
    mass_deficit = full_table['mass_in_ifu'].to(u.dex(u.Msun)) - \
        (full_table['ml_fluxwt'] + full_table['ifu_absmag'][:, mlband_ix].to(
             u.dex(m.bandpass_sol_l_unit), totalmass.bandpass_flux_to_solarunits(
                 mlband_absmag_sun)))
    mass_deficit_order = np.argsort(mass_deficit)[::-1]


    compare_outerml_ring_cmlr(full_table)
    compare_missing_mass(full_table)
    make_missing_mass_fig(full_table, mltype='ring')
    make_missing_mass_fig(full_table, mltype='cmlr')
    make_missing_flux_fig(full_table)
    compare_mtot_pca_nsa(full_table, jhumpa, mltype='ring')
    #compare_mtot_pca_nsa(full_table, jhumpa, mltype='cmlr')
    make_meanstdtauV_vs_dMass_fig(full_table)
    make_stdtauV_vs_dMass_ba_fig(full_table)
    make_stdtauV_vs_dMass_fig(full_table)
    
    make_stdtauV_vs_dMass_ssfrsd_fig(full_table, sfrsd_tab, mltype='ring')
    make_stdtauV_vs_dMass_ssfrsd_fig(full_table, sfrsd_tab, mltype='cmlr')
    make_stdtauV_vs_ssfrsd_dMass_fig(full_table, sfrsd_tab, mltype='ring')
    make_stdtauV_vs_ssfrsd_dMass_fig(full_table, sfrsd_tab, mltype='cmlr')
    make_meantauV_vs_ba_fig(full_table)

    lwmass_olsfit = fit_dlogM_mw(full_table, sfrsd_tab)
    lwmass_olsfit.summary()
