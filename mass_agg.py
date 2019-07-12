#!/usr/bin/env python3

import os
from glob import glob
from warnings import warn, filterwarnings, catch_warnings, simplefilter
from functools import partial
import dataclasses
import multiprocessing as mpc

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
from matplotlib.legend_handler import HandlerTuple

from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant as sm_add_constant

drpall = m.load_drpall(mpl_v, index='plateifu')

dapall = m.load_dapall(mpl_v)
dapall = dapall[dapall['DAPDONE'] * (dapall['DAPTYPE'] == daptype)]
dapall.add_index('PLATEIFU')

pca_system = read_results.PCASystem.fromfile(os.path.join(csp_basedir, 'pc_vecs.fits'))

jhumpa = t.Table.read('/usr/data/minhas/zpace/stellarmass_pca/jhu_mpa_{}.fits'.format(
    mpl_v.replace('-', '').lower()))
jhumpa['plateifu'] = [plateifu.strip(' ') for plateifu in jhumpa['PLATEIFU']]
jhumpa = jhumpa['plateifu', 'LOG_MSTAR']
jhumpa = jhumpa[jhumpa['LOG_MSTAR'] > 0.]

sfrsd_tab = t.Table.read('/usr/data/minhas/zpace/stellarmass_pca/sigma_sfr.fits')
sfrsd_tab['plateifu'] = sfrsd_tab['names']
del sfrsd_tab['names']
sfrsd_tab.add_index('plateifu')

@dataclasses.dataclass
class MassAggregationManager(object):
    cspbase: str = csp_basedir
    globstring: str = '*/*-*_zpres.fits'
    masstable_fname_base: str = 'masstables/{}.ecsv'
    mlband: str = 'i'

    @staticmethod
    def plateifu_from_fn(fn):
        fn_base = os.path.basename(fn)
        plateifu = fn_base.split('_')[0]
        return plateifu

    @staticmethod
    def find_results(cspbase, res_globstr, masstable_fname_base, redo=False):
        # search in results directory for results files
        results_fnames = glob(os.path.join(cspbase, res_globstr), recursive=True)

        # if redo option false, get only filenames that have no associated mass table
        if redo:
            pass
        else:
            results_fnames = list(filter(
                lambda fn: not os.path.isfile(os.path.join(
                    cspbase,
                    masstable_fname_base.format(
                        MassAggregationManager.plateifu_from_fn(fn)))),
                results_fnames))

        plateifus = list(map(MassAggregationManager.plateifu_from_fn, results_fnames))

        return results_fnames, plateifus

    def find(self, redo=False):
        return self.find_results(
            cspbase=self.cspbase, res_globstr=self.globstring,
            masstable_fname_base=self.masstable_fname_base, redo=redo)

    def start_agg_into_tables(self, redo=False, processes=None, limit=None):
        '''begin the asynchronous aggregation
        '''

        # start pool
        with mpc.Pool(processes=processes) as pool:
            # which file names and plateifus to loop over
            results_fnames, results_plateifus = self.find_results(
                cspbase=self.cspbase, res_globstr=self.globstring,
                masstable_fname_base=self.masstable_fname_base, redo=redo)

            # map table maker over lists of fnames and plateifus
            self.current_async_result = pool.starmap_async(
                aggregate_into_table_file, 
                zip(results_fnames[:limit], results_plateifus[:limit]))

    def agg_done(self):
        '''is aggregation into tables done?
        '''
        if not hasattr(self, 'current_async_result'):
            raise ValueError('no pool associated with this instance!')
        else:
            return self.current_async_result.ready()

    def agg_tasks_remaining(self):
        '''status of mass aggregation
        '''
        if not hasattr(self, 'current_async_result'):
            raise ValueError('no pool associated with this instance!')
        elif self.current_async_result.ready():
            return 0
        else:
            return self.current_async_result._number_left
            

    def table(self):
        if not self.agg_done():
            raise UserWarning('aggregation not complete')
        return t.vstack([t.QTable.read(fn, format='ascii.ecsv') for fn in
                         glob(os.path.join(
                            self.cspbase, self.masstable_fname_base.format('*')))])


def _aggregate_into_table_file(args):
    return aggregate_into_table_file(*args)


def aggregate_into_table_file(res_fname, mlband, cspbase, masstable_fname_base, plateifu):
    try:
        qt = aggregate_one(res_fname, mlband=mlband)
        table_dest = os.path.join(cspbase, masstable_fname_base.format(plateifu))
        qt.write(table_dest, overwrite=True, format='ascii.ecsv')
    except (SystemExit, KeyboardInterrupt) as e:
        raise e
    except Exception as e:
        print(e)
        return False
    else:
        return True

def aggregate_one(res_fname, mlband):
    with read_results.PCAOutput.from_fname(res_fname) as res:
        with res.get_drp_logcube(mpl_v) as drp, res.get_dap_maps(mpl_v, daptype) as dap:
        
            plateifu = res[0].header['PLATEIFU']
            plate, ifu = plateifu.split('-')

            stellarmass = totalmass.StellarMass(
                res, pca_system, drp, dap, drpall.loc[plateifu],
                cosmo, mlband=mlband)

            mstar_map = stellarmass.mstar[stellarmass.bands_ixs[mlband], ...]
            tauVmu_med = res.param_dist_med('tau_V mu')
            tauV1mmu_med = res.param_dist_med('tau_V (1 - mu)')
            tauV_med = tauVmu_med + tauV1mmu_med

            mean_atten_mwtd = np.average(
                tauV_med, weights=(mstar_map * ~res.mask))
            std_atten_mwtd = np.sqrt(np.average(
                (tauV_med - mean_atten_mwtd)**2., weights=(mstar_map * ~res.mask)))

            mass_in_ifu = stellarmass.mstar_in_ifu[stellarmass.bands_ixs[stellarmass.mlband]]
            sollum_in_ifu = stellarmass.sollum_bands.to(m.bandpass_sol_l_unit).sum(axis=(1, 2))
            sollum_nsa = stellarmass.nsa_absmags_cosmocorr.to(
                m.bandpass_sol_l_unit,
                totalmass.bandpass_flux_to_solarunits(stellarmass.absmag_sun))
            ml_fluxwt = stellarmass.logml_fnuwt
            outerml_ring = stellarmass.ml_ring()

            sollum_nsa_names = list(map(
                lambda n: 'sollum_nsa_{}'.format(n),
                stellarmass.bands))
            sollum_in_ifu_names = list(map(
                lambda n: 'sollum_in_ifu_{}'.format(n),
                stellarmass.bands))

    data = [plateifu, mean_atten_mwtd, std_atten_mwtd,
            mass_in_ifu, *sollum_in_ifu, *sollum_nsa,
            ml_fluxwt.to(m.m_to_l_unit), outerml_ring.to(m.m_to_l_unit)]
    names = ['plateifu', 'mean_atten_mwtd' ,'std_atten_mwtd',
             'mass_in_ifu', *sollum_in_ifu_names, *sollum_nsa_names,
             'ml_fluxwt', 'outerml_ring']

    qt = t.QTable()
    for d, n in zip(data, names):
        qt[n] = np.atleast_1d(d)

    return qt


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
        simplefilter('ignore')
        mass_table_new_entry = stellarmass.to_table()

    mstar_map = stellarmass.mstar[stellarmass.bands_ixs[mlband], ...]
    tauVmu_med = res.param_dist_med('tau_V mu')
    tauV1mmu_med = res.param_dist_med('tau_V (1 - mu)')
    tauV_med = tauVmu_med + tauV1mmu_med

    mean_atten_mwtd = np.average(
        tauV_med, weights=(mstar_map * ~res.mask))
    std_atten_mwtd = np.sqrt(np.average(
        (tauV_med - mean_atten_mwtd)**2., weights=(mstar_map * ~res.mask)))

    mass_table_new_entry['mean_atten_mwtd'] = [mean_atten_mwtd]
    mass_table_new_entry['std_atten_mwtd'] = [std_atten_mwtd]

    drp.close()
    dap.close()
    res.close()

    return mass_table_new_entry


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
    '''figure with comparison between aperture correction two M/L
    
    Parameters
    ----------
    tab : astropy.table.Table
        full aggregation results table
    mlb : {str}, optional
        bandpass for mass-to-light ratio
        (the default is 'i', which [default_description])
    cb1 : {str}, optional
        bluer bandpass for color (the default is 'g', SDSS g band)
    cb2 : {str}, optional
        redder bandpass for color (the default is 'r', SDSS r band)
    '''

    cb1_nsa_mag = tab[f'mag_nsa_{cb1}']
    cb2_nsa_mag = tab[f'mag_nsa_{cb2}']

    broadband_color = cb1_nsa_mag - cb2_nsa_mag
    ml_cmlr_ring = tab['outerml_diff']
    lum_frac_outer = tab[f'flux_outer_{mlb}'] / tab[f'flux_nsa_{mlb}']

    valid = np.isfinite(tab['outerml_cmlr'])

    primarysample = m.mask_from_maskbits(tab['mngtarg1'], [10])
    secondarysample = m.mask_from_maskbits(tab['mngtarg1'], [11])

    fig, main_ax, hist_ax = make_panel_hist(top=0.875)

    for selection, label, marker, color in zip(
        [primarysample, secondarysample], ['Primary+', 'Secondary'],
        ['o', 'D'], ['r', 'b']):
        
        main_ax.scatter(x=broadband_color[selection * valid], y=ml_cmlr_ring[selection * valid],
                       c=color, marker=marker, s=10. * lum_frac_outer[selection * valid],
                       edgecolor='None', label=label, alpha=0.5)

        hist_ax.hist(ml_cmlr_ring[selection * valid], color=color, density=True, bins='auto',
                     histtype='step', orientation='horizontal', linewidth=0.75)

    main_ax.legend(loc='lower right', prop={'size': 'xx-small'})

    # make point size legend
    legend_lumfracs = np.array([.05, .1, .25, .5])

    sc_pri = [main_ax.scatter(
                  [], [], c='r', marker='o', s=10. * frac, edgecolor='None', alpha=0.5)
              for frac in legend_lumfracs]
    sc_sec = [main_ax.scatter(
                  [], [], c='b', marker='D', s=10. * frac, edgecolor='None', alpha=0.5)
              for frac in legend_lumfracs]

    merged_markers = list(zip(sc_pri, sc_sec))
    merged_labels = [r'{:.0f}\%'.format(frac * 100.) for frac in legend_lumfracs]
    hmap = {tuple: HandlerTuple(ndivide=None)}
    lumfrac_legend = hist_ax.legend(
        list(zip(sc_pri, sc_sec)), merged_labels,
        handler_map=hmap, loc='upper right', prop={'size': 'xx-small'},
        title='\% flux outside IFU', title_fontsize='xx-small')

    main_ax.tick_params(labelsize='xx-small')
    main_ax.set_xlabel(r'${}-{}$'.format(cb1, cb2), size='x-small')
    main_ax.set_ylabel(r'$\log{\frac{\Upsilon^*_{\rm CMLR}}{\Upsilon^*_{\rm ring}}}$',
                       size='x-small')
    main_ax.set_xlim([0.1, 0.9])
    main_ax.set_ylim([-0.5, 0.5])
    fig.suptitle(r'$\Upsilon^*_{\rm CMLR}$ vs $\Upsilon^*_{\rm ring}$', size='small')
    fig.savefig(os.path.join(csp_basedir, 'lib_diags/', 'outer_ml.png'))

def make_missing_mass_fig(tab, mltype='ring', mlb='i', cb1='g', cb2='r'):
    '''figure with comparison between aperture correction two M/L
    
    Parameters
    ----------
    tab : astropy.table.Table
        full aggregation results table
    mltype : {str}
        'ring' or 'cmlr', the M/L applied to the ouside flux
        (the default is 'ring')
    mlb : {str}, optional
        bandpass for mass-to-light ratio
        (the default is 'i')
    cb1 : {str}, optional
        bluer bandpass for color (the default is 'g', SDSS g band)
    cb2 : {str}, optional
        redder bandpass for color (the default is 'r', SDSS r band)
    '''
    cb1_nsa_mag = tab[f'mag_nsa_{cb1}']
    cb2_nsa_mag = tab[f'mag_nsa_{cb2}']

    broadband_color = cb1_nsa_mag - cb2_nsa_mag
    outermass = (tab[f'logsollum_outer_{mlb}'] + tab[f'outerml_{mltype}']).to(
        u.Msun)

    outermass_frac = outermass / (
        tab['mass_in_ifu'].to(u.Msun) + outermass.to(u.Msun))

    valid = np.isfinite(tab['outerml_{}'.format(mltype)])

    primarysample = m.mask_from_maskbits(tab['mngtarg1'], [10])
    secondarysample = m.mask_from_maskbits(tab['mngtarg1'], [11])

    fig, main_ax, hist_ax = make_panel_hist(top=0.875)

    for selection, label, marker, color in zip(
        [primarysample, secondarysample], ['Primary+', 'Secondary'],
        ['o', 'D'], ['r', 'b']):
    
        main_ax.scatter(
            x=broadband_color[selection * valid], y=outermass_frac[selection * valid],
            c=color, edgecolor='None', s=3., marker=marker, label=label,
            alpha=0.5)

        hist_ax.hist(outermass_frac[selection * valid], color=color, density=True, bins='auto',
                     histtype='step', orientation='horizontal', linewidth=0.75)

    main_ax.set_xlim([0.1, 0.9])

    main_ax.legend(loc='best', prop={'size': 'xx-small'})
    main_ax.tick_params(labelsize='xx-small')
    main_ax.set_xlabel(r'${}-{}$'.format(cb1, cb2), size='x-small')
    main_ax.set_ylabel('Stellar-mass fraction outside IFU', size='x-small')
    fig.suptitle('Inferred mass fraction outside IFU', size='small')
    fig.savefig(os.path.join(csp_basedir, 'lib_diags/', f'mass_outside_ifu_{mltype}.png'))

def make_missing_flux_fig(tab, mlb='i', cb1='g', cb2='r'):
    '''make figure comparing missing flux for P+ & S samples
    
    Figure shows P+ (red) and S (blue) samples,
    with their integrated color, fraction of bandpass flux outside IFU
    
    Parameters
    ----------
    tab : astropy.table.Table
        full aggregation results table
    mlb : {str}, optional
        bandpass for mass-to-light ratio
        (the default is 'i', which [default_description])
    cb1 : {str}, optional
        bluer bandpass for color (the default is 'g', SDSS g band)
    cb2 : {str}, optional
        redder bandpass for color (the default is 'r', SDSS r band)
    '''

    cb1_nsa_mag = tab[f'mag_nsa_{cb1}']
    cb2_nsa_mag = tab[f'mag_nsa_{cb2}']

    broadband_color = cb1_nsa_mag - cb2_nsa_mag
    outerlum_frac = tab[f'flux_outer_{mlb}'] / tab[f'flux_nsa_{mlb}']

    primarysample = m.mask_from_maskbits(tab['mngtarg1'], [10])
    secondarysample = m.mask_from_maskbits(tab['mngtarg1'], [11])

    fig, main_ax, hist_ax = make_panel_hist(top=0.875)

    for selection, label, marker, color in zip(
        [primarysample, secondarysample], ['Primary+', 'Secondary'],
        ['o', 'D'], ['r', 'b']):
    
        main_ax.scatter(
            x=broadband_color[selection], y=outerlum_frac[selection],
            c=color, edgecolor='None', s=3., marker=marker, label=label,
            alpha=0.5)

        hist_ax.hist(outerlum_frac[selection], color=color, density=True, bins='auto',
                     histtype='step', orientation='horizontal', linewidth=0.75)

    main_ax.set_xlim([0.1, 0.9])

    main_ax.legend(loc='best', prop={'size': 'xx-small'})
    main_ax.tick_params(labelsize='xx-small')
    main_ax.set_xlabel(r'${}-{}$'.format(cb1, cb2), size='x-small')
    main_ax.set_ylabel('Flux fraction outside IFU', size='x-small')
    fig.suptitle('Flux fraction outside IFU', size='small')
    fig.savefig(os.path.join(csp_basedir, 'lib_diags/', 'flux_outside_ifu.png'))

def compare_missing_mass(tab, mlb='i', cb1='g', cb2='r'):
    '''make figure showing effect of aperture correction on total mass
    
    Figure shows P+ (red) and S (blue) samples,
    with their integrated color, difference between aperture-corrected
    masses using CMLR and ring method
    
    Parameters
    ----------
    tab : astropy.table.Table
        full aggregation results table
    mlb : {str}, optional
        bandpass for mass-to-light ratio
        (the default is 'i', which [default_description])
    cb1 : {str}, optional
        bluer bandpass for color (the default is 'g', SDSS g band)
    cb2 : {str}, optional
        redder bandpass for color (the default is 'r', SDSS r band)
    '''
    cb1_nsa_mag = tab[f'mag_nsa_{cb1}']
    cb2_nsa_mag = tab[f'mag_nsa_{cb2}']

    broadband_color = cb1_nsa_mag - cb2_nsa_mag
    outermass_cmlr = (tab[f'logsollum_outer_{mlb}'] + tab['outerml_cmlr']).to(
        u.Msun)
    mass_cmlr = (tab['mass_in_ifu'].to(u.Msun) + outermass_cmlr)
    outermass_ring = (tab[f'logsollum_outer_{mlb}'] + tab['outerml_ring']).to(
        u.Msun)
    mass_ring = (tab['mass_in_ifu'].to(u.Msun) + outermass_ring)
    dlogmass_cmlr_ring = mass_cmlr.to(u.dex(u.Msun)) - mass_ring.to(u.dex(u.Msun))

    primarysample = m.mask_from_maskbits(tab['mngtarg1'], [10])
    secondarysample = m.mask_from_maskbits(tab['mngtarg1'], [11])

    fig, main_ax, hist_ax = make_panel_hist(top=0.9, left=.225)

    valid = np.isfinite(dlogmass_cmlr_ring)

    for selection, label, marker, color in zip(
        [primarysample, secondarysample], ['Primary', 'Secondary'],
        ['o', 'D'], ['r', 'b']):
    
        main_ax.scatter(
            x=broadband_color[selection * valid], y=dlogmass_cmlr_ring[selection * valid],
            c=color, edgecolor='None', s=3., marker=marker, label=label,
            alpha=0.5)

        hist_ax.hist(dlogmass_cmlr_ring[selection * valid], color=color, density=True, bins='auto',
                     histtype='step', orientation='horizontal', linewidth=0.75)

    main_ax.set_xlim(np.percentile(broadband_color[valid], [5., 99.]))
    main_ax.set_ylim(np.percentile(dlogmass_cmlr_ring[valid], [5., 99.]))

    main_ax.legend(loc='best', prop={'size': 'xx-small'})
    main_ax.tick_params(labelsize='xx-small')
    main_ax.set_xlabel(r'${}-{}$'.format(cb1, cb2), size='x-small')
    main_ax.set_ylabel(r'$\log \frac{M^{\rm tot}_{\rm CMLR}}{M^{\rm tot}_{\rm ring}}$',
                  size='x-small')
    fig.suptitle(r'Impact of aperture-correction on $M^{\rm tot}$', size='small')
    fig.savefig(os.path.join(csp_basedir, 'lib_diags/', 'mtot_compare_cmlr_ring.png'))

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

    cb1_nsa_mag = jointab[f'mag_nsa_{cb1}']
    cb2_nsa_mag = jointab[f'mag_nsa_{cb2}']

    broadband_color = cb1_nsa_mag - cb2_nsa_mag

    outer_mass = (jointab[f'outerml_{mltype}'] + \
                  jointab[f'logsollum_outer_{mlb}']).to(u.dex(u.Msun))
    mass_pca = jointab['mass_in_ifu'].to(u.Msun) + outer_mass.to(u.Msun)

    nsa_h = 1.
    mass_nsa = (jointab['nsa_elpetro_mass'] * u.Msun * (nsa_h * u.littleh)**-2).to(
        u.Msun, u.with_H0(cosmo.H0))

    jhumpa_h = 1. / .7
    chabrier_to_kroupa_dex = .05
    mass_jhumpa = (10.**(jointab['LOG_MSTAR'] + chabrier_to_kroupa_dex) * \
                   u.Msun * (jhumpa_h * u.littleh)**-2.).to(u.Msun, u.with_H0(cosmo.H0))

    lowess_grid = np.linspace(np.nanmin(broadband_color), np.nanmax(broadband_color), 100).value
    lowess_pca_nsa, swt_nsa = smooth(
        x=broadband_color.value, y=np.log10(mass_pca / mass_nsa).value,
        xgrid=lowess_grid, bw=.01)
    lowess_pca_jhumpa, swt_jhumpa = smooth(
        x=broadband_color.value, y=np.log10(mass_pca / mass_jhumpa).value,
        xgrid=lowess_grid, bw=.01)
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

    fig.savefig(os.path.join(csp_basedir, 'lib_diags/', 'dMasses.png'), dpi=fig.dpi)

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
    '''global-local plot: mean tau, dmass, sig tau
    
    make figure plotting mass deficit on y,
    mass weighted atten on x, colored by std of mass weighted atten
    
    Parameters
    ----------
    tab : astropy.table.Table
        full aggregation results table
    mlb : {str}, optional
        mass-to-light bandpass (the default is 'i', SDSS i-band)
    '''

    fig, ax, cax, hist_ax = make_panel_hcb_hist(figsize=(3, 3), dpi=300)

    logmass_in_ifu = tab['mass_in_ifu'].to(u.dex(u.Msun))
    logmass_in_ifu_lw = tab['ml_fluxwt'] + tab[f'logsollum_in_ifu_{mlb}']
    std_atten_mwtd = tab['std_atten_mwtd']
    mean_atten_mwtd = tab['mean_atten_mwtd']

    sc = ax.scatter(
        x=std_atten_mwtd, y=(logmass_in_ifu - logmass_in_ifu_lw), c=mean_atten_mwtd,
        edgecolor='k', linewidths=.125, s=1., cmap='viridis_r', alpha=0.5,
        norm=mcolors.LogNorm(), vmin=.5, vmax=5.)

    ax.set_ylim([-.1, 0.3])

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
        os.path.join(csp_basedir, 'lib_diags/', 'stdtauV_dMglobloc_meantauV.png'),
        dpi=fig.dpi)

def make_stdtauV_vs_dMass_ba_fig(tab, mlb='i'):
    '''global-local plot: mean tau, dmass, b/a
    
    make figure plotting mass deficit on y,
    mass weighted atten on x, colored by NSA b/a
    
    Parameters
    ----------
    tab : astropy.table.Table
        full aggregation results table
    mlb : {str}, optional
        mass-to-light bandpass (the default is 'i', SDSS i-band)
    '''

    fig, ax, cax, hist_ax = make_panel_hcb_hist(figsize=(3, 3), dpi=300)

    logmass_in_ifu = tab['mass_in_ifu'].to(u.dex(u.Msun))
    logmass_in_ifu_lw = tab['ml_fluxwt'] + tab[f'logsollum_in_ifu_{mlb}']
    std_atten_mwtd = tab['std_atten_mwtd']
    mean_atten_mwtd = tab['mean_atten_mwtd']
    ba = tab['nsa_elpetro_ba']

    sc = ax.scatter(
        x=std_atten_mwtd, y=(logmass_in_ifu - logmass_in_ifu_lw), c=ba,
        edgecolor='k', linewidths=.125, s=1., cmap='viridis_r', alpha=0.5,
        vmin=.15, vmax=.8)

    ax.set_ylim([-.1, 0.3])

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
        os.path.join(csp_basedir, 'lib_diags/', 'stdtauV_dMglobloc_ba.png'),
        dpi=fig.dpi)

def make_meanstdtauV_vs_dMass_fig(tab, mlb='i'):
    '''global-local plot: sig tau, mean tau, dmass
    
    make figure plotting mass weighted atten on x,
    std of mass weighted atten on y, colored by mass deficit
    
    Parameters
    ----------
    tab : astropy.table.Table
        full aggregation results table
    mlb : {str}, optional
        mass-to-light bandpass (the default is 'i', SDSS i-band)
    '''

    fig, ax, cax, hist_ax = make_panel_hcb_hist(figsize=(3, 3), dpi=300)

    logmass_in_ifu = tab['mass_in_ifu'].to(u.dex(u.Msun))
    logmass_in_ifu_lw = tab['ml_fluxwt'] + tab[f'logsollum_in_ifu_{mlb}']
    std_atten_mwtd = tab['std_atten_mwtd']
    mean_atten_mwtd = tab['mean_atten_mwtd']

    sc = ax.scatter(
        y=std_atten_mwtd, x=mean_atten_mwtd, c=(logmass_in_ifu - logmass_in_ifu_lw),
        edgecolor='k', linewidths=.125, s=1., cmap='viridis_r', alpha=0.5,
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
        os.path.join(csp_basedir, 'lib_diags/', 'mean+stdtauV_dMglobloc.png'),
        dpi=fig.dpi)

def make_stdtauV_vs_dMass_ssfrsd_fig(tab, sfrsd_tab, mltype='ring', mlb='i'):
    '''global-local plot: sig tau, dmass, ssfrsd
    
    [description]
    
    Parameters
    ----------
    tab : astropy.table.Table
        full aggregation results table
    sfrsd_tab : astropy.table.Table
        specific star formation rate table
    mlb : {str}, optional
        mass-to-light bandpass (the default is 'i', SDSS i-band)
    mltype : {str}
        'ring' or 'cmlr', the M/L applied to the ouside flux
        (the default is 'ring')
    '''
    merge_tab = t.join(tab, sfrsd_tab, 'plateifu')

    fig, ax, cax, hist_ax = make_panel_hcb_hist(figsize=(3, 3), dpi=300, top=.8)

    logmass_in_ifu = merge_tab['mass_in_ifu'].to(u.dex(u.Msun))
    logmass_in_ifu_lw = merge_tab['ml_fluxwt'] + merge_tab[f'logsollum_in_ifu_{mlb}']
    std_atten_mwtd = merge_tab['std_atten_mwtd']
    mean_atten_mwtd = merge_tab['mean_atten_mwtd']
    ha_corr = np.exp(merge_tab['mean_atten_mwtd'] * (6563 / 5500)**-1.3)
    sfrsd = merge_tab['sigma_sfr'] * ha_corr * u.Msun / u.yr / u.pc**2
    outer_mass = (merge_tab[f'outerml_{mltype}'] + \
                  merge_tab[f'logsollum_outer_{mlb}']).to(u.Msun)
    mass_pca = merge_tab['mass_in_ifu'].to(u.Msun) + outer_mass
    ssfrsd = sfrsd / mass_pca

    sc = ax.scatter(
        x=std_atten_mwtd, y=(logmass_in_ifu - logmass_in_ifu_lw),
        c=ssfrsd.to(u.dex(ssfrsd.unit)),
        edgecolor='k', linewidths=.125, s=1., cmap='viridis_r', alpha=0.5,
        vmin=-15., vmax=-10.)

    ax.set_ylim([-.1, 0.3])

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
    fig.subplots_adjust(left=0.25)

    fig.savefig(
        os.path.join(csp_basedir, 'lib_diags/', 'stdtauV_dMglobloc_ssfrsd.png'),
        dpi=fig.dpi)

def make_stdtauV_vs_ssfrsd_dMass_fig(tab, sfrsd_tab, mltype='ring', mlb='i'):
    merge_tab = t.join(tab, sfrsd_tab, 'plateifu')

    mlb_ix = totalmass.StellarMass.bands_ixs[mlb]
    absmag_sun_mlb = totalmass.StellarMass.absmag_sun[mlb_ix]

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)

    logmass_in_ifu = merge_tab['mass_in_ifu'].to(u.dex(u.Msun))
    logmass_in_ifu_lw = merge_tab['ml_fluxwt'] + merge_tab[f'logsollum_in_ifu_{mlb}']
    std_atten_mwtd = merge_tab['std_atten_mwtd']
    mean_atten_mwtd = merge_tab['mean_atten_mwtd']
    ha_corr = np.exp(merge_tab['mean_atten_mwtd'] * (6563 / 5500)**-1.3)
    sfrsd = merge_tab['sigma_sfr'] * ha_corr * u.Msun / u.yr / u.pc**2
    outer_mass = (merge_tab[f'outerml_{mltype}'] + \
                  merge_tab[f'logsollum_outer_{mlb}']).to(u.Msun)
    mass_pca = merge_tab['mass_in_ifu'].to(u.Msun) + outer_mass
    ssfrsd = sfrsd / mass_pca

    sc = ax.scatter(
        x=np.log10(std_atten_mwtd), c=(logmass_in_ifu - logmass_in_ifu_lw),
        y=ssfrsd.to(u.dex(ssfrsd.unit)),
        edgecolor='k', linewidths=.125, s=1., cmap='viridis_r', alpha=0.5,
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
    fig.subplots_adjust(left=.225, bottom=.125, right=.9, top=.925)

    fig.savefig(
        os.path.join(csp_basedir, 'lib_diags/', 'stdtauV_ssfrsd_dMglobloc.png'),
        dpi=fig.dpi)

def make_meantauV_vs_ba_fig(tab):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)

    mean_atten_mwtd = tab['mean_atten_mwtd']
    ba = tab['nsa_elpetro_ba']
    mask = np.logical_or.reduce((~np.isfinite(ba), ~np.isfinite(mean_atten_mwtd),
                                 (ba < 0.)))

    sc = ax.scatter(
        x=np.log10(ba[~mask]), y=np.log10(mean_atten_mwtd[~mask]),
        edgecolor='k', linewidths=.125, s=1., label='PCA fits', alpha=0.5)
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
        os.path.join(csp_basedir, 'lib_diags/', 'meantauV_ba.png'),
        dpi=fig.dpi)

def fit_dlogM_mw(tab, sfrsd_tab, mltype='ring', mlb='i'):
    merge_tab = t.join(tab, sfrsd_tab, 'plateifu')
    is_agn = m.mask_from_maskbits(merge_tab['mngtarg3'], [1, 2, 3, 4])

    mlb_ix = totalmass.StellarMass.bands_ixs[mlb]
    absmag_sun_mlb = totalmass.StellarMass.absmag_sun[mlb_ix]

    logmass_in_ifu = merge_tab['mass_in_ifu'].to(u.dex(u.Msun))
    logmass_in_ifu_lw = merge_tab['ml_fluxwt'] + merge_tab[f'logsollum_in_ifu_{mlb}']
    merge_tab['dlogmass_lw'] = logmass_in_ifu - logmass_in_ifu_lw
    std_atten_mwtd = merge_tab['std_atten_mwtd']
    mean_atten_mwtd = merge_tab['mean_atten_mwtd']
    ha_corr = np.exp(merge_tab['mean_atten_mwtd'] * (6563 / 5500)**-1.3)
    sfrsd = merge_tab['sigma_sfr'] * ha_corr * u.Msun / u.yr / u.pc**2
    outer_mass = (merge_tab[f'outerml_{mltype}'] + \
                  merge_tab[f'logsollum_outer_{mlb}']).to(u.Msun)
    mass_pca = merge_tab['mass_in_ifu'].to(u.Msun) + outer_mass
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

def mag_to_dexmasstolight(q, cmlr_poly):
    qmag = q.to('mag')

    return np.polyval(cmlr_poly, qmag.value) * u.dex(m.m_to_l_unit)


if __name__ == '__main__':
    mlband = 'i'

    aggman = MassAggregationManager(
        cspbase='/usr/data/minhas2/zpace/sdss/sas/mangawork/manga/sandbox/mangapca/zachpace/CSPs_CKC14_MaNGA_20190215-1/',
        globstring='**/*-*/*-*_zpres.fits',
        masstable_fname_base='v2_5_3/2.3.0/masstables/{}.ecsv')
    res_fnames, res_plateifus = aggman.find(redo=False)
    nres_to_agg = len(res_fnames)

    if nres_to_agg > 0:
        print('aggregating {}'.format(nres_to_agg))
        ProgressBar.map(
            _aggregate_into_table_file,
            list(zip(res_fnames, [mlband, ] * nres_to_agg, [aggman.cspbase, ] * nres_to_agg,
                     res_plateifus)))

    mass_table = t.vstack(
        [t.QTable.read(fn, format='ascii.ecsv') for fn in glob(
            os.path.join(aggman.cspbase, aggman.masstable_fname_base.format('*')))])

    mass_table['distmod'] = cosmo.distmod([drpall.loc[obj]['nsa_zdist'] for obj in mass_table['plateifu']])

    for band in 'griz':
        outerfluxname = f'flux_outer_{band}'
        outermagname = f'mag_outer_{band}'
        outersollumname = f'sollum_outer_{band}'
        outerlogsollumname = f'logsollum_outer_{band}'

        for coltype in ['nsa', 'in_ifu']:
            sollumname = f'sollum_{coltype}_{band}'
            logsollumname = f'logsollum_{coltype}_{band}'
            absmagname = f'absmag_{coltype}_{band}'
            magname = f'mag_{coltype}_{band}'
            fluxname = f'flux_{coltype}_{band}'

            absmag_sun = totalmass.StellarMass.absmag_sun[
                totalmass.StellarMass.bands_ixs[band]]

            mass_table[logsollumname] = mass_table[sollumname].to(
                u.dex('bandpass_solLum'))
            mass_table[absmagname] = mass_table[logsollumname].to(
                u.ABmag, totalmass.bandpass_flux_to_solarunits(
                    absmag_sun))
            mass_table[magname] = mass_table[absmagname] + mass_table['distmod']
            mass_table[fluxname] = mass_table[magname].to(m.Mgy)

        mass_table[outerfluxname] = (
            mass_table[f'flux_nsa_{band}'] - \
            mass_table[f'flux_in_ifu_{band}']).clip(0. * m.Mgy, np.inf * m.Mgy)
        mass_table[outermagname] = mass_table[outerfluxname].to(u.ABmag)
        mass_table[outersollumname] = (mass_table[f'sollum_nsa_{band}'] - \
                                       mass_table[f'sollum_in_ifu_{band}']).clip(
                                           0. * m.bandpass_sol_l_unit, 
                                           np.inf * m.bandpass_sol_l_unit)
        mass_table[outerlogsollumname] = mass_table[outersollumname].to(
            u.dex('bandpass_solLum'))

    mass_table['ml_fluxwt'] = mass_table['ml_fluxwt'].to(
        u.dex('mass_to_light'))
    mass_table['outerml_ring'] = mass_table['outerml_ring'].to(
        u.dex('mass_to_light'))

    # use the CMLR found in Paper I
    cmlr_gr_i = totalmass.cmlr_kwargs['cmlr_poly']
    cb1, cb2 = totalmass.cmlr_kwargs['cb1'], totalmass.cmlr_kwargs['cb2']
    mass_table['outerml_cmlr'] = mag_to_dexmasstolight(
        mass_table[f'mag_outer_{cb1}'] - mass_table[f'mag_outer_{cb2}'],
        cmlr_gr_i)
    mass_table['outerml_diff'] = mass_table['outerml_cmlr'] - mass_table['outerml_ring']

    drpall.keep_columns(['plateifu', 'mangaid', 'objra', 'objdec', 'ebvgal', 
                         'mngtarg1', 'mngtarg2', 'mngtarg3', 'nsa_iauname', 'ifudesignsize',
                         'nsa_z', 'nsa_zdist', 'nsa_nsaid', 'nsa_elpetro_ba',
                         'nsa_elpetro_mass', 'nsa_elpetro_absmag'])
    full_table = t.join(mass_table, drpall, 'plateifu', join_type='inner')

    single_aper_mass = (
        full_table['ml_fluxwt'] + \
        full_table['logsollum_in_ifu_{}'.format(mlband)]).to(u.dex(u.Msun))

    full_table['single_aper_mass_deficit'] = \
        (full_table['mass_in_ifu'].to(u.dex(u.Msun)) - \
         single_aper_mass).to(u.dex)
    mass_deficit_order = np.argsort(full_table['single_aper_mass_deficit'])[::-1]

    #'''
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
    #'''

    mass_table_loc = os.path.join(
        aggman.cspbase, os.path.dirname(aggman.masstable_fname_base), 'totalmass.fits')
    mass_table_to_write = mass_table['plateifu', 'mass_in_ifu']
    mass_table_to_write['mass_outer_ring'] = (
        mass_table['outerml_ring'] + mass_table[f'logsollum_outer_{mlband}']).to(u.Msun)
    mass_table_to_write['mass_outer_cmlr'] = (
        mass_table['outerml_cmlr'] + mass_table[f'logsollum_outer_{mlband}']).to(u.Msun)
    mass_table_to_write['mass_outer_cmlr'][
    ~np.isfinite(mass_table_to_write['mass_outer_cmlr'])] = 0. * u.Msun

    mass_table_to_write.write(mass_table_loc, overwrite=True)