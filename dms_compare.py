import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

import astropy.table as t
from astropy import units as u, constants as c

from importer import *
import manga_tools as m
import smsd

from radial import radial_gp

def make_manga_dms_comparison(res, pca_system, band, cosmo, drpall_row,
                              dms_label, dms_fname):
    fig, ax = plt.subplots(figsize=(3, 2), dpi=300)

    smsd_radial_plot(res, pca_system, band, cosmo, drpall_row, ax)
    overplot_dms_dmsd(dms_fname, ax, dms_label)
    ax.legend(loc='best', prop={'size': 'x-small'})
    ax.set_title('MaNGA-DMS Mass Surf. Dens. Comparison', size='x-small')

    return fig, ax

def smsd_radial_plot(res, pca_system, band, cosmo, drpall_row, ax, h_old, incl_over=None):
    smsd_map, smsd_lunc_map, smsd_uunc_map = smsd.smsd(
        res, pca_system, band, cosmo, drpall_row, incl_over)

    mask = np.logical_or(res.mask, res.badPDF())

    dap_data = m.load_dap_maps(
        plate=drpall_row['plate'], ifu=drpall_row['ifudsgn'], mpl_v=mpl_v,
        kind='SPX-GAU-MILESHC')

    r_ang = dap_data['SPX_ELLCOO'].data[0, ...] * u.arcsec
    r_lin = (cosmo.kpc_proper_per_arcmin(drpall_row['nsa_zdist']) * r_ang).to(u.kpc)

    r_corr = h_old / cosmo.h

    ax.errorbar(
        x=np.ma.array(r_lin.value, mask=mask).flatten() * r_corr,
        y=smsd_map.value.flatten(),
        yerr=np.row_stack([smsd_lunc_map.value.flatten(),
                           smsd_uunc_map.value.flatten()]),
        linestyle='None', marker='o', markerfacecolor='g', ecolor='g',
        markeredgecolor='None', elinewidth=0.5, markersize=2., capsize=2.,
        label=r'MaNGA $M^*$: {}'.format(res[0].header['PLATEIFU']), capthick=.5,
        errorevery=20, markevery=20, alpha=.5, zorder=1)

    ax.set_xlabel(r'$R ~ {\rm [kpc]}$', size='x-small')
    ax.set_ylabel(r'$\Sigma ~ {\rm [\frac{M_\odot}{pc^{-2}}]}$',
                  size='x-small')
    ax.set_yscale('log')

    dap_data.close()

def overplot_dms_dmsd(fname, ax, label, h_old, h_new, factor=1., datacolor='k', linestyle='-'):
    dms_table = t.Table.read(
        fname, format='ascii', guess=False, delimiter=' ', comment='%',
        names=['R_arcsec', 'R_kpc', 'sigma_dyn', 'd_sigma_dyn'], data_start=0)
    sigma_dyn_corr = h_new / h_old

    if factor == 1.:
        datalabel = r'DMS $M_{{dyn}}$: {}'.format(label)
    else:
        datalabel = r'DMS $M_{{dyn}}$: {} ($f_{{h_z}} = {:.2f}$)'.format(label, factor)

    if factor == 1.:
        ax.errorbar(dms_table['R_kpc'], dms_table['sigma_dyn'] * sigma_dyn_corr * factor,
                    yerr=dms_table['d_sigma_dyn'], linestyle='-', color=datacolor,
                    marker='D', markerfacecolor=datacolor, ecolor=datacolor, linewidth=0.5,
                    markeredgecolor='None', capthick=.5, elinewidth=.5,
                    markersize=3., capsize=3., zorder=2,
                    label=datalabel)
    else:
        ax.plot(dms_table['R_kpc'], dms_table['sigma_dyn'] * sigma_dyn_corr * factor,
                linestyle=linestyle, color=datacolor,
                marker='None', linewidth=0.5, zorder=2, label=datalabel)

    return dms_table

def find_galaxy_hz_scaling(res, pca_system, dms_table, cosmo, drpall_row, h_old,
                           Fbar0, Rbulge, band='i', incl_over=None):
    '''
    Find the optimal disk scale-height scaling for one galaxy
    '''

    # get stellar mass SD from PCA results
    smsd_map, smsd_lunc_map, smsd_uunc_map = smsd.smsd(
        res, pca_system, band, cosmo, drpall_row, incl_over)

    mask = np.logical_or(res.mask, res.badPDF())
    smsd_a, smsd_lunc_a, smsd_uunc_a = smsd_map[~mask], smsd_lunc_map[~mask], \
                                       smsd_uunc_map[~mask]
    smsd_unc_a = 0.5 * (smsd_lunc_a + smsd_uunc_a)

    # transform angular bins to linear
    r_corr = h_old / cosmo.h
    radialbin_edges_ang = np.array([0., 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5])
    radialbin_ang_to_lin = np.polyfit(x=dms_table['R_arcsec'], y=dms_table['R_kpc'] * r_corr,
                                      deg=1, full=False, cov=False)
    radialbin_edges_lin = np.polyval(radialbin_ang_to_lin, radialbin_edges_ang)
    radialbin_edges_lin[0] = 0.
    radialbin_ctrs = 0.5 * (radialbin_edges_lin[1:] + radialbin_edges_lin[:-1])
    bin_within_bulge = radialbin_ctrs <= Rbulge * r_corr

    # get MaNGA bin radius from center (in kpc)
    dap_data = res.get_dap_maps(mpl_v, 'SPX-GAU-MILESHC')
    r_ang = dap_data['SPX_ELLCOO'].data[0, ...] * u.arcsec
    r_lin = (cosmo.kpc_proper_per_arcmin(drpall_row['nsa_zdist']) * r_ang).to(u.kpc)[~mask]
    dap_data.close()

    # mean, standard-deviation, and counts within each DMS bin
    n_in_bin, *_ = stats.binned_statistic(
        x=r_lin.value, values=smsd_a.value,
        bins=radialbin_edges_lin, statistic='count')

    mean_smsd_in_bin, *_ = stats.binned_statistic(
        x=r_lin.value, values=smsd_a.value, bins=radialbin_edges_lin,
        range=[0., radialbin_edges_lin[-1]], statistic='mean')
    std_smsd_in_bin, *_ = stats.binned_statistic(
        x=r_lin.value, values=smsd_a.value, bins=radialbin_edges_lin,
        range=[0., radialbin_edges_lin[-1]],
        statistic=np.std)

    h_new = cosmo.h
    sigma_dyn_corr = h_new / h_old
    dmsd = np.concatenate(
        [dms_table['sigma_dyn'] * sigma_dyn_corr,
         np.zeros(len(radialbin_ctrs) - len(dms_table))])
    usebin = (n_in_bin >= 1) * ~bin_within_bulge

    Sdyn2Sstel = lambda Sdyn, fhz: (Sdyn * fhz)
    popt, pcov = curve_fit(
        f=Sdyn2Sstel, xdata=dmsd[usebin], ydata=mean_smsd_in_bin[usebin],
        sigma=std_smsd_in_bin[usebin], absolute_sigma=True, p0=[2.])

    return popt, pcov


if __name__ == '__main__':
    import read_results
    import os

    pca_system = read_results.PCASystem.fromfile(os.path.join(basedir, 'pc_vecs.fits'))
    drpall = m.load_drpall(mpl_v, index='plateifu')
    all_dms_galaxies = drpall[m.mask_from_maskbits(drpall['mngtarg3'], [16])]
    h_dms = 0.7

    fig, (ax1, ax2, ax3, ax4, ax5, _) = plt.subplots(
        nrows=2, ncols=3, figsize=(7, 8), sharex=True, sharey=True)
    for plateifu, dms_name, ax, Rbulge, F, incl in zip(
        ['8566-12705', '8567-12701', '8939-12704', '10494-12705', '10510-12704'],
        ['UGC3997', 'UGC4107', 'UGC4368', 'UGC4380', 'UGC6918'],
        [ax1, ax2, ax3, ax4, ax5],
        [1.97, 1.18, 2.23, 2.44, 0.655],
        [.48, .56, .72, .46, .64],
        np.array([26.2, 24.1, 45.3, 14.2, 38.0]) * u.deg):

        ugc_num = dms_name[3:]
        dms_fname = \
            '/usr/data/minhas/zpace/stellarmass_pca/diskmass/U0{}.Sigma.dyn.tbl'.format(
                ugc_num)
        drpall_row = drpall.loc[plateifu]
        res = read_results.PCAOutput.from_plateifu(
            os.path.join(basedir, 'results'),
            drpall_row['plate'], drpall_row['ifudsgn'])
        smsd_radial_plot(
            res, pca_system, 'i', WMAP9, drpall_row, ax, h_old=h_dms, incl_over=incl)
        dms_table = overplot_dms_dmsd(dms_fname, ax, dms_name, h_old=h_dms, h_new=WMAP9.h)

        f_hz_factor_1_5 = 1.5
        F_1_5 = F * np.sqrt(f_hz_factor_1_5)
        f_hz_factor_2_5 = 2.5
        F_2_5 = F * np.sqrt(f_hz_factor_2_5)

        '''
        ax.plot([0.05 * ax.get_xlim()[1], 0.05 * ax.get_xlim()[1]],
                [1.5 * ax.get_ylim()[0], 1.5 * ax.get_ylim()[0] * f_hz_factor_1_5],
                c='k', linestyle='--', linewidth=0.5,
                label=r'${:.2g} h_z (\mathcal{{F}}_b^{{2.2 h_R}} = {:.2g})$'.format(
                    1. / f_hz_factor_1_5, F_1_5), zorder=0)
        ax.plot([0.1 * ax.get_xlim()[1], 0.1 * ax.get_xlim()[1]],
                [1.5 * ax.get_ylim()[0], 1.5 * ax.get_ylim()[0] * f_hz_factor_2_5],
                c='k', linestyle=':', linewidth=0.5,
                label=r'${:.2g} h_z (\mathcal{{F}}_b^{{2.2 h_R}} = {:.2g})$'.format(
                    1. / f_hz_factor_2_5, F_2_5), zorder=0)
        '''
        r_corr = h_dms / WMAP9.h
        ax.axvline(Rbulge * r_corr, linestyle='--', c='k')
        ax.tick_params(labelsize='xx-small')

        #'''
        # find preferred value
        popt, pcov = find_galaxy_hz_scaling(
            res, pca_system, dms_table, WMAP9, drpall_row,
            h_old=h_dms, Fbar0=F, Rbulge=Rbulge, incl_over=incl)
        print(' / '.join((plateifu, dms_name)))
        print('fhz: {:.3g} +/- {:.3g}'.format(popt[0], np.sqrt(np.diag(pcov))[0]))
        print('F0: {:.3g}'.format(F))
        print('F1: {:.3g}'.format(F * np.sqrt(popt[0])))
        #'''
        res.close()

        '''
        ax.plot([0.15 * ax.get_xlim()[1], 0.15 * ax.get_xlim()[1]],
                [1.5 * ax.get_ylim()[0], 1.5 * ax.get_ylim()[0] * popt[0]],
                c='k', linestyle='-', linewidth=0.5,
                label=r'${:.2g} h_z (\mathcal{{F}}_b^{{2.2 h_R}} = {:.2g})$'.format(
                    1. / popt[0], F * np.sqrt(popt[0])), zorder=0)
        '''
        _ = overplot_dms_dmsd(dms_fname, ax, dms_name, h_old=h_dms, h_new=WMAP9.h,
                              factor=1.1, datacolor='orange', linestyle='--')
        _ = overplot_dms_dmsd(dms_fname, ax, dms_name, h_old=h_dms, h_new=WMAP9.h,
                              factor=popt[0], datacolor='b', linestyle='--')
        _ = overplot_dms_dmsd(dms_fname, ax, dms_name, h_old=h_dms, h_new=WMAP9.h,
                              factor=1.5, datacolor='cyan', linestyle='--')

        ax.legend(prop={'size': 'xx-small'})

    fig.suptitle(
        'DiskMass-MaNGA SMSD Comparisons',
        size='x-small')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.15, top=.95, left=.14, right=.98)
    fig.savefig(os.path.join(basedir, 'diskmass/', 'dms_compare.png'))

