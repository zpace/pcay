from importer import *
import manga_tools as m
import spectrophot
from dered_drizzle import drizzle_flux
from totalmass import infer_masked
import read_results

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import Rbf
from scipy.optimize import newton as newtonroot

import os
import glob
import warnings
from astropy.io import fits
from astropy import wcs
from astropy import units as u, constants as c
import extinction

preim_loc = '/usr/data/minhas2/zpace/sdss/manga/preimaging/'

sdss_filters = spectrophot.filters.load_filters('sdss2010-*')

class MaNGA_PreImage(fits.HDUList):
    '''
    class for MaNGA pre-imaging
    '''

    @classmethod
    def from_designid_mangaid(cls, designid, mangaid):
        '''
        load from designid and mangaid, given in FITS header of LOGCUBE
        '''
        glob_search_string = os.path.join(
            preim_loc, '*/{}/preimage-{}.fits.gz'.format(designid, mangaid))
        possible_paths = glob.glob(glob_search_string)
        
        if len(possible_paths) == 0:
            raise OSError(
                'no matching preimaging to designid {}, mangaid {}'.format(
                    designid, mangaid))
        elif len(possible_paths) > 1:
            raise Warning(
                'more than one preimaging file matching designid {}, mangaid {}'.format(
                    designid, mangaid))

        return super().fromfile(possible_paths[0])

    def _dustcorr(self, b, ebvgal, dustlaw=extinction.fitzpatrick99):
        l = dict(zip(
            sdss_filters.names, sdss_filters.effective_wavelengths))['sdss2010-{}'.format(b)]
        r_v = 3.1
        a_v = r_v * ebvgal
        ext_mag = dustlaw(np.array([l.value]), a_v, r_v)
        return 10.**(0.4 * ext_mag)

    def img(self, b, dustcorr=False, ebvgal=0.):
        if not dustcorr:
            return self['{} img'.format(b)].data
        else:
            return self['{} img'.format(b)].data * self._dustcorr(b, ebvgal)

    def img_ivar(self, b, dustcorr=False, ebvgal=0.):
        if not dustcorr:
            return self['{} ivar'.format(b)].data
        else:
            return self['{} ivar'.format(b)].data / self._dustcorr(b, ebvgal)**2.

    def color(self, b1, b2, dustcorr=False, ebvgal=0.):
        return 2.5 * np.log10(self.img(b2, dustcorr, ebvgal) / self.img(b1, dustcorr, ebvgal))

def kcorr_spec_to_phot(drp_logcube, ifu_ivar_hdu, z, cb1='g', cb2='r', mlb='i'):
    '''
    figure out k-correction from spectroscopy, and then apply it to photometry
    '''
    ifu_wcs = wcs.WCS(ifu_ivar_hdu.header)
    ifu_AA, ifu_DD = ifu_wcs.wcs_pix2world(*np.meshgrid(
        *[np.linspace(0., s - 1., s) for s in ifu_ivar_hdu.data.shape]), 1)

    ebvgal = drp_logcube[0].header['EBVGAL']
    r_v = 3.1
    dustcorr = 10.**(0.4 * extinction.fitzpatrick99(
        drp_logcube['WAVE'].data, a_v=r_v * ebvgal, r_v=r_v))[:, None, None]

    sphot = dict()
    sphot['obs'] = spectrophot.Spec2Phot(
        lam=drp_logcube['WAVE'].data, flam=1.0e-17 * drp_logcube['FLUX'].data * dustcorr)
    sphot['rest'] = spectrophot.Spec2Phot(
        lam=drp_logcube['WAVE'].data / (1. + z),
        flam=1.0e-17 * drp_logcube['FLUX'].data * (1. + z) * dustcorr)

    obs_cb1 = sphot['obs'].ABmags['sdss2010-{}'.format(cb1)] * u.ABmag
    rest_cb1 = sphot['rest'].ABmags['sdss2010-{}'.format(cb1)] * u.ABmag
    obs_cb2 = sphot['obs'].ABmags['sdss2010-{}'.format(cb2)] * u.ABmag
    rest_cb2 = sphot['rest'].ABmags['sdss2010-{}'.format(cb2)] * u.ABmag
    obs_mlb = sphot['obs'].ABmags['sdss2010-{}'.format(mlb)] * u.ABmag
    rest_mlb = sphot['rest'].ABmags['sdss2010-{}'.format(mlb)] * u.ABmag

    kcorr_map_cb1 = (rest_cb1 - obs_cb1).clip(-.2, .2)
    kcorr_map_cb2 = (rest_cb2 - obs_cb2).clip(-.2, .2)
    kcorr_map_mlb = (rest_mlb - obs_mlb).clip(-.2, .2)

    good_phot = np.logical_and.reduce((
        ifu_ivar_hdu.data.astype(bool), np.isfinite(kcorr_map_mlb),
        np.isfinite(kcorr_map_cb1), np.isfinite(kcorr_map_cb2)))

    kcorr_cb1_interp = Rbf(ifu_AA[good_phot], ifu_DD[good_phot], kcorr_map_cb1[good_phot],
                           bounds_error=False, fill_value=0., function='thin_plate')
    kcorr_cb2_interp = Rbf(ifu_AA[good_phot], ifu_DD[good_phot], kcorr_map_cb2[good_phot],
                           bounds_error=False, fill_value=0., function='thin_plate')
    kcorr_mlb_interp = Rbf(ifu_AA[good_phot], ifu_DD[good_phot], kcorr_map_mlb[good_phot],
                           bounds_error=False, fill_value=0., function='thin_plate')

    return kcorr_cb1_interp, kcorr_cb2_interp, kcorr_mlb_interp


def make_stellarmass_aperture_plot(preimaging, dap_maps, drp_logcube, cmlr, drpall_row, mlb='i', cb1='g', cb2='r'):
    '''
    make aperture plot of included mass (from CMLR and preimaging) versus radius (from DAP)

    preimaging: MaNGA_PreImage object
    dap_maps: FITS HDUList with DAP MAPS outputs
    '''
    ebvgal = drpall_row['ebvgal']

    # WCS from DAP determines sampling for Re grid, and where k-correction map is sampled
    dap_wcs = wcs.WCS(dap_maps['SPX_SNR'].header)
    dap_AA, dap_DD = dap_wcs.wcs_pix2world(*np.meshgrid(
        *[np.linspace(0., s - 1., s) for s in dap_maps['SPX_SNR'].data.shape]), 1)

    # map sky coordinates to number of effective radii from center
    pos_to_nRe = RotatedParaboloid(
        ctr=np.array([drpall_row['objra'], drpall_row['objdec']]),
        phi=drpall_row['nsa_elpetro_phi'] * u.deg, axis_ratio=drpall_row['nsa_elpetro_ba'],
        Re=drpall_row['nsa_elpetro_th50_r'] / 3600.)

    # WCS from preimaging determines where elliptical projection is resampled
    preimaging_wcs = wcs.WCS(preimaging['{} img'.format(mlb)].header)
    preimaging_II, preimaging_JJ = np.meshgrid(
        *[np.linspace(0., s - 1., s)
          for s in preimaging.img(mlb).shape], indexing='ij')
    preimaging_AA, preimaging_DD = preimaging_wcs.wcs_pix2world(
        preimaging_JJ, preimaging_II, 1)

    # evaluate Re interpolation on imaging RA/Dec grid
    preimaging_Re = pos_to_nRe(
        np.column_stack([preimaging_AA.flatten(), preimaging_DD.flatten()])).reshape(
        preimaging_AA.shape)

    kcorr_cb1_interp, kcorr_cb2_interp, kcorr_mlb_interp = kcorr_spec_to_phot(
        drp_logcube, dap_maps['SPX_MFLUX_IVAR'], drpall_row['nsa_z'], cb1, cb2, mlb)

    kcorr_cb1_phot = kcorr_cb1_interp(
        preimaging_AA.flatten(), preimaging_DD.flatten()).reshape(preimaging_AA.shape) * u.mag
    kcorr_cb2_phot = kcorr_cb2_interp(
        preimaging_AA.flatten(), preimaging_DD.flatten()).reshape(preimaging_AA.shape) * u.mag
    kcorr_color = kcorr_cb1_phot - kcorr_cb2_phot
    kcorr_mlb_phot = kcorr_mlb_interp(
        preimaging_AA.flatten(), preimaging_DD.flatten()).reshape(preimaging_AA.shape) * u.mag

    # evaluate CMLR given preimaging
    ml_from_cmlr = 10.**cmlr(
        color=preimaging.color(
            cb1, cb2, dustcorr=True, ebvgal=ebvgal) + kcorr_color.value) * m.m_to_l_unit
    img_mag = (preimaging.img(mlb) * 1.0e-9 * m.Mgy).to(u.ABmag) + kcorr_mlb_phot
    distmod = cosmo.distmod(drpall_row['nsa_zdist'])
    img_MAG = img_mag - distmod

    mlb_sollum = 10.**(-0.4 * (img_MAG - spectrophot.absmag_sun_band[mlb] * u.ABmag)).value * \
                 m.bandpass_sol_l_unit
    mass_from_cmlr = (ml_from_cmlr * mlb_sollum).to(u.Msun)

    nRe_to_plot = np.linspace(0., 0.9 * preimaging_Re.max(), 1000)
    
    photomass_within_nRe = np.array([sum_within_nRe(mass_from_cmlr, preimaging_Re, n)
                                     for n in nRe_to_plot])
    
    # solve for number of effective radii enclosing all of NSA flux
    NSA_bands = 'FNugriz'
    NSA_bands_ixs = dict(zip(NSA_bands, range(len(NSA_bands))))
    NSA_MAG = drpall_row['nsa_elpetro_absmag'][NSA_bands_ixs[mlb]] * \
              (u.ABmag - u.MagUnit(u.littleh**2))
    NSA_kcorr_flux = (NSA_MAG.to(u.ABmag, u.with_H0(cosmo.H0)) + \
                      cosmo.distmod(drpall_row['nsa_zdist'])).to(m.Mgy)

    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(9, 7), dpi=150, sharex=True)
    ax1.plot(nRe_to_plot, photomass_within_nRe, linewidth=1., color='C0')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$\frac{M_{< R}}{M_{\odot}}$', size='small')

    ax2.scatter(preimaging_Re.flatten(), np.log10(ml_from_cmlr.value.flatten()),
                edgecolor='None', s=.5, c='C0')
    ax2.set_ylabel(r'$\log \Upsilon^*_{{{}}}$'.format(mlb), size='small')
    ax2.set_ylim([-2.5, 2.5])

    summed_flux = np.array([sum_within_nRe(
        arr=img_mag.to(m.Mgy), Re_a=preimaging_Re, nRe=n) for n in nRe_to_plot])
    img_var = 1. / preimaging.img_ivar(mlb) * (1.0e-9 * m.Mgy)**2
    quadsummed_fluxunc = np.sqrt(np.array([sum_within_nRe(
        arr=img_var, Re_a=preimaging_Re, nRe=n) for n in nRe_to_plot]))
    ax3.plot(nRe_to_plot, summed_flux, label='K-Corr. Phot.', color='C0')
    ax3.fill_between(
        nRe_to_plot, summed_flux - quadsummed_fluxunc, summed_flux + quadsummed_fluxunc,
        alpha=0.5, linewidth=0.25, color='C0')
    ax3.set_xlabel(r'$\frac{R}{R_e}$', size='small')
    ax3.set_ylabel(r'$\frac{F_{<R}}{\rm Mgy}$', size='small')
    ax3.set_yscale('log')

    fig.suptitle(drpall_row['plateifu'])

    return fig, ax1, ax2, ax3
    

def sum_within_nRe(arr, Re_a, nRe):
    '''
    sums contents of `arr` whose corresponding value in `Re_a` is less than `nRe`
    '''
    return (arr[(Re_a < nRe) * np.isfinite(arr)]).sum().value

def find_nRe_for_NSAflux(flux_a, Re_a, fluxlim, r0):
    '''
    find number of effective radii corresponding to the NSA flux listed
    '''
    objective = lambda n: sum_within_nRe(flux_a, Re_a, n) - fluxlim.value

    nRe = newtonroot(objective, x0=r0, tol=.01)
    return nRe


class RotatedParaboloid(object):
    def __init__(self, ctr, phi, axis_ratio, Re):
        self.ctr = ctr
        tilt = phi + 90. * u.deg
        self.tilt = tilt
        self.T = np.array([[np.cos(tilt), -np.sin(tilt)], [np.sin(tilt), np.cos(tilt)]])
        self.T_inv = np.linalg.inv(self.T)
        self.a = Re
        self.b = Re * axis_ratio

    def __call__(self, xy, which='r'):
        if which == 'r':
            ij = (xy - self.ctr) @ self.T_inv
            z = np.sqrt((ij[:, 0] / self.a)**2. + (ij[:, 1] / self.b)**2.)

        elif which == 'phi':
            ij = (xy - self.ctr) @ self.T_inv
            z = np.arctan2(ij[:, 1] * self.a / self.b, ij[:, 0])
        return z

def overplot_spectroscopic(res, drp_logcube, drpall_row, mass_ax, ml_ax, f_ax,
                           pca_system, res_wcs, cmlr, mlb='i'):
    '''
    overplot cumulative spectroscopic masses, log M/L, cumulative fluxes
    '''

    # synthetic photometry of IFU data in rest frame
    ebvgal = drpall_row['ebvgal']
    r_v = 3.1
    dustcorr = 10.**(0.4 * extinction.fitzpatrick99(
        drp_logcube['WAVE'].data, a_v=r_v * ebvgal, r_v=r_v))[:, None, None]

    ifu_s2p = spectrophot.Spec2Phot(
        lam=drp_logcube['WAVE'].data / (1. + drpall_row['nsa_z']),
        flam=1.0e-17 * drp_logcube['FLUX'].data * (1. + drpall_row['nsa_z']) * dustcorr)
    res_flux = (ifu_s2p.ABmags['sdss2010-{}'.format(mlb)] * u.ABmag).to(m.Mgy)

    # map sky coordinates to number of effective radii from center
    pos_to_nRe = RotatedParaboloid(
        ctr=np.array([drpall_row['objra'], drpall_row['objdec']]),
        phi=drpall_row['nsa_elpetro_phi'] * u.deg, axis_ratio=drpall_row['nsa_elpetro_ba'],
        Re=drpall_row['nsa_elpetro_th50_r'] / 3600.)

    badpdf = res.badPDF()
    ml_mask = np.logical_or(res.mask, badpdf)

    # WCS from results file determines where elliptical projection is sampled
    #res_wcs = wcs.WCS(res['SNRMED'].header)
    res_II, res_JJ = np.meshgrid(
        *[np.linspace(0., s - 1., s) for s in res['SNRMED'].shape], indexing='ij')
    res_AA, res_DD = res_wcs.wcs_pix2world(res_JJ, res_II, 1)
    
    # sample Re at spaxel centers
    res_Re = pos_to_nRe(
        np.column_stack([res_AA.flatten(), res_DD.flatten()])).reshape(res_AA.shape)
    nRe_to_plot = np.linspace(0., res_Re[~ml_mask].max(), 50.)

    # mass, log M/L
    badpdf = res.badPDF()
    ml_mask = np.logical_or.reduce((res.mask, badpdf))
    interior_mask = np.logical_or.reduce((
        badpdf, (m.mask_from_maskbits(drp_logcube['MASK'].data, [3]).sum(axis=0) > 0),
        (m.mask_from_maskbits(drp_logcube['MASK'].data, [2]).sum(axis=0) > 0)))
    logml = infer_masked(res.param_dist_med(extname='ML{}'.format(mlb)), ml_mask, interior_mask)
    ml = 10.**logml * m.m_to_l_unit
    res_MAG = res_flux.to(u.ABmag) - cosmo.distmod(drpall_row['nsa_zdist'])
    mlb_sollum = 10.**(-0.4 * (res_MAG - spectrophot.absmag_sun_band[mlb] * u.ABmag)).value * \
                 m.bandpass_sol_l_unit
    spectro_mass = (ml * mlb_sollum).to(u.Msun)

    # sum mass within some number of Re
    mass_within_nRe = np.array([sum_within_nRe(spectro_mass, res_Re, n)
                                for n in nRe_to_plot])
    mass_ax.plot(nRe_to_plot, mass_within_nRe, c='C1')

    # plot mass to light versus Re
    ml_sc = ml_ax.scatter(res_Re[~ml_mask], logml[~ml_mask], c='C1', edgecolor='None', s=.5)

    res_Re_m = np.ma.array(res_Re, mask=ml_mask)
    # find ring method log M/L
    outer_ring = np.logical_and((res_Re_m <= res_Re_m.max()), (res_Re_m >= res_Re_m.max() - .5))
    outer_logml_ring = np.median(logml[~ml_mask * outer_ring])
    ml_ax.scatter(x=[np.median(res_Re_m[~ml_mask * outer_ring])], y=[outer_logml_ring],
                  marker='x', c='C1')
    # find CMLR log M/L
    nsa_MAG = (np.array(drpall_row['nsa_elpetro_absmag'][2:]) * \
                        (u.ABmag - u.MagUnit(u.littleh**2))).to(u.ABmag, u.with_H0(cosmo.H0))
    nsa_mag = (nsa_MAG + cosmo.distmod(drpall_row['nsa_zdist']))
    nsa_flux = nsa_mag.to(m.Mgy)
    ifu_mag = np.array(
        [ifu_s2p.ABmags['sdss2010-{}'.format(b_)] for b_ in 'ugriz']) * u.ABmag
    ifu_mag[~np.isfinite(ifu_mag)] = 40. * u.ABmag
    ifu_flux = ifu_mag.to(m.Mgy)
    flux_deficit = nsa_flux - ifu_flux.sum(axis=(1, 2))
    logml_missingflux_cmlr = cmlr(2.5 * np.log10(flux_deficit[1] / flux_deficit[2]))
    ml_ax.scatter(x=[np.median(res_Re_m[~ml_mask * outer_ring])], y=[logml_missingflux_cmlr],
                  marker='s', c='C1', edgecolor='k')

    #
    ml_missingflux_cmlr = 10.**logml_missingflux_cmlr * m.m_to_l_unit
    ml_missingflux_ring = 10.**outer_logml_ring * m.m_to_l_unit
    missing_mlb_MAG = flux_deficit[3].to(u.ABmag) - cosmo.distmod(drpall_row['nsa_zdist'])
    missing_mlb_sollum = 10.**(-0.4 * (missing_mlb_MAG - spectrophot.absmag_sun_band[mlb] * u.ABmag)).value * \
        m.bandpass_sol_l_unit
    missing_mass_cmlr = (ml_missingflux_cmlr * missing_mlb_sollum).to(u.Msun)
    missing_mass_ring = (ml_missingflux_ring * missing_mlb_sollum).to(u.Msun)

    mass_ax.scatter(x=res_Re.max() - .1,
                    y=mass_within_nRe[-1] * u.Msun + missing_mass_cmlr,
                    marker='s', c='C1', edgecolor='k', zorder=3)
    mass_ax.scatter(x=res_Re.max(),
                    y=mass_within_nRe[-1] * u.Msun + missing_mass_ring,
                    marker='x', c='C1', edgecolor='k', zorder=3)

    summed_flux = np.array([sum_within_nRe(arr=res_flux, Re_a=res_Re, nRe=n)
                            for n in nRe_to_plot])
    f_ax.plot(nRe_to_plot, summed_flux, label='IFU', c='C1')
    f_ax.scatter(x=res_Re.max() + .1,
                 y=summed_flux[-1] * m.Mgy + flux_deficit[3],
                 marker='o', c='C1', edgecolor='k', zorder=3)


def overplot_cmlr_ml_Re(preimaging, drpall_row, drp_logcube, dap_maps, mass_ax, ml_ax, cmlr, 
                        target_snr=10., Re_bds=[0., 0.5, 1.5, 4.],
                        cb1='g', cb2='r', mlb='i'):
    '''
    overplot the CMLR-derived log M/L in radial-azimuthal bins
    '''
    print(drpall_row['plateifu'])
    ebvgal = drpall_row['ebvgal']

    # WCS from preimaging helps us with elliptical coordinates
    preimaging_wcs = wcs.WCS(preimaging['{} img'.format(mlb)].header)
    preimaging_II, preimaging_JJ = np.meshgrid(
        *[np.linspace(0., s - 1., s)
          for s in preimaging.img(mlb).shape], indexing='ij')
    preimaging_AA, preimaging_DD = preimaging_wcs.wcs_pix2world(
        preimaging_JJ, preimaging_II, 1)

    # map sky coordinates to number of effective radii from center
    ell = RotatedParaboloid(
        ctr=np.array([drpall_row['objra'], drpall_row['objdec']]),
        phi=drpall_row['nsa_elpetro_phi'] * u.deg, axis_ratio=drpall_row['nsa_elpetro_ba'],
        Re=drpall_row['nsa_elpetro_th50_r'] / 3600.)

    preimaging_Re = ell(
        np.column_stack([preimaging_AA.flatten(), preimaging_DD.flatten()]), 'r').reshape(
        preimaging_AA.shape)
    preimaging_phi = ell(
        np.column_stack([preimaging_AA.flatten(), preimaging_DD.flatten()]), 'phi').reshape(
        preimaging_AA.shape)

    flux_agg_cb1 = [None, ] * (len(Re_bds) - 1)
    flux_agg_cb2 = [None, ] * (len(Re_bds) - 1)
    flux_agg_mlb = [None, ] * (len(Re_bds) - 1)
    rad_ = [None, ] * (len(Re_bds) - 1)

    kcorr_cb1_interp, kcorr_cb2_interp, kcorr_mlb_interp = kcorr_spec_to_phot(
        drp_logcube, dap_maps['SPX_MFLUX_IVAR'], drpall_row['nsa_z'], cb1, cb2, mlb)

    kcorr_cb1_phot = kcorr_cb1_interp(
        preimaging_AA.flatten(), preimaging_DD.flatten()).reshape(preimaging_AA.shape) * u.mag
    kcorr_cb2_phot = kcorr_cb2_interp(
        preimaging_AA.flatten(), preimaging_DD.flatten()).reshape(preimaging_AA.shape) * u.mag
    kcorr_color = kcorr_cb1_phot - kcorr_cb2_phot
    kcorr_mlb_phot = kcorr_mlb_interp(
        preimaging_AA.flatten(), preimaging_DD.flatten()).reshape(preimaging_AA.shape) * u.mag

    # loop through radial bins, and assign pixels to azimuthal bins
    for ri, (Re0, Re1) in enumerate(zip(Re_bds[:-1], Re_bds[1:])):
        in_radialbin = (preimaging_Re >= Re0) * (preimaging_Re < Re1)

        kc_radbin_mlb = kcorr_mlb_phot[in_radialbin]
        kc_radbin_cb1 = kcorr_cb1_phot[in_radialbin]
        kc_radbin_cb2 = kcorr_cb2_phot[in_radialbin]

        f_cb1 = preimaging.img(cb1, dustcorr=True, ebvgal=ebvgal)[in_radialbin] * (1.0e-9 * m.Mgy)
        f_cb2 = preimaging.img(cb2, dustcorr=True, ebvgal=ebvgal)[in_radialbin] * (1.0e-9 * m.Mgy)
        f_mlb = preimaging.img(mlb, dustcorr=True, ebvgal=ebvgal)[in_radialbin] * (1.0e-9 * m.Mgy)

        ivar_cb1 = preimaging.img_ivar(cb1, dustcorr=True, ebvgal=ebvgal)[in_radialbin] * (1.0e-9 * m.Mgy)**-2
        ivar_cb2 = preimaging.img_ivar(cb2, dustcorr=True, ebvgal=ebvgal)[in_radialbin] * (1.0e-9 * m.Mgy)**-2
        ivar_mlb = preimaging.img_ivar(mlb, dustcorr=True, ebvgal=ebvgal)[in_radialbin] * (1.0e-9 * m.Mgy)**-2

        # assuming SNR per pixel is closely dispersed around median of radialbin in mlb
        snr_fid = np.median(f_mlb * np.sqrt(ivar_mlb)).value

        # number of bins
        n_pix_per_bin = (target_snr / snr_fid)**2
        n_azi_bins = np.rint(in_radialbin.sum() / n_pix_per_bin).astype(int)
        if n_azi_bins < 1:
            n_azi_bins = 1
        print('\t[{}, {}): med. pix. SNR = {} --> nbins = {}'.format(Re0, Re1, snr_fid, n_azi_bins))
        # assume bins have equal azimuthal spacing
        azi_bin_bounds = np.linspace(-np.pi, np.pi, n_azi_bins + 1)
        azi_bin_assign = np.digitize(preimaging_phi[in_radialbin], azi_bin_bounds)
        
        # put flux in bins
        azi_bin_f_cb1 = flux_agg_cb1[ri] = np.bincount(
            azi_bin_assign, weights=(f_cb1 * 2.5**-kc_radbin_cb1.value))[1:]
        azi_bin_f_cb2 = flux_agg_cb2[ri] = np.bincount(
            azi_bin_assign, weights=(f_cb2 * 2.5**-kc_radbin_cb2.value))[1:]
        azi_bin_f_mlb = flux_agg_mlb[ri] = np.bincount(
            azi_bin_assign, weights=(f_mlb * 2.5**-kc_radbin_mlb.value))[1:]
        #print(azi_bin_f_mlb)
        rad_[ri] = np.bincount(azi_bin_assign, weights=preimaging_Re[in_radialbin])[1:] / \
                   np.bincount(azi_bin_assign)[1:]
        #print(azi_bin_f_cb1)
        #print(azi_bin_f_cb2)
        #print(azi_bin_f_mlb)
        
    rad = np.concatenate(rad_)
    agg_color = np.concatenate(
        [-2.5 * np.log10(fcb1 / fcb2) for fcb1, fcb2 in zip(flux_agg_cb1, flux_agg_cb2)])
    agg_mlbf = np.concatenate(flux_agg_mlb) * m.Mgy
    logml = cmlr(agg_color)
    #print(logml)
    
    # plot mass-to-light
    ml_ax.scatter(rad, logml, c='C0', marker='D', s=5., edgecolors='k', linewidths=0.25)

    ml = 10.**logml * m.m_to_l_unit
    agg_mlb_MAG = agg_mlbf.to(u.ABmag) - cosmo.distmod(drpall_row['nsa_zdist'])
    agg_mlb_sollum = 10.**(-0.4 * (agg_mlb_MAG - spectrophot.absmag_sun_band[mlb] * u.ABmag)).value * \
                     m.bandpass_sol_l_unit
    agg_mass = (ml * agg_mlb_sollum).to(u.Msun)

    # plot cumulative mass
    rad_order = np.argsort(rad)
    mass_ax.scatter(
        rad[rad_order], np.cumsum(agg_mass[rad_order]),
        c='C0', marker='D', s=5., edgecolors='k', linewidths=0.25, zorder=3)


def make_photo_spectro_compare(plateifu, pca_system):
    drpall_row = drpall.loc[plateifu]
    plate, ifu = plateifu.split('-')
    Cgr_i_cmlr = lambda color: -0.496 + 1.147 * color
    
    try:
        drp_logcube = m.load_drp_logcube(plate, ifu, mpl_v)
        dap_maps = m.load_dap_maps(plate, ifu, mpl_v, 'SPX-GAU-MILESHC')
        preimaging = MaNGA_PreImage.from_designid_mangaid(
            drpall_row['designid'], drpall_row['mangaid'])
        res = read_results.PCAOutput.from_plateifu(
            os.path.join(basedir, 'results/'), plate, ifu)
    except IOError:
        return False
    else:
        fig, ax1, ax2, ax3 = make_stellarmass_aperture_plot( 
            preimaging=preimaging, dap_maps=dap_maps, drp_logcube=drp_logcube,
            cmlr=Cgr_i_cmlr, drpall_row=drpall_row)
        overplot_spectroscopic(
            res, drp_logcube, drpall.loc[plateifu], ax1, ax2, ax3,
            pca_system, res_wcs=wcs.WCS(dap_maps['SPX_MFLUX'].header), cmlr=Cgr_i_cmlr)
        ax1.axhline(
            (drpall_row['nsa_elpetro_mass'] * u.Msun * u.littleh**-2).to(
                 u.Msun, u.with_H0(cosmo.H0)).value,
            c='k', linestyle='--')
        overplot_cmlr_ml_Re(
            preimaging, drpall_row, drp_logcube, dap_maps, ax1, ax2, Cgr_i_cmlr,
            target_snr=20., 
            Re_bds=[0., .1, .2, .3, .4, .5, .675, .75, .875, 1., 1.25, 1.5, 2., 3.5, 6.])
        ax3.legend(loc='best', prop={'size': 'small'})
        fig.tight_layout(rect=(0., 0., 1., 0.95))
        
        drp_logcube.close()
        dap_maps.close()
        preimaging.close()
        res.close()
        if ax1.get_xlim()[-1] >= 8.:
            ax1.set_xlim([-.5, 8.])
        fig.savefig( 
            os.path.join(basedir, 'results/', plateifu,  
                         '{}_photspec_compare.png'.format(plateifu)), 
            dpi=300)
    finally:
        plt.close('all')

if __name__ == '__main__':
    plateifu = '8335-9102'
    drpall = m.load_drpall(mpl_v, index='plateifu')
    pca_system = read_results.PCASystem.fromfile(os.path.join(basedir, 'pc_vecs.fits'))
    #'''
    results_fnames = glob.glob(os.path.join(basedir, 'results/*-*/*-*_res.fits'))
    results_plateifus = [fn.split('/')[-2] for fn in results_fnames]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for plateifu in results_plateifus:
            try:
                make_photo_spectro_compare(plateifu, pca_system)
            except:
                pass
    #'''
    #make_photo_spectro_compare(plateifu, pca_system)
