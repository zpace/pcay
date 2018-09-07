import numpy as np

from warnings import warn, filterwarnings, catch_warnings, simplefilter

# plotting
import matplotlib.pyplot as plt
from matplotlib import cm as mplcm
from matplotlib import gridspec
import matplotlib.ticker as mticker

# astropy ecosystem
from astropy import constants as c, units as u, table as t
from astropy.io import fits
from astropy import wcs
from astropy import coordinates as coord
from astropy.wcs.utils import pixel_to_skycoord

import os
import sys

# sklearn
from sklearn.neighbors import KNeighborsRegressor

# local
from importer import *
import read_results
import spectrophot

# personal
import manga_tools as m
import spec_tools

spec_unit = 1e-17 * u.erg / u.s / u.cm**2. / u.AA
l_unit = u.AA
bandpass_sol_l_unit = u.def_unit(
    s='bandpass_solLum', format={'latex': r'\overbar{\mathcal{L}_{\odot}}'},
    prefixes=False)
m_to_l_unit = 1. * u.Msun / bandpass_sol_l_unit

band_ix = dict(zip('FNugriz', range(len('FNugriz'))))

def estimate_total_stellar_mass(results, pca_system, drp, dap, drpall_row, dapall_row,
                                cosmo, band='i', missing_mass_kwargs={}):
    '''
    find the total stellar mass of a galaxy with incomplete measurements
    '''

    # stellar mass to light ratio
    ml = results.cubechannel('ML{}'.format(band), 0)
    badpdf = results.cubechannel('GOODFRAC', 2) < 1.0e-4
    ml_mask = np.logical_or.reduce((results.mask, badpdf))

    ml_final = 1. * ml

    # coordinate arrays
    II, JJ = np.meshgrid(*list(map(np.arange, ml.shape)), indexing='ij')

    # set up KNN regressor
    knn = knn_regr([II, JJ], ml, ~ml_mask)
    # infer values in interior spaxels affected by one of the following:
    # bad PDF, foreground star, dead fiber
    interior_mask = np.logical_or.reduce((
        badpdf, (m.mask_from_maskbits(drp['MASK'].data, [3]).sum(axis=0) > 0),
        (m.mask_from_maskbits(drp['MASK'].data, [2]).sum(axis=0) > 0)))
    ml_final[interior_mask] = infer_from_knn(knn, (II, JJ))[interior_mask]

    # convert spectroscopy to photometry
    results.setup_photometry(pca_system)
    s2p = results.spec2phot
    mag_band = s2p.ABmags['sdss2010-{}'.format(band)] * u.ABmag
    distmod = cosmo.distmod(drpall_row['nsa_zdist'])
    absmag_sun_band = spectrophot.absmag_sun_band[band] * u.ABmag
    bandpass_solLum = 10.**(
        -0.4 * (mag_band - distmod - absmag_sun_band).value) * \
        m.bandpass_sol_l_unit

    # figure out flux deficit in stated bandpass
    # first figure out cosmological correction for H0
    nsa_h = 1.0
    cosmocorr_mag = -5. * np.log10(nsa_h / cosmo.h) * u.mag
    nsa_elpetro_absmag_k = np.array(
        [nsa_absmag(drpall_row, band, 'elpetro') for band in 'FNugriz']) * u.ABmag + \
        cosmocorr_mag
    # then turn into app mag and flux
    nsa_elpetro_appmag_k = nsa_elpetro_absmag_k + distmod
    nsa_elpetro_fnu_k = nsa_elpetro_appmag_k.to(m.Mgy)
    nsa_elpetro_fnu_k_band = nsa_elpetro_appmag_k.to(m.Mgy)[band_ix[band]]

    # tabulate flux
    obs_bandpass_fnu = mag_band.to(m.Mgy)
    inf_ABmag = ~np.isfinite(mag_band)
    exterior_mask = np.logical_or(ml_mask, inf_ABmag) * (~interior_mask)
    flux_in_ifu = obs_bandpass_fnu[~exterior_mask].sum()
    # find missing flux (if any)
    missing_bandpass_fnu = nsa_elpetro_fnu_k_band - flux_in_ifu

    mass_in_ifu = (10.**ml_final * m.m_to_l_unit * bandpass_solLum)[~exterior_mask].sum()

    # calculate luminosity-weighted mass-to-light
    logml_fnuwt = np.average(
        ml_final[~exterior_mask], weights=obs_bandpass_fnu[~exterior_mask])
    ifu_lum = bandpass_solLum[~exterior_mask].sum()

    # if there's no (or negative) missing flux, just return coadded IFU
    if missing_bandpass_fnu <= 0. * m.Mgy:
        outer_logml_ring = -np.inf
        outer_logml_cmlr = -np.inf
        missing_bandpass_solLum = 0. * m.bandpass_sol_l_unit

    else:
        # missing flux in one band, to mag, to sol units
        missing_bandpass_mag = missing_bandpass_fnu.to(u.ABmag)
        missing_bandpass_solLum = 10.**(
            -0.4 * (missing_bandpass_mag - distmod - absmag_sun_band).value) * \
            m.bandpass_sol_l_unit

        # assume missing mass has M/L equal to average of outermost 0.5 Re
        # now find the median of the outer unmasked .5 Re
        reff = np.ma.array(dap['SPX_ELLCOO'].data[1], mask=ml_mask)
        outer_ring = np.logical_and((reff <= reff.max()), (reff >= reff.max() - .5))
        outer_logml_ring = np.median(ml[~ml_mask * outer_ring])

        # apply cmlr
        outer_logml_cmlr = apply_cmlr(
            ifu_s2p=s2p, f_tot=nsa_elpetro_fnu_k, fluxes_keys='FNugriz',
            mlrb=band, exterior_mask=exterior_mask, **missing_mass_kwargs)

    return (mass_in_ifu, 10.**logml_fnuwt, ifu_lum.value,
            10.**outer_logml_ring, 10.**outer_logml_cmlr, missing_bandpass_solLum.value)


def knn_regr(trn_coords, trn_vals, good_trn, k=8):
    '''
    return a trained estimator for k-nearest-neighbors

    - trn_coords: list containing row-coordinate map and col-coordinate map
    - trn_vals: map of values used for training
    - good_trn: binary map (True signifies good data)
    '''
    II, JJ = trn_coords
    good = good_trn.flatten()
    coo = np.column_stack([II.flatten()[good], JJ.flatten()[good]])
    vals = trn_vals.flatten()[good]

    knn = KNeighborsRegressor(
        n_neighbors=k, weights='uniform', p=2)
    knn.fit(coo, vals)

    return knn

def infer_from_knn(knn, coords):
    '''
    use KNN regressor to infer values over a grid
    '''
    II, JJ = coords
    coo = np.column_stack([II.flatten(), JJ.flatten()])
    vals = knn.predict(coo).reshape(II.shape)

    return vals

def apply_cmlr(ifu_s2p, cb1, cb2, mlrb, cmlr_poly, f_tot, exterior_mask,
               fluxes_keys='FNugriz'):
    '''
    apply a color-mass-to-light relation
    '''
    f_tot_d = dict(zip(fluxes_keys, f_tot))

    # find missing flux in color-band 1
    fb1_ifu = (ifu_s2p.ABmags['sdss2010-{}'.format(cb1)] * u.ABmag)[~exterior_mask].to(m.Mgy).sum()
    dfb1 = f_tot_d[cb1] - fb1_ifu
    # find missing flux in color-band 2
    fb2_ifu = (ifu_s2p.ABmags['sdss2010-{}'.format(cb2)] * u.ABmag)[~exterior_mask].to(m.Mgy).sum()
    dfb2 = f_tot_d[cb2] - fb2_ifu

    # if there's no missing flux in one/both bands, then this method fails
    if (dfb1.value <= 0.) or (dfb2.value <= 0.):
        return -np.inf

    c = dfb1.to(u.ABmag) - dfb2.to(u.ABmag)

    logml = np.polyval(p=cmlr_poly, x=c.value)

    return logml

def nsa_mass(drpall_row, band, kind='elpetro'):
    # kind is elpetro or sersic
    mass = drpall_row['nsa_{}_mass'.format(kind)][band_ix[band]]
    return mass

def nsa_flux(drpall_row, band, kind='elpetro'):
    # kind is petro, elpetro, or sersic
    flux = drpall_row['nsa_{}_flux'.format(kind)][band_ix[band]]
    return flux

def nsa_absmag(drpall_row, band, kind='elpetro'):
    # kind is petro, elpetro, or sersic
    flux = drpall_row['nsa_{}_absmag'.format(kind)][band_ix[band]]
    return flux
