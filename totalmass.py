import numpy as np
from scipy.special import expit
from scipy.optimize import curve_fit

from warnings import warn, filterwarnings, catch_warnings, simplefilter
from functools import lru_cache

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
from astropy.utils.decorators import lazyproperty
from astropy.utils.console import ProgressBar

import os
import sys
from glob import glob
from functools import partial

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

sdss_bands = 'ugriz'
nsa_bands = 'FNugriz'

pca_system = read_results.PCASystem.fromfile(os.path.join(csp_basedir, 'pc_vecs.fits'))

class Sigmoid(object):
    p0 = [70., .1, -5., 20.]

    def __init__(self, vscale, hscale, xoffset, yoffset):
        self.vscale, self.hscale = vscale, hscale
        self.xoffset, self.yoffset = xoffset, yoffset

    @staticmethod
    def sigmoid(x, vscale, hscale, xoffset, yoffset):
        return vscale * expit(x / hscale + xoffset) + yoffset

    @classmethod
    def from_points(cls, x, y):
        params, *_ = curve_fit(f=Sigmoid.sigmoid, xdata=x, ydata=y, 
                                p0=Sigmoid.p0)
        return cls(*params)

    def __call__(self, x):
        return self.sigmoid(x, self.vscale, self.hscale, self.xoffset, self.yoffset)

    def __repr__(self):
        return 'Sigmoid function: \
                <vscale = {:.02e}, hscale = {:.02e}, xoffset = {:.02e}, yoffset = {:.02e}>'.format(
                    self.vscale, self.hscale, self.xoffset, self.yoffset)

ba_to_majaxis_angle = Sigmoid.from_points(
    [-1000., 0.2, 0.45, 0.65, 0.9, 1000.], [20., 30., 50., 70., 90., 100.])

def infer_masked(q_trn, bad_trn, infer_here):
    '''
    infer the masked values in the interior of an IFU (foreground stars, dropped fibers)
    '''

    q_final = 1. * q_trn

    # coordinate arrays
    II, JJ = np.meshgrid(*list(map(np.arange, q_trn.shape)), indexing='ij')

    # set up KNN regressor
    knn = knn_regr(trn_coords=[II, JJ], trn_vals=q_trn, good_trn=~bad_trn)
    q_final[infer_here] = infer_from_knn(knn=knn, coords=(II, JJ))[infer_here]

    return q_final

def ifu_bandmag(s2p, b, low_or_no_cov, drp3dmask_interior):
    '''
    flux in bandpass, inferring at masked spaxels
    '''
    mag_band = s2p.ABmags['sdss2010-{}'.format(b)]
    nolight = ~np.isfinite(mag_band)
    # "interpolate" over spaxels with foreground stars or other non-inference-related masks
    mag_band = infer_masked(
        q_trn=mag_band, bad_trn=np.logical_or.reduce((low_or_no_cov, drp3dmask_interior, nolight)),
        infer_here=drp3dmask_interior)
    mag_band[nolight] = np.inf
    return mag_band

def bandpass_flux_to_solarunits(sun_flux_band):
    '''
    equivalency for bandpass fluxes and solar units
    '''
    sun_flux_band_Mgy = sun_flux_band.to(m.Mgy).value
    def convert_flux_to_solar(f):
        s = f / sun_flux_band_Mgy
        return s

    def convert_solar_to_flux(s):
        f = s * sun_flux_band_Mgy
        return f

    return [(m.Mgy, m.bandpass_sol_l_unit, convert_flux_to_solar, convert_solar_to_flux)]

cmlr_kwargs = {
    'cb1': 'g', 'cb2': 'r',
    'cmlr_poly': np.array([ 1.15614812, -0.48479653])}

def cmlr_equivalency(slope, intercept):
    '''
    '''
    def color_to_logml(c):
        return slope * c + intercept

    def logml_to_color(logml):
        return (logml - intercept) / slope

    return [(u.mag, u.dex(m.m_to_l_unit), color_to_logml, logml_to_color)]

class StellarMass(object):
    '''
    calculating galaxy stellar mass with incomplete measurements
    '''
    bands = 'griz'
    bands_ixs = dict(zip(bands, range(len(bands))))
    absmag_sun = np.array([spectrophot.absmag_sun_band[b] for b in bands]) * u.ABmag

    def __init__(self, results, pca_system, drp, dap, drpall_row, cosmo, mlband='i'):
        self.results = results
        self.pca_system = pca_system
        self.drp = drp
        self.dap = dap
        self.drpall_row = drpall_row
        self.cosmo = cosmo
        self.mlband = mlband

        with catch_warnings():
            simplefilter('ignore')
            self.results.setup_photometry(pca_system)

        self.s2p = results.spec2phot

        # stellar mass to light ratio
        self.ml0 = results.cubechannel('ML{}'.format(mlband), 0)
        self.badpdf = results.cubechannel('GOODFRAC', 2) < 1.0e-4
        self.ml_mask = np.logical_or.reduce((self.results.mask, self.badpdf))

        # infer values in interior spaxels affected by one of the following:
        # bad PDF, foreground star, dead fiber
        self.low_or_no_cov = m.mask_from_maskbits(
            self.drp['MASK'].data, [0, 1]).mean(axis=0) > .3
        self.drp3dmask_interior = m.mask_from_maskbits(
            self.drp['MASK'].data, [2, 3]).mean(axis=0) > .3
        self.interior_mask = np.logical_or.reduce((self.badpdf, self.drp3dmask_interior))

        self.logml_final = infer_masked(
            q_trn=self.ml0, bad_trn=self.ml_mask,
            infer_here=self.interior_mask) * u.dex(m.m_to_l_unit)

    @classmethod
    def from_plateifu(cls, plateifu, res_basedir, pca_system, cosmo=cosmo, mlband='i'):
        plate, ifu = plateifu.split('-')
        results = read_results.PCAOutput.from_plateifu(basedir=res_basedir, plate=plate, ifu=ifu)
        drp = m.load_drp_logcube(plate, ifu, mpl_v)
        dap = m.load_dap_maps(plate, ifu, mpl_v, daptype)
        drpall = m.load_drpall(mpl_v, 'plateifu')
        drpall_row = drpall.loc['{}-{}'.format(plate, ifu)]

        return cls(results, pca_system, drp, dap, drpall_row, cosmo, mlband)

    @lazyproperty
    def distmod(self):
        return self.cosmo.distmod(self.drpall_row['nsa_zdist'])

    @lazyproperty
    def nsa_absmags(self):
        absmags = np.array([nsa_absmag(self.drpall_row, band, kind='elpetro')
                            for band in self.bands]) * (u.ABmag - u.MagUnit(u.littleh**2))
        return absmags

    @lazyproperty
    def nsa_absmags_cosmocorr(self):
        return self.nsa_absmags.to(u.ABmag, u.with_H0(self.cosmo.H0))

    @lazyproperty
    def mag_bands(self):
        return np.array([ifu_bandmag(self.s2p, b, self.low_or_no_cov, self.drp3dmask_interior)
                         for b in self.bands]) * u.ABmag

    @lazyproperty
    def flux_bands(self):
        return self.mag_bands.to(m.Mgy)

    @lazyproperty
    def absmag_bands(self):
        return self.mag_bands - self.distmod

    @lazyproperty
    def ifu_flux_bands(self):
        return self.flux_bands.sum(axis=(1, 2))

    @lazyproperty
    def ifu_mag_bands(self):
        return self.ifu_flux_bands.to(u.ABmag)

    @lazyproperty
    def sollum_bands(self):
        with catch_warnings():
            simplefilter('ignore', category=RuntimeWarning)
            sollum = self.absmag_bands.to(
                u.dex(m.bandpass_sol_l_unit),
                bandpass_flux_to_solarunits(self.absmag_sun[..., None, None]))
        return sollum

    @lazyproperty
    def logml_fnuwt(self):
        mask = np.logical_or(self.ml_mask, self.badpdf)
        return np.average(
            self.logml_final.value,
            weights=(self.flux_bands[self.bands_ixs[self.mlband]].value * ~mask)) * \
                     self.logml_final.unit

    @lazyproperty
    def mstar(self):
        return (self.sollum_bands + self.logml_final).to(u.Msun)

    @lazyproperty
    def mstar_in_ifu(self):
        return self.mstar.sum(axis=(1, 2))

    def to_table(self):
        '''
        make table of stellar-mass results
        '''

        tab = t.QTable()
        tab['plateifu'] = [self.drpall_row['plateifu']]

        # tabulate mass in IFU
        tab['mass_in_ifu'] = self.mstar_in_ifu[None, ...][:, self.bands_ixs[self.mlband]]
        nsa_absmag = self.nsa_absmags_cosmocorr
        #tab['nsa_absmag'].meta['bands'] = self.bands
        ifu_absmag = (self.ifu_flux_bands.to(u.ABmag) - self.distmod)
        #tab['ifu_absmag'].meta['bands'] = self.bands
        missing_flux =  (
            (nsa_absmag + self.distmod).to(m.Mgy) - 
            (ifu_absmag + self.distmod).to(m.Mgy)).clip(a_min=0. * m.Mgy, a_max=np.inf * m.Mgy)

        for i, b in enumerate(self.bands):
            outer_flux = missing_flux[i]
            if outer_flux <= 0. * m.Mgy:
                tab['outer_absmag_{}'.format(b)] = np.inf * u.ABmag
                tab['outer_lum_{}'.format(b)] = -np.inf * u.dex(m.bandpass_sol_l_unit)
            else:
                tab['outer_absmag_{}'.format(b)] = outer_flux.to(u.ABmag) - self.distmod
                tab['outer_lum_{}'.format(b)] = tab['outer_absmag_{}'.format(b)].to(
                    u.dex(m.bandpass_sol_l_unit),
                    bandpass_flux_to_solarunits(StellarMass.absmag_sun[i]))

        tab['outer_ml_ring'] = self.ml_ring()
        #tab['outer_ml_ring'].meta['band'] = self.mlband

        tab['ml_fluxwt'] = self.logml_fnuwt
        #tab['ml_fluxwt'].meta['band'] = self.mlband

        tab['distmod'] = self.distmod[None, ...]

        return tab

    def ml_ring(self, azi_selection={'ba_th': .35, 'azi_th': 30. * u.deg}):
        '''
        "ring" aperture-correction
        '''
        phi = self.dap['SPX_ELLCOO'].data[2, ...] * u.deg
        angle_from_majoraxis = np.minimum.reduce(
            (np.abs(phi), np.abs(180. * u.deg - phi), np.abs(360. * u.deg - phi))) * phi.unit

        # how close to major axis must a spaxel be in order to consider it?
        # case: galaxies below b/a threshold  ==> azimuthal angle threshold
        if type(azi_selection) is dict:
            if self.drpall_row['nsa_elpetro_ba'] > azi_selection['ba_th']:
                close_to_majaxis = np.ones_like(angle_from_majoraxis.value).astype(bool)
            else:
                close_to_majaxis = (angle_from_majoraxis < azi_selection['azi_th'])
        # case: b/a determines azimuthal angle threshold based on sigmoid function
        elif isinstance(azi_selection, Sigmoid):
            close_to_majaxis = (angle_from_majoraxis <= ba_to_majaxis_angle(
                self.drpall_row['nsa_elpetro_ba']))

        reff = np.ma.array(
            self.dap['SPX_ELLCOO'].data[1], mask=np.logical_or(self.ml_mask, ~close_to_majaxis))
        outer_ring = np.logical_and.reduce((
            (reff <= reff.max()), (reff >= reff.max() - .5)))
        outer_logml_ring = np.median(self.ml0[~self.ml_mask * outer_ring]) * self.logml_final.unit

        return outer_logml_ring

    def close(self):
        self.results.close()
        self.drp.close()
        self.dap.close()

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

def mass_agg_onegal(res_fname, mlband, drpall):
    res = read_results.PCAOutput.from_fname(res_fname)
    plateifu = res[0].header['PLATEIFU']
    plate, ifu = plateifu.split('-')
    drp = res.get_drp_logcube(mpl_v)
    dap = res.get_dap_maps(mpl_v, daptype)

    stellarmass = StellarMass(
        res, pca_system, drp, dap, drpall.loc[plateifu],
        WMAP9, mlband=mlband)

    with catch_warnings():
        simplefilter('ignore')
        mass_table_new_entry = stellarmass.to_table()

    drp.close()
    dap.close()
    res.close()

    return mass_table_new_entry


def update_mass_table(res_fnames, mlband='i'):
    '''
    '''

    # filter out whose that have not been done
    if mass_table_old is None:
        already_aggregated = [False for _ in range(len(res_fnames))]
    else:
        already_aggregated = [os.path.split(fn)[1].split('_')[0] in mass_table_old['plateifu']
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

    cmlr = cmlr_kwargs
    
    cb1, cb2 = cmlr['cb1'], cmlr['cb2']
    color_missing_flux = mass_table['outer_absmag_{}'.format(cb1)] - \
                         mass_table['outer_absmag_{}'.format(cb2)]

    mass_table['outer_ml_cmlr'] = np.polyval(
        cmlr['cmlr_poly'], color_missing_flux.value) * u.dex(m.m_to_l_unit)

    mass_table['outer_mass_ring'] = \
        (mass_table['outer_lum_{}'.format(mlband)] + \
         mass_table['outer_ml_ring']).to(u.Msun)
    mass_table['outer_mass_cmlr'] = \
        (mass_table['outer_lum_{}'.format(mlband)] + \
         mass_table['outer_ml_cmlr']).to(u.Msun)

    return mass_table['plateifu', 'mass_in_ifu', 'outer_mass_cmlr', 'outer_mass_ring']

def chunks(l, nchunks):
    """Yield n number of striped chunks from l."""
    for i in range(0, nchunks):
        yield l[i::nchunks]

if __name__ == '__main__':
    drpall = m.load_drpall(mpl_v, 'plateifu')
    mlband = 'i'

    mass_table_fname = os.path.join(csp_basedir, 'mass_table.fits')

    # what galaxies are available to aggregate?
    res_fnames = glob(os.path.join(csp_basedir, 'results/*-*/*-*_res.fits'))

    mass_table = None
    for i, rfn in enumerate(res_fnames): 
        if mass_table is None:
            mass_table = mass_agg_onegal(rfn, mlband, drpall)
        else:
            mass_table.add_row(mass_agg_onegal(rfn, mlband, drpall)[0])

        print('{:^6} / {:^6} completed'.format(i + 1, len(res_fnames)), end='\r')

    cmlr = cmlr_kwargs
    
    cb1, cb2 = cmlr['cb1'], cmlr['cb2']
    color_missing_flux = mass_table['outer_absmag_{}'.format(cb1)] - \
                         mass_table['outer_absmag_{}'.format(cb2)]
    mass_table['outer_ml_cmlr'] = np.polyval(
        cmlr['cmlr_poly'], color_missing_flux.value) * u.dex(m.m_to_l_unit)
    mass_table['outer_mass_cmlr'] = (
        mass_table['outer_lum_{}'.format(mlband)] + mass_table['outer_ml_cmlr']).to(u.Msun)
    mass_table['outer_mass_ring'] = (
        mass_table['outer_lum_{}'.format(mlband)] + mass_table['outer_ml_ring']).to(u.Msun)

    mass_table['plateifu', 'mass_in_ifu', 'outer_mass_cmlr', 'outer_mass_ring'].write(
            mass_table_fname, format='fits', overwrite=True)
