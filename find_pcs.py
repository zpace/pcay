#!/usr/bin/env python3

import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib import cm as mplcm
from matplotlib import gridspec
import matplotlib.ticker as mticker
from cycler import cycler

# astropy ecosystem
from astropy import constants as c, units as u, table as t
from astropy.io import fits
from astropy import wcs
from astropy.utils.console import ProgressBar
from astropy.cosmology import WMAP9
from astropy import coordinates as coord
from astropy.wcs.utils import pixel_to_skycoord

import os
import sys
from warnings import warn, filterwarnings, catch_warnings, simplefilter
from traceback import print_exception
from functools import lru_cache
import pickle

# scipy
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import entropy

# statsmodels
from statsmodels.nonparametric.kde import KDEUnivariate

# local
from importer import *
import csp
import cov_obs
import figures_tools
from spectrophot import (lumspec2lsun, color, C_ML_conv_t as CML,
                         Spec2Phot, absmag_sun_band as Msun)
import utils as ut
from fakedata import FakeData, SkyContamination
from linalg import *
from param_estimate import *
from rectify import MaNGA_deredshift
import pca_status

# personal
import manga_tools as m

eps = np.finfo(float).eps

class StellarPop_PCA(object):

    '''
    class for determining PCs of a library of synthetic spectra
    '''

    def __init__(self, l, trn_spectra, gen_dicts, metadata, K_obs, src,
                 sfh_fnames, nsubpersfh, nsfhperfile, basedir,
                 dlogl=None, lllim=3700. * u.AA, lulim=7400. * u.AA):
        '''
        params:
         - l: length-n array-like defining the wavelength bin centers
            (should be log-spaced)
         - spectra: m-by-n array of spectra (individual spectrum contained
            along one index in dimension 0), in units of 1e-17 erg/s/cm2/AA
         - gen_dicts: length-m list of FSPS_SFHBuilder.FSPS_args dicts,
            ordered the same as `spectra`
         - metadata: table of derived SSP properties used for regression
            (D4000 index, Hd_A index, r-band luminosity-weighted age,
             mass-weighted age, i-band mass-to-light ratio,
             z-band mass-to-light ratio, mass fraction formed in past 1Gyr,
             formation time, eftu, metallicity, tau_V, mu, sigma)
            this somewhat replicates data in `gen_dicts`, but that's ok
        '''
        self.basedir = basedir

        l_good = np.ones_like(l, dtype=bool)
        if lllim is not None:
            l_good *= (l >= lllim)
        if lulim is not None:
            l_good *= (l <= lulim)

        self.l = l[l_good]
        self.logl = np.log10(self.l.to('AA').value)
        if not dlogl:
            dlogl = np.round(np.mean(self.logl[1:] - self.logl[:-1]), 8)

        self.dlogl = dlogl

        self.trn_spectra = trn_spectra[:, l_good]

        self.metadata = metadata

        # metadata array is anything with a 'TeX' metadata entry
        metadata_TeX = [metadata[n].meta.get('TeX', False)
                        for n in metadata.colnames]
        metadata_incl = np.array([True if m is not False else False
                                  for m in metadata_TeX])
        self.metadata_TeX = [m for m, n in zip(metadata_TeX, metadata.colnames)
                             if m is not False]

        # a kludgey conversion from structured array to regular array
        metadata_a = np.array(self.metadata)
        metadata_a = metadata_a.view((metadata_a.dtype[0],
                                      len(metadata_a.dtype.names)))
        self.metadata_a = metadata_a[:, metadata_incl]

        self.src = src

        self.sfh_fnames = sfh_fnames

        self.nsubpersfh = nsubpersfh
        self.nsfhperfile = nsfhperfile

        self.important_params = ['MLi', 'Dn4000', 'Hdelta_A',
                                 'MWA', 'sigma', 'logzsol',
                                 'tau_V', 'mu', 'tau_V mu',
                                 'Mg_b', 'Ca_HK', 'F_1G',
                                 'logQHpersolmass']

        self.importantplus_params = self.important_params + \
                                    ['F_200M', 'tf', 'd1', 'tt', 'MLV']

        self.confident_params = ['MLi', 'logzsol',
                                 'tau_V', 'mu', 'tau_V mu', 'tau_V (1 - mu)',
                                 'logQHpersolmass', 'uv_slope']

        # observational covariance matrix
        if not isinstance(K_obs, cov_obs.Cov_Obs):
            raise TypeError('incorrect observational covariance matrix class!')

        if not np.isclose(K_obs.dlogl, self.dlogl, rtol=1.0e-3):
            raise PCAError('non-matching log-lambda spacing ({}, {})'.format(
                           K_obs.dlogl, self.dlogl))

    @classmethod
    def from_FSPS(cls, K_obs, lsf, base_dir, nfiles=None,
                  log_params=['MWA', 'MLr', 'MLi', 'MLz', 'MLV',
                              'F_20M', 'F_100M', 'F_200M', 'F_500M', 'F_1G'],
                  inf_replace=dict(zip(['F_20M', 'F_100M', 'F_200M', 'F_500M', 'F_1G'],
                                       [-20., -20., -20., -20., -20.])),
                  vel_params={}, dlogl=1.0e-4, z0_=.04,
                  preload_llims=[3000. * u.AA, 10000. * u.AA], **kwargs):
        '''
        Read in FSPS outputs (dicts & metadata + spectra) from some directory
        '''

        from glob import glob
        from utils import pickle_loader, add_losvds
        from itertools import chain

        csp_fnames = glob(os.path.join(base_dir,'CSPs_*.fits'))
        sfh_fnames = glob(os.path.join(base_dir,'SFHs_*.fits'))
        print('Building training library in directory: {}'.format(base_dir))
        print('CSP files used: {}'.format(' '.join(tuple(csp_fnames))))

        if nfiles is not None:
            csp_fnames = csp_fnames[:nfiles]
            sfh_fnames = sfh_fnames[:nfiles]

        l = fits.getdata(csp_fnames[0], 'lam') * u.AA
        logl = np.log10(l.value)

        Nsubsample = fits.getval(sfh_fnames[0], ext=0, keyword='NSUBPER')
        Nsfhper = fits.getval(sfh_fnames[0], ext=0, keyword='NSFHPER')

        meta = t.vstack([t.Table.read(f, format='fits', hdu=1)
                         for f in csp_fnames])
        spec = np.row_stack(list(map(
            lambda fn: fits.getdata(fn, 'flam'), csp_fnames)))

        in_lrange = (l >= preload_llims[0]) * (l <= preload_llims[1])
        spec = spec[:, in_lrange]
        l = l[in_lrange]
        logl = logl[in_lrange]

        meta['tau_V mu'] = meta['tau_V'] * meta['mu']
        meta['tau_V (1 - mu)'] = meta['tau_V'] * (1. - meta['mu'])
        meta['QLi'] = 10.**meta['logQHpersolmass'] * meta['MLi']

        for k in meta.colnames:
            if len(meta[k].shape) > 1:
                del meta[k]

        del meta['mstar']

        meta['MWA'].meta['TeX'] = r'MWA'
        meta['Dn4000'].meta['TeX'] = r'D$_{n}$4000'
        meta['Hdelta_A'].meta['TeX'] = r'H$\delta_A$'
        meta['Hdelta_F'].meta['TeX'] = r'H$\delta_F$'
        meta['Hgamma_A'].meta['TeX'] = r'H$\gamma_A$'
        meta['Hgamma_F'].meta['TeX'] = r'H$\gamma_F$'
        meta['H_beta'].meta['TeX'] = r'H$\beta$'
        meta['Mg_1'].meta['TeX'] = r'Mg$_1$'
        meta['Mg_2'].meta['TeX'] = r'Mg$_2$'
        meta['Mg_b'].meta['TeX'] = r'Mg$_b$'
        meta['Ca_HK'].meta['TeX'] = r'CaHK'
        meta['Na_D'].meta['TeX'] = r'Na$_D$'
        meta['CN_1'].meta['TeX'] = r'CN$_1$'
        meta['CN_2'].meta['TeX'] = r'CN$_2$'
        meta['TiO_1'].meta['TeX'] = r'TiO$_1$'
        meta['TiO_2'].meta['TeX'] = r'TiO$_2$'
        meta['logzsol'].meta['TeX'] = r'$\log{\frac{Z}{Z_{\odot}}}$'
        meta['tau_V'].meta['TeX'] = r'$\tau_V$'
        meta['mu'].meta['TeX'] = r'$\mu$'
        meta['tau_V mu'].meta['TeX'] = r'$\tau_V ~ \mu$'
        meta['tau_V (1 - mu)'].meta['TeX']  = r'$\tau_V ~ (1 - \mu)$'
        meta['MLr'].meta['TeX'] = r'$\log \Upsilon^*_r$'
        meta['MLi'].meta['TeX'] = r'$\log \Upsilon^*_i$'
        meta['MLz'].meta['TeX'] = r'$\log \Upsilon^*_z$'
        meta['MLV'].meta['TeX'] = r'$\log \Upsilon^*_V$'
        meta['sigma'].meta['TeX'] = r'$\sigma$'
        meta['F_20M'].meta['TeX'] = r'$F_m^{\rm .02G}$'
        meta['F_50M'].meta['TeX'] = r'$F_m^{\rm .05G}$'
        meta['F_100M'].meta['TeX'] = r'$F_m^{\rm .1G}$'
        meta['F_200M'].meta['TeX'] = r'$F_m^{\rm .2G}$'
        meta['F_500M'].meta['TeX'] = r'$F_m^{\rm .5G}$'
        meta['F_1G'].meta['TeX'] = r'$F_m^{\rm 1G}$'
        meta['gamma'].meta['TeX'] = r'$\gamma$'
        meta['theta'].meta['TeX'] = r'$\Theta$'
        meta['d1'].meta['TeX'] = r'$\tau_{\rm SFH}$'
        meta['tf'].meta['TeX'] = r'$t_{\rm form}$'
        meta['tt'].meta['TeX'] = r'$t_{\rm trans}$'
        meta['nburst'].meta['TeX'] = r'$N_{\rm burst}$'
        meta['Cgr'].meta['TeX'] = r'$C_{gr}$'
        meta['Cri'].meta['TeX'] = r'$C_{ri}$'
        meta['Cgr_z015'].meta['TeX'] = r'$C^{.15}_{gr}$'
        meta['Cri_z015'].meta['TeX'] = r'$C^{.15}_{ri}$'
        meta['sbss'].meta['TeX'] = r'$S_{\rm BSS}$'
        meta['fbhb'].meta['TeX'] = r'$f_{\rm BHB}$'
        meta['logQHpersolmass'].meta['TeX'] = r'$\log{\frac{Q_H}{M_{\odot}}}$'
        meta['QLi'].meta['TeX'] = r'$\log \frac{Q_H}{\mathcal{L}_i}$'
        meta['uv_slope'].meta['TeX'] = r'$\beta_{UV}$'

        for n in meta.colnames:
            if n in log_params:
                meta[n] = np.log10(meta[n])
                meta[n].meta['scale'] = 'log'
                if n in inf_replace.keys():
                    meta[n][~np.isfinite(meta[n])] = inf_replace[n]
                if 'ML' in n:
                    meta[n].meta['unc_incr'] = .008

        #spec, meta = spec[models_good, :], meta[models_good]

        # convolve spectra with instrument LSF
        dlogl_hires = ut.determine_dlogl(logl)
        spec_lsf = lsf(y=spec, lam=(l.value) * (1. + z0_),
                       dlogl=dlogl_hires, z=z0_)

        # interpolate models to desired l range
        logl_final = np.arange(np.log10(l.value.min()),
                               np.log10(l.value.max()), dlogl)

        l_final = 10.**logl_final
        spec_lores = ut.interp_large(x0=logl, y0=spec_lsf, xnew=logl_final,
                                     axis=-1, kind='linear')

        spec_lores /= spec_lores.max(axis=1)[..., None]

        for k in meta.colnames:
            meta[k] = meta[k].astype(np.float32)

        return cls(l=l_final * l.unit, trn_spectra=spec_lores,
                   gen_dicts=None, metadata=meta, sfh_fnames=sfh_fnames,
                   K_obs=K_obs, dlogl=None, src='FSPS',
                   nsubpersfh=Nsubsample, nsfhperfile=Nsfhper, basedir=base_dir,
                   **kwargs)

    # =====
    # methods
    # =====

    def xval(self, specs, qmax=30):
        # reconstruction error
        qs = np.linspace(1, qmax, qmax, dtype=int)
        err = np.empty_like(qs, dtype=float)

        # normalize mean of each training spectrum to 1
        specs_norm, a = self.scaler(specs, lam_axis=1)
        S = specs_norm - self.M

        def recon_rms(q):
            A = quick_data_to_PC(S, self.evecs_[:q])
            # reconstructed spectra
            S_recon = np.dot(A, self.evecs_[:q])
            resid = S_recon - S

            # fractional reconstruction error
            e = np.sqrt(np.mean(resid**2.))
            return e

        for i, q in enumerate(qs):
            try:
                err[i] = recon_rms(q)
            except:
                err[i] = 1.
                continue

        return qs, err

    def xval_fromfile(self, fname, lsf, z0, qmax=50, target=.01):
        hdulist = fits.open(fname)

        specs_full = hdulist['flam'].data
        l_full = hdulist['lam'].data
        logl_full = np.log10(l_full)

        # convolve spectra with instrument LSF
        dlogl_hires = ut.determine_dlogl(logl_full)
        specs_lsf = lsf(y=specs_full, lam=(l_full) * (1. + z0),
                        dlogl=dlogl_hires, z=z0)

        specs_interp = interp1d(x=logl_full, y=specs_lsf, kind='linear', axis=-1)
        specs = specs_interp(self.logl)

        qs, err = self.xval(specs, qmax)

        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = fig.add_subplot(111)
        ax.plot(qs, err)
        ax.set_yscale('log')

        loc = mticker.MaxNLocator(nbins=5, integer=True, steps=[1, 2, 5, 10])
        ax.xaxis.set_major_locator(loc)

        ax.set_xlabel('Number of PCs')
        ax.set_ylabel('Frac. Recon. Err.')
        fig.tight_layout()

        # return smallest q value for which target FRE is reached
        try:
            q = np.min(qs[err <= target])
        except ValueError:
            q = None

        hdulist.close()

        fig.savefig(os.path.join(self.basedir, 'xval_test.png'))

        return q

    def run_pca_models(self, q):
        '''
        run PCA on library of model spectra
        '''

        self.scaler = ut.MedianSpecScaler(X=self.trn_spectra)
        self.normed_trn = self.scaler.X_sc
        self.M = np.median(self.normed_trn, axis=0)
        self.S = self.normed_trn - self.M

        self.evals_, self.evals, self.evecs_, self.PCs = run_pca(self.S, q)
        self.trn_PC_wts = quick_data_to_PC(self.S, self.PCs)

        # reconstruct the best approximation for the spectra from PCs
        self.trn_recon = np.dot(self.trn_PC_wts, self.PCs)
        # calculate covariance using reconstruction residuals
        self.trn_resid = self.normed_trn - self.trn_recon

        # percent variance explained
        self.PVE = (self.evals_ / self.evals_.sum())[:q]

        self.cov_th = np.cov(self.trn_resid, rowvar=False)

    def project_cube(self, f, ivar, mask_spax=None, mask_spec=None,
                     mask_cube=None, ivar_as_weights=True):
        '''
        project real spectra onto principal components, given a
            flux array & inverse-variance array, plus optional
            additional masks

        params:
         - f: flux array (should be regridded to rest, need not be normalized)
            with shape (nl, m, m) (shape of IFU datacube)
         - ivar: inverse-variance (same shape as f)
         - mask_spax: sets ivar in True spaxels to zero at all wavelengths
            within a given spaxel
         - mask_spec: sets ivar in True elements to zero in all spaxels
         - mask_cube: sets ivar in True elements to zero in only the
            corresponding elements

        Note: all three masks can be used simultaneously, but mask_spax
            is applied first, then mask_spec, then mask_cube
        '''

        assert ivar.shape == f.shape, \
            'cube shapes must be equal, are {}, {}'.format(ivar.shape, f.shape)
        cube_shape = f.shape

        # manage masks
        if mask_spax is not None:
            ivar = ivar * (~mask_spax).astype(float)
        if mask_spec is not None:
            ivar = ivar * (~mask_spec[:, None, None]).astype(float)
        if mask_cube is not None:
            ivar = ivar * (~mask_cube).astype(float)

        # run through same scaling normalization as training data
        O_norm, a = self.scaler(f)
        ivar_sc = ivar * a**2.
        O_sub = O_norm - self.M[:, None, None]

        if ivar_as_weights:
            w = ivar_sc + eps
        else:
            w = None

        A = robust_project_onto_PCs(e=self.PCs, f=O_sub, w=w)

        return A, self.M, a, O_sub, O_norm, ivar_sc

    def write_pcs_fits(self):
        '''
        write training data mean and PCs to fits
        '''

        hdulist = fits.HDUList([fits.PrimaryHDU()])
        lam_hdu = fits.ImageHDU(self.l.value)
        lam_hdu.header['EXTNAME'] = 'LAM'
        lam_hdu.header['BUNIT'] = 'AA'
        hdulist.append(lam_hdu)

        mean_hdu = fits.ImageHDU(self.M)
        mean_hdu.header['EXTNAME'] = 'MEAN'
        hdulist.append(mean_hdu)

        pc_hdu = fits.ImageHDU(self.PCs)
        pc_hdu.header['EXTNAME'] = 'EVECS'
        hdulist.append(pc_hdu)

        hdulist.writeto(os.path.join(self.basedir, 'pc_vecs.fits'), overwrite=True)

    def reconstruct_normed(self, A):
        '''
        reconstruct spectra to (one-normalized) cube

        params:
         - A: array of weights per spaxel
        '''

        R = np.einsum('nij,nl->lij', A, self.PCs) + self.M[:, None, None]

        return R

    def reconstruct_full(self, A, a):
        '''
        reconstruct spectra to properly-scaled cube

        params:
         - A: array of weights per spaxel
         - a: "surface-brightness" multiplier, used to scale the cube
        '''

        # R = a * (S + M)
        # S = A dot E

        R = a[None, ...] * (np.einsum('nij,nl->lij', A, self.PCs) +
                            self.M[:, None, None])

        return R

    def _compute_i0_map(self, cov_logl, z_map):
        '''
        compute the index of some array corresponding to the given
            wavelength at some redshift

        params:
         - tem_logl0: the smallest wavelength of the fixed-grid template
            that will be the destination of the bin-shift
         - logl: the wavelength grid that will be transformed
         - z_map: the 2D array of redshifts used to figure out the offset
        '''

        l0_map = 10.**self.logl[0] * np.ones(z_map.shape)[None, ...]

        rules = [dict(name='l', exponent=+1, array_in=l0_map)]
        ll0z_map = np.log10(
            ut.slrs(rules=rules, z_in=0., z_out=z_map)['l'])

        # find the index for the wavelength that best corresponds to
        # an appropriately redshifted wavelength grid
        ll_d = ll0z_map - np.tile(cov_logl[..., None, None],
                                  (1, ) + z_map.shape)

        i0_map = np.argmin(np.abs(ll_d), axis=0)

        return i0_map

    def compute_model_weights(self, P, A):
        '''
        compute model weights for each combination of spaxel (PC fits)
            and model

        params:
         - P: inverse of PC covariance matrix, shape (q, q)
         - A: PC weights OF OBSERVED DATA obtained from weighted PC
            projection routine (robust_project_onto_PCs),
            shape (q, NX, NY)

        NOTE: this is the equivalent of taking model weights a = A[n, x, y]
            in some spaxel (x, y), and the corresp. inv-cov matrix
            p = P[..., x, y], training data PC weights C; constructing
            D = C - a; and taking D \dot p \dot D
        '''

        C = self.trn_PC_wts
        # C shape: [MODELNUM, PCNUM]
        # A shape: [PCNUM, XNUM, YNUM]
        D = C[..., None, None] - A[None, ...]
        # D shape: [MODELNUM, PCNUM, XNUM, YNUM]

        dist2 = np.einsum('cixy,ijxy,cjxy->cxy', D, P, D)
        det_K = 1. / np.linalg.det(np.moveaxis(P, [0, 1, 2, 3], [2, 3, 0, 1]))
        c = 0.5 * (np.log(det_K) + self.PCs.shape[0] * np.log(2. * np.pi))
        w = np.exp(-0.5 * dist2 - c)

        return w

    def param_pct_map(self, qty, W, P, mask, order=None, factor=None, add=None):
        '''
        This is no longer iteration based, which is awesome.

        params:
         - qty: string, specifying which quantity you want (qty must be
            an element of self.metadata.colnames)
         - W: cube of shape (nmodels, NX, NY), with weights for each
            combination of spaxel and model
         - P: percentile(s)
         - factor: array to multiply metadata[qty] by. This basically
            lets you get M by multiplying M/L by L
         - add: array to add to metadata[qty]. Equivalent to factor for
             log-space data
        '''

        cubeshape = W.shape[-2:]
        Q = self.metadata[qty][np.isfinite(self.metadata[qty])]
        W = W[np.isfinite(self.metadata[qty])]

        if factor is None:
            factor = np.ones(cubeshape)

        if add is None:
            add = np.zeros(cubeshape)

        A = param_interp_map(v=Q, w=W, pctl=np.array(P), mask=mask, order=order)

        return (A + add[None, ...]) * factor[None, ...]

    def param_cred_intvl(self, qty, W, mask, order=None, factor=None):
        '''
        find the median and Bayesian credible interval size (two-sided)
            of some param's PDF
        '''

        P = [16., 50., 84.]

        # get scale for qty, default to linear
        scale = self.metadata[qty].meta.get('scale', 'linear')

        if scale == 'log':
            # it's CRITICAL that factor is in compatible units to qty
            if factor is not None:
                add, factor = np.log10(factor), None
            else:
                add, factor = None, None
        else:
            add = None

        # get uncertainty increase
        unc_incr = self.metadata[qty].meta.get('unc_incr', 0.)

        # get param pctl maps
        P = self.param_pct_map(qty=qty, W=W, P=P, mask=mask, order=order,
                               factor=factor, add=add)

        P16, P50, P84 = tuple(map(np.squeeze, np.split(P, 3, axis=0)))
        if scale == 'log':
            l_unc, u_unc = (np.abs(P50 - P16) + unc_incr,
                            np.abs(P84 - P50) + unc_incr)
        else:
            l_unc, u_unc = (np.abs(P50 - P16) + unc_incr,
                            np.abs(P84 - P50) + unc_incr)

        return P50, l_unc, u_unc, scale

    def make_PCs_fig(self):
        '''
        plot eigenspectra
        '''

        q = self.PCs.shape[0]
        wdim, hdim = (6, 0.8 + 0.5 * (q + 1.))
        fig = plt.figure(figsize=(wdim, hdim), dpi=300)
        gs = gridspec.GridSpec((q + 1), 1)
        hborder = (0.55 / hdim, 0.35 / hdim) #  height border
        wborder = (0.55 / wdim, 0.25 / hdim) #  width border
        hspace = (hdim - 1.) / 20.

        gs.update(left=wborder[0], right=1. - wborder[1], wspace=0.,
                  bottom=hborder[0], top=1. - hborder[1], hspace=hspace)

        PCs = np.row_stack([self.M, self.PCs])

        for i in range(q + 1):
            ax = plt.subplot(gs[i])
            ax.plot(self.l, PCs[i, :], color='k', linestyle='-',
                    drawstyle='steps-mid', linewidth=0.5)
            if i == 0:
                pcnum = 'Median'
            else:
                pcnum = 'PC{}'.format(i)
            ax.set_ylabel(pcnum, size=6)

            loc = mticker.MaxNLocator(nbins=5, prune='upper')
            ax.yaxis.set_major_locator(loc)

            if i != q:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.tick_params(axis='x', color='k', labelsize=8)

            ax.tick_params(axis='y', color='k', labelsize=6)

        # use last axis to give wavelength
        ax.set_xlabel(r'$\lambda~[\textrm{\AA}]$')
        plt.suptitle('Eigenspectra')

        fig.savefig(os.path.join(self.basedir, 'PCs_{}.png'.format(self.src)),
                    dpi=300)

    def make_params_vs_PCs_fig(self):
        '''
        make a triangle-plot-like figure with PC amplitudes plotted against components
        '''

        from astropy.visualization import hist as ahist
        from itertools import product as iproduct

        q = ncols = self.PCs.shape[0]
        nparams = nrows = self.metadata_a.shape[1]

        # dimensions of component subplots
        sc_ht, sc_wid = 1., 1.
        pch_ht, pch_wid = .6, 1.
        pah_ht, pah_wid = 1., .6
        lbord, rbord, ubord, dbord = 0.8, 0.4, 0.6, 0.6
        wspace, hspace = 0.5, 0.5

        wdim = lbord + rbord + pah_wid + ncols * sc_wid
        hdim = ubord + dbord + pch_ht + nrows * sc_ht

        wrs = [1 for _ in range(ncols)]
        hrs = [1 for _ in range(nrows)]
        wrs.append(pch_wid / sc_wid)
        hrs.append(pah_ht / sc_ht)

        fig = plt.figure(figsize=(wdim, hdim), dpi=300)

        gs = gridspec.GridSpec(ncols=(ncols + 1), nrows=(nrows + 1),
                               left=(lbord / wdim), right=(1. - rbord / wdim),
                               bottom=(dbord / hdim), top=(1. - ubord / hdim),
                               wspace=(wspace / wdim), hspace=(hspace / hdim),
                               width_ratios=wrs, height_ratios=hrs)

        # lists of hist axes, to allow sharex and sharey
        PC_hist_axes = [None for _ in range(q)]
        param_hist_axes = [None for _ in range(nparams)]

        # PC histograms in top row
        for i in range(q):
            ax = fig.add_subplot(gs[0, i])
            try:
                ahist(self.trn_PC_wts[:, i], bins='knuth', ax=ax,
                      histtype='step', orientation='vertical',
                      linewidth=0.5)
            # handle when there are tons and tons of models
            except MemoryError:
                ahist(self.trn_PC_wts[:, i], bins=50, ax=ax,
                      histtype='step', orientation='vertical',
                      linewidth=0.5)
            except ValueError:
                pass
            ax.tick_params(axis='x', labelbottom=False)
            ax.tick_params(axis='y', labelleft=False)
            PC_hist_axes[i] = ax

        # param histograms in right column
        for i in range(nrows):
            ax = fig.add_subplot(gs[i + 1, -1])
            try:
                ahist(self.metadata_a[:, i], bins='knuth', ax=ax,
                      histtype='step', orientation='horizontal',
                      linewidth=0.5)
            # handle when there are tons and tons of models
            except MemoryError:
                ahist(self.metadata_a[:, i], bins=50, ax=ax,
                      histtype='step', orientation='horizontal',
                      linewidth=0.5)
            except ValueError:
                pass
            ax.tick_params(axis='x', labelbottom=False)
            yloc = mticker.MaxNLocator(nbins=5, prune='upper')
            # tick labels on RHS of hists
            ax.yaxis.set_major_locator(yloc)
            ax.tick_params(axis='y', labelleft=False, labelright=True,
                           labelsize=6)
            param_hist_axes[i] = ax

        # scatter plots everywhere else
        for i, j in iproduct(range(nrows), range(ncols)):
            # i is param number
            # j is PC number

            ax = fig.add_subplot(gs[i + 1, j], sharex=PC_hist_axes[j],
                                 sharey=param_hist_axes[i])
            ax.scatter(self.trn_PC_wts[:, j], self.metadata_a[:, i],
                       facecolor='k', edgecolor='None', marker='.',
                       s=1., alpha=0.4)

            # suppress x axis and y axis tick labels
            # (except in bottom row and left column, respectively)

            if i != nparams - 1:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                xloc = mticker.MaxNLocator(nbins=5, prune='upper')
                ax.xaxis.set_major_locator(xloc)
                ax.tick_params(axis='x', labelsize=6)
                ax.set_xlabel('PC{}'.format(j + 1), size=8)

            if j != 0:
                ax.tick_params(axis='y', labelleft=False)
            else:
                yloc = mticker.MaxNLocator(nbins=5, prune='upper')
                ax.yaxis.set_major_locator(yloc)
                ax.tick_params(axis='y', labelsize=6)
                ax.set_ylabel(self.metadata_TeX[i], size=8)

        fig.suptitle('PCs vs params')

        plt.savefig(os.path.join(self.basedir, 'PCs_params_{}.png'.format(self.src)),
                    dpi=300)

    def find_PC_param_coeffs(self):
        '''
        find the combination of PC amplitudes that predict the parameters

        a X + Z = b
        '''

        # dependent variable (the parameter values)
        b_ = self.metadata_a

        # independent variable (the PC weights)
        a_ = np.column_stack(
            [self.trn_PC_wts,
             np.ones(self.trn_PC_wts.shape[0])])

        X = np.stack([np.linalg.lstsq(a=a_, b=b_[:, i], rcond=None)[0]
                      for i in range(b_.shape[-1])])

        # X has shape (nparams, q)
        return X

    def make_PC_param_regr_fig(self):
        '''
        make a figure that compares each parameter against the PC
            combination that most closely predicts it
        '''

        # how many params are there?
        # try to make a square grid, but if impossible, add another row
        nparams = self.metadata_a.shape[1]
        gs, fig = figures_tools.gen_gridspec_fig(N=nparams)

        # regression result
        A = self.find_PC_param_coeffs()

        for i in range(nparams):
            # set up subplots
            ax = fig.add_subplot(gs[i])

            x = np.column_stack([self.trn_PC_wts,
                                 np.ones(self.trn_PC_wts.shape[0])])
            y = self.metadata_a[:, i]
            y_regr = A[i].dot(x.T).flatten()
            ax.scatter(y_regr, y, marker='.', facecolor='b', edgecolor='None',
                       s=1., alpha=0.4)
            xgrid = np.linspace(y.min(), y.max())
            ax.plot(xgrid, xgrid, linestyle='--', c='g', linewidth=1)

            ax_ = ax.twinx()
            ax_.set_ylim([0., 1.])
            ax_.text(x=y_regr.min(), y=0.85, s=self.metadata_TeX[i], size=6)
            # rms
            rms = np.sqrt(np.mean((y_regr - y)**2))
            ax_.text(x=y_regr.min(), y=0.775, s='rms = {:.3f}'.format(rms),
                     size=6)

            locx = mticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10])
            locy = mticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10])
            locy_ = mticker.NullLocator()
            ax.xaxis.set_major_locator(locx)
            ax.yaxis.set_major_locator(locy)
            ax_.yaxis.set_major_locator(locy_)

            ax.tick_params(axis='both', color='k', labelsize=6)

        fig.suptitle(t=r'$Z + A \cdot X$ vs $\{P_i\}$')

        fig.savefig(os.path.join(self.basedir, 'param_regr_{}.png'.format(self.src)),
                    dpi=300)

    def make_PC_param_importance_fig(self):

        fig = plt.figure(figsize=(4, 3), dpi=300)
        ax = fig.add_subplot(111)

        X = self.find_PC_param_coeffs()[:, :-1] #  (p, q)
        C = self.trn_PC_wts #  (n, q)
        P = self.metadata_a #  (n, p)

        N_PC_a = np.abs(X[:, None, :] * C[None, :, :]).sum(axis=1)

        F_PC_a = N_PC_a / N_PC_a.sum(axis=1)[:, None]

        cyc_color = cycler(color=['#1b9e77','#d95f02','#7570b3'])
        # qualitative colorblind cycle from ColorBrewer
        cyc_marker = cycler(marker=['o', '>', 's', 'd', 'x'])
        cyc_prop = cyc_marker * cyc_color

        p, q = X.shape
        for i, (sty, k) in enumerate(zip(cyc_prop,
                                         self.metadata.colnames)):
            # plot each param's dependence on each PC
            TeX = self.metadata[k].meta.get('TeX', k)
            pc_num = np.linspace(1, q, q)
            fpc = F_PC_a[i, :]
            ax.plot(pc_num, fpc, label=TeX, markersize=2,
                    **sty)

        ax.set_xlabel('PC')
        ax.set_xticks(np.linspace(1, q, q).astype(int))
        ax.set_ylabel(r'$F_{PC}(\alpha)$')
        ax.legend(loc='best', prop={'size': 5})

        ax2 = ax.twinx()
        ax2.plot(np.linspace(1, q, q), (1. - self.PVE.cumsum()),
                 c='c', linestyle='--', marker='None')
        ax2.set_yscale('log')
        ax2.set_ylim([1.0e-3, 1.])
        ax2.set_ylabel('fraction unexplained variance', size=5)
        ax2.yaxis.label.set_color('c')
        ax2.tick_params(axis='y', colors='c', labelsize=5)

        ax.set_xlim([0, q + 1.5])

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.basedir, 'PC_param_importance_{}.png'.format(self.src)),
            dpi=300)

    def make_prior_fig(self):
        nparams = len(self.metadata.colnames)
        gs, fig = figures_tools.gen_gridspec_fig(N=nparams)

        for i, n in enumerate(self.metadata.colnames):
            # set up subplots
            ax = fig.add_subplot(gs[i])
            label = self.metadata[n].meta.get('TeX', n)
            ax.hist(self.metadata[n].flatten(), bins=50, histtype='step',
                    linewidth=0.5)
            ax.tick_params(labelsize='xx-small')
            ax.set_yticks([])
            ax.set_xlabel(label, size='x-small')

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.basedir, 'prior_allparams.png'))


    # =====
    # properties
    # =====

    @property
    def Cov_th(self):
        R = (self.normed_trn_spectra - self.mean_trn_spectrum) - \
            self.trn_recon

        return np.cov(R)

    @property
    def l_lower(self):
        return 10.**(self.logl - self.dlogl / 2)

    @property
    def l_upper(self):
        return 10.**(self.logl + self.dlogl / 2)

    @property
    def dl(self):
        return self.l_upper - self.l_lower

    # =====
    # under the hood
    # =====

    def __str__(self):
        return 'PCA object: q = {0[0]}, l = {0[1]}'.format(self.PCs.shape)


class PCAError(Exception):
    '''
    general error for PCA
    '''
    pass


class PCProjectionWarning(UserWarning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HistFailedWarning(UserWarning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def select_cubesequence_from_start(a, i0, nl):
    '''
    select sequence along axis of cube with different starting indices
    '''
    mapshape = a.shape[-2:]
    # construct map indexing arrays
    II, JJ = np.meshgrid(*map(range, mapshape), indexing='ij')
    II, JJ = II[None, ...], JJ[None, ...]
    # all axis-0 indices for left-contributors
    LL_all = np.arange(nl, dtype=int)[:, None, None] + i0[None, :, :]

    LL_all_split = np.array_split(LL_all, 100, axis=0)

    # contributor arrays: we extract all at once because advanced indexing
    # copies data, and this means we only have to do it once per flux or ivar
    a_all = np.concatenate([a[LL_sect, II, JJ] for LL_sect in LL_all_split], axis=0)

    return a_all

def conservative_maskprop(a, i0, nl):
    '''
    propagate masks in most conservative fashion: masked pixels,
        and their neighbors, are all masked
    '''
    mask = np.logical_or.reduce(
        [select_cubesequence_from_start(a, i0 + m, nl) for m in [-1, 0, 1]])
    return mask

class PCA_Result(object):

    '''
    store results of PCA for one galaxy using this
    '''

    def __init__(self, pca, dered, K_obs, z, cosmo, figdir='.',
                 truth=None, truth_sfh=None, dered_method='nearest',
                 dered_kwargs={}, pc_cov_method='full_iter',
                 cov_sys_incr=4.0e-4):
        self.objname = dered.drp_hdulist[0].header['plateifu']
        self.pca = pca
        self.dered = dered
        self.cosmo = cosmo
        self.z = z
        self.K_obs = K_obs
        self.truth = truth  #  known-truth parameters for fake data
        self.truth_sfh = truth_sfh  #  known-true SFH for fake data

        # where to save all figures
        self.figdir = figdir
        self.__setup_figdir__()

        self.E = pca.PCs
        self.l = 10.**self.pca.logl
        self.M = self.pca.M

        self.O, self.ivar, self.mask_spax = dered.correct_and_match(
            template_logl=pca.logl, template_dlogl=pca.dlogl,
            method=dered_method, dered_kwargs=dered_kwargs)
 
        # compute starting index of obs cov for each spaxel
        self.i0_map = self.pca._compute_i0_map(self.K_obs.logl, self.dered.z_map)

        self.drppixmask = conservative_maskprop(
            m.mask_from_maskbits(dered.drp_hdulist['MASK'].data, [0, 3, 10]),
            self.i0_map, len(pca.l))
        self.eline_mask = dered.compute_eline_mask(
            template_logl=pca.logl, template_dlogl=self.pca.dlogl,
            half_dv=300. * u.km / u.s)

        self.nl, *self.map_shape = self.O.shape
        self.map_shape = tuple(self.map_shape)
        self.ifu_ctr_ix = [s // 2 for s in self.map_shape]

        self.SNR_med = np.median(self.O * np.sqrt(self.ivar) + eps,
                                 axis=0)
        # no data
        self.nodata = (dered.drp_hdulist['RIMG'].data == 0.)

        # guess bad data not caught in drp pixel mask
        self.guessbaddata = ut.find_bad_data(self.O, self.ivar, wid=51)

        # combine masks
        self.to_impute = np.logical_or.reduce((
            self.drppixmask, self.guessbaddata, self.eline_mask))
        self.mask_cube = np.logical_or(
            self.to_impute, np.logical_or(self.nodata, self.mask_spax)[None, ...])

        # normalize data
        self.O_norm, self.a_map = self.pca.scaler(self.O)
        self.ivar_norm = self.ivar * self.a_map**2.

        # subtract mean spectrum
        self.S = (self.O / self.a_map) - self.M[:, None, None]
        # censor masked values with weighted mean of nearby values
        self.S_cens = ut.replace_bad_data_with_wtdmean(
            self.S, self.ivar_norm, self.mask_cube, wid=101)

        # original spectrum
        self.O = np.ma.array(self.O, mask=self.mask_cube)
        self.O_norm = np.ma.array(self.O_norm, mask=self.mask_cube)

    def solve(self, vdisp_wt=False, cosmo_wt=True):
        '''
        packages together logic that solves for PC weights
        '''

        # solve for PC coefficients and covariances
        self.A, self.P_PC, self.fit_success = self.solve_cube()

        self.w = pca.compute_model_weights(P=self.P_PC, A=self.A)

        if cosmo_wt:
            # disallow models that are at too high a redshift for their age
            tf_earliest = self.cosmo.age(0.).value - self.cosmo.age(self.z).value
            tsc = .1
            self.w *= np.minimum(
                1., np.exp(
                    -(tf_earliest - self.pca.metadata['tf'][:, None, None]) / tsc))

        if vdisp_wt:
            vdisp_fill = 30.
            vdisp_raw = self.dered.dap_hdulist['STELLAR_SIGMA'].data
            vdisp_corr = self.dered.dap_hdulist['STELLAR_SIGMACORR'].data
            vdisp2 = vdisp_raw**2. - vdisp_corr**2.
            vdisp = np.sqrt(vdisp2)
            vdisp[vdisp2 <= 0.] = vdisp_fill

            vdisp_ivar = self.dered.dap_hdulist['STELLAR_SIGMA_IVAR'].data
            vdisp_bitmask = m.mask_from_maskbits(
                a=self.dered.dap_hdulist['STELLAR_SIGMA_MASK'].data, b=[30])
            vdisp_ivar *= (~ vdisp_bitmask)
            vdisp_ivar[vdisp2 <= 0.] = (3. * vdisp_fill)**-2.
            vdisp_ivar[vdisp_ivar < 1.0e-8] = 1.0e-8
            vdisp_ivar[vdisp_ivar > 100.] = 100.
            vdisp_wts = ut.gaussian_weightify(
                mu=vdisp, ivar=vdisp_ivar, vals=self.pca.metadata['sigma'].data,
                soft=4.)
            self.w *= vdisp_wts
        else:
            pass

        self.mask_map = np.logical_or.reduce(
            (self.mask_spax, self.nodata))

        # spaxel is bad if < 25 models have weights 1/100 max, and no other problems
        self.badPDF = np.logical_and.reduce(
            ((self.sample_diag(f=.01) < 10), ~self.mask_map))
        self.goodPDF = ~self.badPDF

    def solve_cube(self):
        '''
        '''
        var_norm = 1. / self.ivar_norm
        
        solver = PCAProjectionSolver(
            e=self.E, K_inst_cacher=self.K_obs, K_th=self.pca.cov_th, regul=1.0e-2)

        solve_all = np.vectorize(
            solver.solve_single, signature='(l),(l),(l),(),(),()->(q),(q,q),()',
            otypes=[np.ndarray, np.ndarray, bool])

        A, P_PC, success = solve_all(
            np.moveaxis(self.S_cens, 0, -1), np.moveaxis(var_norm, 0, -1),
            np.moveaxis(self.mask_cube, 0, -1), self.a_map, self.i0_map, self.nodata)

        P_PC = np.moveaxis(P_PC, [0, 1, 2, 3], [2, 3, 0, 1]).astype(float)
        A = np.moveaxis(A, -1, 0).astype(float)
        return A, P_PC, success

    def reconstruct(self):
        '''
        spectral reconstruction logic
        '''
        self.O_recon = np.ma.array(pca.reconstruct_normed(self.A),
                                   mask=self.mask_cube)

        self.resid = (self.O_recon - self.O_norm)

    def fluxdens(self, band='i'):
        '''
        return spaxel map of flux in the specified bandpass
        '''

        flux_im = (self.dered.drp_hdulist[
            '{}IMG'.format(band)].data * 3.631e-6 * u.Jy)

        return flux_im

    def lum(self, band='i'):
        '''
        return spaxel map estimate of luminosity, in solar units

        Retrieves the correct bandpass image, and converts to Lsun assuming
            some cosmology and redshift
        '''

        # retrieve k-corrected apparent AB mag from dered object
        ABmag = self.dered.S2P_rest.ABmags['-'.join(
            ('sdss2010', band))]

        # convert to an absolute magnitude
        ABMag = ABmag - 5. * np.log10(
            (self.dist / (10. * u.pc)).to('').value)

        # convert to solar units
        M_sun = Msun[band]
        Lsun = 10.**(-0.4 * (ABMag - M_sun))

        return Lsun

    def lum_plot(self, ax, ix, band='i'):

        im = ax.imshow(
            np.log10(np.ma.array(self.lum(band=band), mask=self.mask_map)),
            aspect='equal')

        cb = plt.colorbar(im, ax=ax, pad=0.025)
        cb.set_label(r'$\log{\mathcal{L}}$ [$L_{\odot}$]', size=8)
        cb.ax.tick_params(labelsize=8)

        Lstar_tot = np.ma.array(self.lum(band=band), mask=self.mask_map).sum()
        ax.axhline(ix[0], c='k')
        ax.axvline(ix[1], c='k')

        ax.text(x=0.2, y=0.2,
                s=''.join((r'$\log{\frac{\mathcal{L}_{*}}{L_{\odot}}}$ = ',
                           '{:.2f}'.format(np.log10(Lstar_tot)))))

        ax.set_title('{}-band luminosity'.format(band), size=8)

        self.__fix_im_axs__(ax, bad=False)
        ax.grid(False)

        return im, cb

    def comp_plot(self, ax1, ax2, ix=None):
        '''
        make plot illustrating fidelity of PCA decomposition in reproducing
            observed data
        '''

        if ix is None:
            ix = self.ifu_ctr_ix

        allzeroweights = (self.w[:, ix[0], ix[1]].max() == 0.)

        # best fitting spectrum
        if not allzeroweights:
            bestfit = self.pca.normed_trn[np.argmax(self.w[:, ix[0], ix[1]]), :]
            bestfit_ = ax1.plot(self.l, bestfit, drawstyle='steps-mid',
                            c='c', label='Best Model', linewidth=0.5, zorder=0)
        else:
            bestfit_ = None

        # original & reconstructed
        O_norm = self.O_norm[:, ix[0], ix[1]]
        O_recon = self.O_recon[:, ix[0], ix[1]]
        ivar = self.ivar_norm[:, ix[0], ix[1]]

        Onm = np.ma.median(O_norm)
        Orm = np.ma.median(O_recon)
        Or95 = np.percentile(O_recon, 95)
        ivm = np.ma.median(ivar)

        orig_ = ax1.plot(self.l, O_norm, drawstyle='steps-mid',
                         c='b', label='Obs.', linewidth=0.25, zorder=1)
        recon_ = ax1.plot(self.l, O_recon, drawstyle='steps-mid',
                          c='g', label='PCA Fit', linewidth=0.25, zorder=2)
        ax1.axhline(y=0., xmin=self.l.min(), xmax=self.l.max(),
                    c='k', linestyle=':')

        # inverse-variance (weight) plot
        ivar_ = ax1.plot(
            self.l, ivar / ivm, drawstyle='steps-mid', c='m',
            label='IVAR', linewidth=0.5, zorder=0, visible=False)

        # residual plot
        resid = self.resid[:, ix[0], ix[1]]
        std_err = (1. / np.sqrt(ivar))
        fit_resid_ = ax2.plot(
            self.l.data, resid, drawstyle='steps-mid', c='green',
            linewidth=0.5, alpha=.5)
        if not allzeroweights:
            model_resid = ax2.plot(
                self.l.data, bestfit - O_norm, drawstyle='steps-mid', c='cyan',
                linewidth=0.5, alpha=.5)
        conf_band_ = ax2.fill_between(
            x=self.l.data, y1=-std_err, y2=std_err,
            linestyle='--', color='salmon', linewidth=0.25, zorder=0)

        ax1.tick_params(axis='y', which='major', labelsize=10,
                        labelbottom=False)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        # force sharing of x axis
        ax1.get_shared_x_axes().join(ax1, ax2)

        ax1.xaxis.set_major_locator(self.lamticks)
        ax1.xaxis.set_ticklabels([])
        ax1.legend(loc='best', prop={'size': 6})
        ax1.set_ylabel(r'$F_{\lambda}$ (rel)')
        ax1.set_ylim([-0.05 * Or95, 1.1 * Or95])
        ax1.set_yticks(np.arange(0.0, ax1.get_ylim()[1], 0.5))

        ax2.xaxis.set_major_locator(self.lamticks)
        ax2.set_xlabel(r'$\lambda$ [$\textrm{\AA}$]')
        ax2.set_ylim([-3. * std_err.mean(), 3. * std_err.mean()])
        ax2.set_ylabel('Resid.')

        return orig_, recon_, bestfit_, ivar_, fit_resid_, conf_band_, ix

    def make_comp_fig(self, ix=None):
        fig = plt.figure(figsize=(8, 3.5), dpi=300)

        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax = fig.add_subplot(gs[0])
        ax_res = fig.add_subplot(gs[1])

        _, _, _, _, _, _, ix = self.comp_plot(ax1=ax, ax2=ax_res, ix=ix)

        fig.suptitle('{0}: ({1[0]}, {1[1]})'.format(self.objname, ix))

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        fname = 'comp_{0}_{1[0]}-{1[1]}.png'.format(self.objname, ix)

        self.savefig(fig, fname, self.figdir, dpi=300)

        return fig

    def qty_map(self, qty_str, ax1, ax2, f=None, norm=[None, None],
                logify=False, TeX_over=None):
        '''
        make a map of the quantity of interest, based on the constructed
            parameter PDF

        params:
         - qty_str: string designating which quantity from self.metadata
            to access
         - ax1: where median map gets shown
         - ax2: where sigma map gets shown
         - f: factor to multiply percentiles by
         - log: whether to take log10 of
        '''

        P50, l_unc, u_unc, scale = self.pca.param_cred_intvl(
            qty=qty_str, factor=f, W=self.w,
            mask=np.logical_or(self.mask_map, ~self.fit_success))

        if not TeX_over:
            med_TeX = self.pca.metadata[qty_str].meta.get('TeX', qty_str)
        else:
            med_TeX = TeX_over

        # manage logs for computation and display simultaneously
        if logify and (scale == 'log'):
            raise ValueError('don\'t double-log a quantity!')
        elif logify:
            P50 = np.log10(P50)
            unc = np.log10((u_unc + l_unc) / 2.)
            sigma_TeX = r'$\sigma~{\rm [dex]}$'
            #med_TeX = ''.join((r'$\log$', med_TeX))
        elif (scale == 'log'):
            unc = (u_unc + l_unc) / 2.
            sigma_TeX = r'$\sigma~{\rm [dex]}$'
            #med_TeX = ''.join((r'$\log$', med_TeX))
        else:
            unc = (l_unc + u_unc) / 2.
            sigma_TeX = r'$\sigma$'

        mask = self.mask_map

        m_vmin, m_vmax = np.percentile(np.ma.array(P50, mask=mask).compressed(), [2., 98.])
        m = ax1.imshow(
            np.ma.array(P50, mask=mask),
            aspect='equal', norm=norm[0], vmin=m_vmin, vmax=m_vmax)

        s_vmin, s_vmax = np.percentile(np.ma.array(unc, mask=mask).compressed(),
                                       [2., 98.])
        s = ax2.imshow(
            np.ma.array(unc, mask=mask),
            aspect='equal', norm=norm[1], vmin=s_vmin, vmax=s_vmax)

        mcb = plt.colorbar(m, ax=ax1, pad=0.025)
        mcb.set_label(med_TeX, size='xx-small')
        mcb.ax.tick_params(labelsize='xx-small')

        scb = plt.colorbar(s, ax=ax2, pad=0.025)
        scb.set_label(sigma_TeX, size=8)
        scb.ax.tick_params(labelsize='xx-small')

        return m, s, mcb, scb, scale

    def make_qty_fig(self, qty_str, qty_tex=None, qty_fname=None, f=None,
                     logify=False, TeX_over=None):
        '''
        make a with a map of the quantity of interest

        params:
         - qty_str: string designating which quantity from self.metadata
            to access
         - qty_tex: valid TeX for plot
         - qty_fname: override for final filename (usually used when `f` is)
         - f: factor by which to multiply map
        '''
        if qty_fname is None:
            qty_fname = qty_str

        if qty_tex is None:
            qty_tex = self.pca.metadata[qty_str].meta.get(
                'TeX', qty_str)

        fig, gs, ax1, ax2 = self.__setup_qty_fig__()

        m, s, mcb, scb, scale = self.qty_map(
            qty_str=qty_str, ax1=ax1, ax2=ax2, f=f, logify=logify,
            TeX_over=TeX_over)

        fig.suptitle('{}: {}'.format(self.objname, qty_tex))

        self.__fix_im_axs__([ax1, ax2])
        fname = '{}-{}.png'.format(self.objname, qty_fname)
        self.savefig(fig, fname, self.figdir, dpi=300)

        return fig

    def Mstar_tot(self, band='r'):
        qty_str = 'ML{}'.format(band)
        f = self.lum(band=band)

        P50, *_, scale = self.pca.param_cred_intvl(
            qty=qty_str, factor=f, W=self.w,
            mask=np.logical_or(self.mask_map, ~self.fit_success))

        if scale == 'log':
            return 10.**P50

        return P50

    def Mstar_integrated(self, band='i'):
        '''
        calculate integrated spectrum, and then compute stellar mass from that
        '''

        _, O, ivar = self.dered.coadd(tem_l=self.pca.l, good=~self.mask_map)
        O_cens_cube = (self.S_cens + self.M[:, None, None]) * self.a_map
        var_cube = 1. / self.ivar_norm

        O_sum = O_cens_cube.sum(axis=(1, 2), keepdims=True)
        var_sum = var_cube.sum(axis=(1, 2), keepdims=True)

        # in the spaxels that were used to coadd, OR-function
        # of mask, over cube
        mask = (np.mean(self.mask_cube * (~self.mask_map[None, ...]),
                axis=(1, 2)) > 0.1)[..., None, None]

        # normalize data
        O_norm, a = self.pca.scaler(O_sum)
        a = a.mean()
        var_norm = var_sum / a**2.

        # subtract mean spectrum
        S = O_norm - self.M[:, None, None]

        solver = PCAProjectionSolver(
            e=self.E, K_inst_cacher=self.K_obs, K_th=self.pca.cov_th)
        solve_all = np.vectorize(
            solver.solve_single, signature='(l),(l),(l),(),(),()->(q),(q,q),()',
            otypes=[np.ndarray, np.ndarray, bool])

        i0 = np.round(np.average(
                      self.i0_map, weights=self.a_map, axis=(0, 1)), 0).astype(int)

        A, P_PC, success = solver.solve_single(
            S.squeeze(), var_norm.squeeze(),
            mask.squeeze(), a, i0, False)

        w = pca.compute_model_weights(P=P_PC[..., None, None], A=A[..., None, None])

        lum = np.ma.masked_invalid(self.lum(band=band))

        # this SHOULD and DOES call the method in PCA rather than
        # in self, since we aren't using self.w
        P50, *_, scale = self.pca.param_cred_intvl(
            qty='ML{}'.format(band), factor=lum.sum(keepdims=True), W=w,
            mask=np.logical_or(self.mask_map, ~self.fit_success))

        if scale == 'log':
            ret = (10.**P50).sum()
        else:
            ret = P50.sum()

        return ret

    def Mstar_surf(self, band='r'):
        spaxel_psize = (self.dered.spaxel_side * self.dist).to(
            'kpc', equivalencies=u.dimensionless_angles())
        # print spaxel_psize
        sig = self.Mstar(band=band) * u.Msun / spaxel_psize**2.
        return sig.to('Msun pc-2').value

    def Mstar_map(self, ax1, ax2, band='i'):
        '''
        make two-axes stellar-mass map

        use stellar mass-to-light ratio PDF

        params:
         - ax1, ax2: axes for median and stdev, passed along
         - band: what bandpass to use
        '''

        from utils import lin_transform as tr

        f = self.lum(band=band)

        qty = 'ML{}'.format(band)
        # log-ify if ML is in linear space
        logify = (self.pca.metadata[qty].meta.get(
                  'scale', 'linear') == 'linear')

        TeX_over = r'$\log M^*_{{{}}} {{\rm [M_{{\odot}}]}}$'.format(band)

        m, s, mcb, scb, scale = self.qty_map(
            ax1=ax1, ax2=ax2, qty_str=qty, f=f, norm=[None, None],
            logify=logify, TeX_over=TeX_over)

        return m, s, mcb, scb

    def make_Mstar_fig(self, band='i'):
        '''
        make stellar-mass figure
        '''

        qty_str = 'Mstar_{}'.format(band)
        qty_tex = r'$\log M_{{*,{}}}$'.format(band)

        fig, gs, ax1, ax2 = self.__setup_qty_fig__()

        self.Mstar_map(ax1=ax1, ax2=ax2, band=band)
        fig.suptitle(' '.join((self.objname, ':', qty_tex)))

        self.__fix_im_axs__([ax1, ax2])

        fname = '{}-{}.png'.format(self.objname, qty_str)
        self.savefig(fig, fname, self.figdir, dpi=300)

        return fig

    def logQH(self, band='i', P=[16., 50., 84.]):
        logQHpersolmass = self.pca.metadata['logQHpersolmass']
        logML = self.pca.metadata['ML{}'.format(band)]
        loglum = np.log10(self.lum(band))
        logQHperlum = logQHpersolmass + logML

        A = param_interp_map(v=logQHperlum, w=self.w, pctl=np.array(P),
                             mask=np.logical_or(self.mask_map, ~self.fit_success))
        A = A[:, None, None] + loglum[None, :, :]
        return A

    def make_logQH_hdu(self):
        V = self.logQH(band='i', P=[16., 50., 84.])
        P16, P50, P84 = tuple(map(np.squeeze, np.split(V, 3, axis=0)))
        l_unc, u_unc = P84 - P50, P50 - P16

        qty_hdu = fits.ImageHDU(np.stack([P50, l_unc, u_unc]))
        qty_hdu.header['LOGSCALE'] = False
        qty_hdu.header['CHANNEL0'] = 'median'
        qty_hdu.header['CHANNEL1'] = 'lower uncertainty'
        qty_hdu.header['CHANNEL2'] = 'upper uncertainty'
        qty_hdu.header['QTYNAME'] = 'logQH'
        qty_hdu.header['EXTNAME'] = 'logQH'

        return qty_hdu


    def qty_kde(self, q, **kwargs):
        '''
        Construct and evaluate KDE for some array `q`,
            passing other kwargs to KDE.fit()
        '''

        kde = KDEUnivariate(q)
        kde.fit(**kwargs)
        qgrid = np.linspace(q.min(), q.max(), len(q))
        pgrid = np.array([kde.evaluate(q) for q in qgrid])
        pgrid /= pgrid.max()

        return qgrid, pgrid

    def qty_errorbar(self, q, w, ax):
        '''
        add errobar notation onto histogram
        '''

        # reorder by param value
        i_ = np.argsort(q)
        q_, w_ = q[i_], w[i_]
        p16, p50, p84 = np.interp(
            xp=100. * w_.cumsum() / w_.sum(), fp=q_,
            x=[16., 50., 84.], left=q_.min(), right=q_.max())
        uerr = np.abs(p84 - p50)
        lerr = np.abs(p50 - p16)

        yllim, yulim = ax.get_ylim()
        ypos = 0.25 * yulim

        ax.errorbar(
            x=[p50], y=ypos, yerr=None, xerr=[[lerr], [uerr]],
            marker='d', markerfacecolor='g', markeredgecolor='None',
            ecolor='g')

    def param_bestmodel(self, q, w, ax):
        # value of best-fit spectrum
        ax.axvline(q[np.argmax(w)], color='c', linewidth=0.5, label='best')

    def qty_hist(self, qty, ix=None, ax=None, f=None, bins=50,
                 legend=False, kde=(False, False), logx=False):
        if ix is None:
            ix = self.ifu_ctr_ix

        if ax is None:
            ax = plt.gca()

        if f is None:
            f = np.ones_like(self.pca.metadata[qty])

        if logx:
            ax.set_xscale('log')

        # whether to use KDE to plot prior and/or posterior
        kde_prior, kde_post = kde

        q = self.pca.metadata[qty]
        w = self.w[:, ix[0], ix[1]]
        isfin = np.isfinite(q)
        q, w = q[isfin], w[isfin]

        if len(q) == 0:
            return None

        TeX = self.pca.metadata[qty].meta.get('TeX', qty)

        scale = self.pca.metadata[qty].meta.get('scale')

        ax_ = ax.twinx()

        # marginalized posterior
        if kde_post:
            qgrid, postgrid = self.qty_kde(
                q=q, weights=w, kernel='gau', bw='scott', fft=False)
            h = ax.plot(qgrid, postgrid, color='k', linestyle='-',
                        label='posterior', linewidth=0.5)
        else:
            try:
                h = ax.hist(
                    q, weights=w, bins=bins, normed=True, histtype='step',
                    color='k', label='posterior', linewidth=0.5)
            except UnboundLocalError:
                h = None
                warn('{} post. hist failed'.format(qty),
                     HistFailedWarning)

        # marginalized prior
        if kde_prior:
            qgrid, prigrid = self.qty_kde(
                q=q, kernel='gau', bw='scott', fft=False)
            hprior = ax.plot(qgrid, prigrid, color='fuchsia', linestyle='-',
                             label='prior', linewidth=0.5)
        else:
            hprior = ax_.hist(
                q, bins=bins, normed=True, histtype='step', color='fuchsia',
                label='prior', linewidth=0.5)

        # log odds ratio
        if kde_prior and kde_post:
            ev_ax_ = ax.twinx()

            log_ev = np.log10(postgrid / prigrid)
            try:
                ev_ax_.plot(qgrid, lfog_ev, color='g', linestyle='--',
                            label='log-odds-ratio')
            except ValueError:
                pass
            he, le = ev_ax_.get_legend_handles_labels()
            ev_ax_.yaxis.label.set_color('g')
            ev_ax_.tick_params(axis='y', color='g', labelsize=8, labelcolor='g')
            ev_ax_.spines['right'].set_color('green')

            if np.median(np.abs(log_ev)) <= 1.0e-2:
                ev_ax_.set_ylim([-6., 1.])
            else:
                ev_ax_.set_ylim([log_ev.max() - 10., log_ev.max() + .1])

        else:
            he, le = [None, ], [None, ]

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_.get_legend_handles_labels()

        ax.yaxis.set_major_locator(plt.NullLocator())
        ax_.yaxis.set_major_locator(plt.NullLocator())

        self.qty_errorbar(q=q, w=w, ax=ax)
        self.param_bestmodel(q=q, w=w, ax=ax)

        # if we're using fake data, we have some ground truth!
        if self.truth is not None:
            truthval = self.truth[qty]
            ax.axvline(truthval, color='r', linewidth=0.5, label='truth')

        ax.set_xlabel(TeX)

        if legend:
            with catch_warnings():
                simplefilter('ignore')
                ax.legend(h1 + h2 + he, l1 + l2 + le, loc='best', prop={'size': 8})

        return h, hprior

    def orig_spax(self, ixx, ixy):
        return self.O[:, ixx, ixy]

    def recon_spax(self, ixx, ixy):
        return self.O_recon[:, ixx, ixy]

    def ivar_spax(self, ixx, ixy):
        return self.ivar[:, ixx, ixy]

    def param_vals_wts(self, ixx, ixy, pname):
        return np.array(self.pca.metadata[pname]), self.w[:, ixx, ixy]

    def __fix_im_axs__(self, axs, bad=True):
        '''
        do all the fixes to make quantity maps look nice in wcsaxes
        '''
        if type(axs) is not list:
            axs = [axs]

        # create a sky offset frame to overlay
        offset_frame = coord.SkyOffsetFrame(
            origin=coord.SkyCoord(*(self.wcs_header.wcs.crval * u.deg)))

        # over ax objects
        for ax in axs:
            # suppress native coordinate system ticks
            for ci in range(2):
                ax.coords[ci].set_ticks(number=5)
                ax.coords[ci].set_ticks_visible(False)
                ax.coords[ci].set_ticklabel_visible(False)
                ax.coords[ci].grid(False)

            # initialize overlay
            offset_overlay = ax.get_coords_overlay(offset_frame)
            offset_overlay.grid(True)
            offset_overlay['lon'].set_coord_type('longitude', coord_wrap=180.)
            ax.set_aspect('equal')

            for ck, abbr, pos in zip(['lon', 'lat'], [r'\alpha', r'\delta'], ['b', 'l']):
                offset_overlay[ck].set_axislabel(
                    r'$\Delta {}~["]$'.format(abbr), size='x-small')
                offset_overlay[ck].set_axislabel_position(pos)
                offset_overlay[ck].set_ticks_position(pos)
                offset_overlay[ck].set_ticklabel_position(pos)
                offset_overlay[ck].set_format_unit(u.arcsec)
                offset_overlay[ck].set_ticks(number=5)
                offset_overlay[ck].set_major_formatter('s.s')

            if bad:
                # figures_tools.annotate_badPDF(ax, self.goodPDF)
                pass

    def __setup_qty_fig__(self):
        fig = plt.figure(figsize=(9, 4), dpi=300)

        gs = gridspec.GridSpec(1, 2, wspace=.2, left=.085, right=.975,
                               bottom=.11, top=.9)
        ax1 = fig.add_subplot(gs[0], projection=self.wcs_header)
        ax2 = fig.add_subplot(gs[1], projection=self.wcs_header)

        # overplot hatches for masks
        # start by defining I & J pixel grid
        II, JJ = np.meshgrid(*(np.linspace(-.5, ms_ - .5, ms_ + 1)
                               for ms_ in self.map_shape))
        IIc, JJc = map(lambda x: 0.5 * (x[:-1, :-1] + x[1:, 1:]), (II, JJ))

        for ax in [ax1, ax2]:
            # pcolor plots are masked where the data are GOOD
            # badpdf mask
            ax.pcolor(II, JJ,
                      np.ma.array(np.zeros_like(IIc), mask=~self.badPDF),
                      hatch='\\'*8, alpha=0.)
            # dered mask
            ax.pcolor(II, JJ,
                      np.ma.array(np.zeros_like(IIc), mask=~self.mask_map),
                      hatch='/'*8, alpha=0.)
            # fit unsuccessful
            ax.pcolor(II, JJ,
                      np.ma.array(np.zeros_like(IIc), mask=self.fit_success),
                      hatch='.' * 8, alpha=0.)

        return fig, gs, ax1, ax2

    def __setup_figdir__(self):
        if not os.path.isdir(self.figdir):
            os.makedirs(self.figdir)

    def savefig(self, *args, **kwargs):
        '''
        wrapper around figures_tools.savefig
        '''
        figures_tools.savefig(*args, **kwargs)

    def map_add_loc(self, ax, ix, **kwargs):
        '''
        add axvline and axhline at the location in the map corresponding to
            some image-frame indices ix
        '''

        pix_coord = ax.wcs.all_pix2world(
            np.atleast_2d(ix), origin=1)

        ax.axhline(pix_coord[1], **kwargs)
        ax.axvline(pix_coord[0], **kwargs)

    def make_full_QA_fig(self, ix=None, kde=(False, False)):
        '''
        use matplotlib to make a full map of the IFU grasp, including
            diagnostic spectral fits, and histograms of possible
            parameter values for each spaxel
        '''

        from utils import matcher

        nparams = len(self.pca.confident_params)
        ncols = 3
        nrows = nparams // ncols + (nparams % ncols != 0)

        wper = 3
        hper = 2
        htoprow = 2.5
        lborder = rborder = 0.25
        tborder = bborder = 0.5
        lborder1, rborder1 = 0.75, 0.25
        fig_height = hper * nrows + htoprow + tborder + bborder
        fig_width = wper * ncols

        llim, rlim = lborder / fig_width, 1. - (rborder / fig_width)
        llim1, rlim1 = lborder1 / fig_width, 1. - (rborder1 / fig_width)
        lolim, uplim = bborder / fig_height, 1. - (tborder / fig_height)

        gs1_loborder = 1. - (tborder + htoprow) / fig_height
        gs2_hiborder = (bborder + hper * nrows) / fig_height

        plt.close('all')

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)

        # gridspec used for map + spec_compare
        gs1 = gridspec.GridSpec(
            3, 4, bottom=gs1_loborder, top=uplim,
            height_ratios=[3, 1, 1], width_ratios=[2, 0.5, 2, 2],
            hspace=0., wspace=.1, left=llim1, right=rlim1)

        gs2 = gridspec.GridSpec(
            nrows, ncols, bottom=lolim, top=gs2_hiborder,
            left=llim, right=rlim, hspace=.35)

        # put the spectrum and residual here!
        spec_ax = fig.add_subplot(gs1[0, 2:])
        resid_ax = fig.add_subplot(gs1[1, 2:])
        orig_, recon_, bestfit_, ivar_, resid_, resid_avg_, ix_ = \
            self.comp_plot(ax1=spec_ax, ax2=resid_ax, ix=ix)

        # image of galaxy in integrated light
        im_ax = fig.add_subplot(gs1[:-1, 0],
                                projection=self.wcs_header)
        lumim, lcb = self.lum_plot(im_ax, ix=ix_, band='r')

        # loop through parameters of interest, and make a weighted
        # histogram for each parameter
        enum_ = enumerate(zip(gs2, self.pca.confident_params))
        for i, (gs_, q) in enum_:
            ax = fig.add_subplot(gs_)
            is_ML = matcher(q, 'ML')
            if is_ML:
                bins = np.linspace(-1.5, 2, 50)
                if self.pca.metadata[q].meta.get('scale', 'linear') != 'log':
                    pass # bins = 10.**bins
            else:
                bins = 50
            if i == 0:
                legend = True
            else:
                legend = False
            h_, hprior_ = self.qty_hist(
                qty=q, ix=ix, ax=ax, bins=bins, legend=legend,
                kde=kde, logx=False)
            ax.tick_params(axis='both', which='major', labelsize=10)

        plt.suptitle('{0}: ({1[0]}-{1[1]})'.format(self.objname, ix_))

        fname = '{0}_fulldiag_{1[0]}-{1[1]}.png'.format(self.objname, ix_)
        self.savefig(fig, fname, self.figdir, dpi=300)

    def color_ML_plot(self, mlb='i', b1='g', b2='r', ax=None, ptcol='r', lab=None):
        '''
        plot color vs mass-to-light ratio, colored by radius/Re
        '''

        if ax is None:
            ax = plt.gca()

        # b1 - b2 color
        b1_ = '-'.join(('sdss2010', b1))
        b2_ = '-'.join(('sdss2010', b2))
        col = self.dered.S2P_rest.color(b1_, b2_)
        col = np.ma.array(col, mask=self.mask_map)
        # retrieve ML ratio
        ml, *_, scale = self.pca.param_cred_intvl(
            qty='ML{}'.format(mlb), W=self.w)

        if scale == 'linear':
            ml = np.log10(ml)

        # size of points determined by signal in redder band
        b2_img = self.dered.drp_hdulist['{}img'.format(b2)].data
        s = 10. * np.arctan(0.05 * b2_img / np.median(b2_img[b2_img > 0.]))

        sc = ax.scatter(col.flatten(), ml.flatten(),
                        c=ptcol.flatten(), edgecolor='None', s=s.flatten(),
                        label=self.objname)

        if type(ptcol) is not str:
            cb = plt.colorbar(sc, ax=ax, pad=.025)
            if lab is not None:
                cb.set_label(lab)

        # spectrophot.py includes conversion from many colors to many M/L ratios
        # from Bell et al -- of form $\log{(M/L)} = a_{\lambda} + b_{\lambda} * C$
        CML_row = CML.loc['{}{}'.format(b1, b2)]
        a_lam = CML_row['a_{}'.format(mlb)]
        b_lam = CML_row['b_{}'.format(mlb)]

        def bell_ML(col):
            return a_lam + (b_lam * col)

        def midpoints(a):
            return 0.5*(a[1:] + a[:-1])

        # plot the predicted Bell et all MLs
        ax.set_xlim([-0.25, 2.25])
        col_grid = np.linspace(*ax.get_xlim(), 90)

        # plot the predicted MLRs from Bell
        ML_pred = bell_ML(col_grid)
        ax.plot(col_grid, ML_pred, c='magenta', linestyle='--', label='Bell et al. (2003)')
        ax.legend(loc='best', prop={'size': 6})

        # plot IFU-integrated colors and mass-to-lights
        ml_integr = np.average(
            ml, weights=self.dered.drp_hdulist['{}img'.format(mlb)].data)
        f1 = self.dered.drp_hdulist['{}img'.format(b1)].data.sum()
        f2 = self.dered.drp_hdulist['{}img'.format(b2)].data.sum()
        color_integr = -2.5 * np.log10(f1 / f2)

        ax.scatter([color_integr], [ml_integr], marker='x', c='r')

        ax.set_ylim([ML_pred.min(), ML_pred.max()])

        ax.set_xlabel(r'${0} - {1}$'.format(b1, b2))
        ax.set_ylabel(''.join((r'$\log$',
                               self.pca.metadata['ML{}'.format(mlb)].meta['TeX'])))

        return sc

    def make_color_ML_fig(self, mlb='i', b1='g', b2='i', colorby='R'):

        fig = plt.figure(figsize=(5, 5), dpi=300)

        ax = fig.add_subplot(111)
        ax.set_title(self.objname)

        if colorby == 'R':
            ptcol = self.dered.Reff
            ptcol_lab = r'$\frac{R}{R_e}$'
            cbstr = 'R'
        else:
            if type(colorby) is str:
                colorby = [colorby]
            ptcol = np.prod(np.stack([self.pca.param_cred_intvl(
                q, factor=None, W=self.w)[0] for q in colorby], axis=0), axis=0)
            ptcol_lab = ''.join(
                (self.pca.metadata[k].meta.get('TeX', k) for k in colorby))
            cbstr = '-'.join(colorby)

        self.color_ML_plot(mlb, b1, b2, ptcol=ptcol, lab=ptcol_lab)

        plt.tight_layout()

        fname = '{}_C{}{}ML{}-{}.png'.format(self.objname, b1, b2, mlb, cbstr)
        self.savefig(fig, fname, self.figdir, dpi=300)

    def sample_diag(self, f=.1, w=None):
        '''
        how many models are within factor f of best-fit?
        '''

        if w == None:
            w = self.w

        max_w = w.max(axis=0)[None, ...]
        N = ((w / max_w) > f).sum(axis=0)

        return N

    def kullback_leibler(self):
        '''
        compute the spaxel-wise Kullback-Leibler divergence (i.e., how much information
            is gained by the PCA analysis relative to the prior)
        '''

        return entropy(qk=self.w, pk=np.ones_like(self.w), base=2)

    def make_sample_diag_fig(self, f=[.5, .1]):
        '''
        fraction of models that have weights at least f[0] and f[1]
            as large as highest-weighted model

        this is basically an estimate of how well the models populate
            parameter space
        '''

        from utils import lin_transform as tr

        fig, gs, ax1, ax2 = self.__setup_qty_fig__()
        self.__fix_im_axs__([ax1, ax2])

        a1 = np.ma.array(self.sample_diag(f=f[0]), mask=self.mask_map)
        a2 = np.ma.array(self.sample_diag(f=f[1]), mask=self.mask_map)

        nmodels = len(self.pca.metadata)

        im1 = ax1.imshow(np.log10(a1 / nmodels),
                         aspect='equal', vmin=-np.log10(nmodels), vmax=0)
        im2 = ax2.imshow(np.log10(a2 / nmodels),
                         aspect='equal', vmin=-np.log10(nmodels), vmax=0)
        cb1 = plt.colorbar(im1, ax=ax1, shrink=0.8, orientation='vertical')
        cb2 = plt.colorbar(im2, ax=ax2, shrink=0.8, orientation='vertical')

        lab = r'$\log \frac{N_{good}}{N_{tot}}$'
        cb1.set_label(lab, size=8)
        cb1.ax.tick_params(labelsize=8)
        cb2.set_label(lab, size=8)
        cb2.ax.tick_params(labelsize=8)

        for ff, ax in zip(f, [ax1, ax2]):
            axxlims, axylims = ax.get_xlim(), ax.get_ylim()

            ax.text(x=tr((0, 1), axxlims, 0.05),
                    y=tr((0, 1), axylims, 0.05),
                    s=''.join((r'$f = $', '{}'.format(ff))))

        fig.suptitle(' '.join((self.dered.plateifu, 'good model fraction')))

        fname = '_'.join((self.dered.plateifu, 'goodmodels.png'))
        self.savefig(fig, fname, self.figdir, dpi=300)

    def compare_sigma(self):
        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = fig.add_subplot(111)

        sig_dap = self.dered.dap_hdulist['STELLAR_SIGMA'].data
        # apply correction for real data, since fake data already builds in LSF
        if self.truth is not None:
            sig_dap_corr = self.dered.dap_hdulist['STELLAR_SIGMACORR'].data
        else:
            sig_dap_corr = 0.
        sig_dap = np.sqrt(sig_dap**2. - sig_dap_corr**2.)
        sig_dap_mask = m.mask_from_maskbits(
            self.dered.dap_hdulist['STELLAR_SIGMA_MASK'].data, b=[30])
        sig_dap = np.ma.array(sig_dap, mask=sig_dap_mask).flatten()
        sig_dap_unc = 1. / np.sqrt(
            self.dered.dap_hdulist['STELLAR_SIGMA_IVAR'].data).flatten()

        sig_pca, sig_pca_lunc, sig_pca_uunc, _ = self.pca.param_cred_intvl(
            qty='sigma', W=self.w)
        sig_pca = np.ma.array(sig_pca, mask=self.mask_map).flatten()
        sig_pca_unc = np.row_stack([sig_pca_lunc.flatten(),
                                    sig_pca_uunc.flatten()])

        s_ = np.linspace(10., 350., 10.)

        ax.errorbar(x=sig_dap, y=sig_pca,
                    xerr=sig_dap_unc, yerr=sig_pca_unc,
                    capsize=0.5, capthick=0.25, linestyle='None', elinewidth=0.25,
                    ecolor='k', color='k', ms=0.5, alpha=0.25, marker='.')

        ax.plot(s_, s_, linestyle='--', marker='None', c='g')
        ax.set_xlabel('DAP value')
        ax.set_ylabel('PCA value')

        fig.tight_layout()
        ax.set_ylim([0., 700.])
        ax.set_ylim([0., 700.])

        fname = '_'.join((self.dered.plateifu, 'sigma_comp.png'))
        self.savefig(fig, fname, self.figdir, dpi=300)

    def sigma_vel(self):
        '''
        compare inferred velocity dispersion and DAP velocity field value

        this is intended to diagnose artificially high inferred veldisp
            due to integer-pixel deredshifting
        '''
        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = fig.add_subplot(111)

        vel = np.ma.array(self.dered.dap_hdulist['STELLAR_VEL'].data,
                          mask=self.mask_map)

        sig_pca, sig_pca_lunc, sig_pca_uunc, _ = self.pca.param_cred_intvl(
            qty='sigma', W=self.w)
        sig_pca = np.ma.array(sig_pca, mask=self.mask_map)
        sig_dap_corr = self.dered.dap_hdulist['STELLAR_SIGMACORR'].data
        sig_pca = np.ma.array(np.sqrt(sig_pca**2. - sig_dap_corr**2.),
                              mask=self.mask_map)

        # velocity width of pixel
        dv_pix = (self.pca.dlogl * np.log(10.) * c.c).to(u.km / u.s).value
        v_offset = vel % dv_pix

        ax.scatter(x=v_offset.flatten(), y=sig_pca.flatten(),
                   s=1., marker='.', c='k')

        if self.truth is not None:
            ax.axhline(self.truth['sigma'], c='r')

        ax.set_xlabel('Vel. offset')
        ax.set_ylabel(r'$\sigma$ (PCA)')

        fig.suptitle('Effects of integer-pixel deredshifting')
        fig.tight_layout()

        fname = '_'.join((self.dered.plateifu, 'sigma_vel.png'))

        self.savefig(fig, fname, self.figdir, dpi=300)

    def plot_sfh(self, i, w, ax, label=None, massnorm='mformed',
                 **kwargs):
        '''
        plot a single SFH
        '''

        ts, sfrs, fii = csp.retrieve_SFHs(
            filelist=self.pca.sfh_fnames, nsubpersfh=self.pca.nsubpersfh,
            nsfhperfile=self.pca.nsfhperfile, i=i, massnorm=massnorm)

        ax.plot(
            ts, w + sfrs[fii, :], alpha=w,
            linewidth=0.5, label=label, **kwargs)

    def plot_top_sfhs(self, ax, ix, n=10):
        '''
        overplot top `n` SFHs (normalized to 10^9 Msun **formed**)
        '''
        nper = len(self.pca.gen_dicts[0]['mu'])
        # find top ten SFHs
        w_spax = self.w[:, ix[0], ix[1]]
        # sort in descending weight order
        best_i = np.argsort(w_spax)[::-1][:(n - 1)] // nper
        ws = np.linspace(1., 0., n + 1)[:-1]

        for ci, (i, w) in enumerate(zip(best_i, ws)):
            self.plot_sfh(
                w=w, ax=ax, i=i, label='SFH {}'.format(i),
                c='C{}'.format(ci))

        if self.truth is not None:
            ts, *_ = csp.retrieve_SFHs(
                filelist=self.pca.sfh_fnames, i=0)
            mnorm = np.trapz(x=ts, y=self.truth_sfh)
            ax.plot(ts, self.truth_sfh / mnorm, linewidth=1., label='Truth', c='r')

    def make_top_sfhs_fig(self, ix=None, n=10, fig=None, ax=None, ax_loc=111):

        if ix is None:
            ix = self.ifu_ctr_ix

        # create figure and axis if necessary
        if not (ax is None):
            pass
        elif (fig is None):
            fig = plt.figure(figsize=(3, 2), dpi=300)
            ax = fig.add_subplot(ax_loc)
        elif (ax is None):
            ax = fig.add_subplot(ax_loc)

        self.plot_top_sfhs(ax, ix, n=n)

        ax.set_title('Best-fitting SFHs - {}: ({}, {})'.format(
            self.objname, ix[0], ix[1]), size=6)

        ax.set_xlabel('time [Gyr]', size=8)
        ax.set_ylabel('Normed SFR ( + C)', size=8)

        ax.tick_params(labelsize=8)

        # compute y-axis limits
        # cover a dynamic range of a few OOM, plus bursts
        ax.set_xlim([0., 13.7])
        ax.legend(loc='best', prop={'size': 4})

        fig.tight_layout()

        fname = '{}_SFHs_{}-{}.png'.format(self.objname, ix[0], ix[1])
        self.savefig(fig, fname, self.figdir, dpi=300)

    def make_all_sfhs_fig(self, ix=None, fig=None, ax=None, ax_loc=111,
                          massnorm='mstar', mass_abs=False, mass_band='i'):
        '''
        make figure with histogram of all sfhs, weighted
        '''
        if ix is None:
            ix = self.ifu_ctr_ix

        w_spax = self.w[:, ix[0], ix[1]]

        w_spax_norm = w_spax / w_spax.max()

        # create figure and axis if necessary
        if not (ax is None):
            pass
        elif (fig is None):
            fig = plt.figure(figsize=(3, 2), dpi=300)
            ax = fig.add_subplot(ax_loc)
        elif (ax is None):
            ax = fig.add_subplot(ax_loc)

        nspecperfile = self.pca.nsfhperfile * self.pca.nsubpersfh
        plchldr_ixs = nspecperfile * \
                      np.arange(0, len(self.pca.sfh_fnames), dtype=int)

        _, allsfrs, _ = zip(
            *map(lambda i: csp.retrieve_SFHs(
                filelist=self.pca.sfh_fnames, i=i, massnorm=massnorm,
                nsubpersfh=self.pca.nsubpersfh, nsfhperfile=self.pca.nsfhperfile),
                plchldr_ixs))
        allsfrs = np.row_stack(allsfrs)

        if (massnorm == 'mstar') and mass_abs:
            ml = self.pca.metadata['ML{}'.format(mass_band)]
            if ml.meta.get('scale', 'linear') == 'log':
                lum = np.log10(self.lum(mass_band))[ix[0], ix[1]]
                m = 10.**(ml + lum)
            else:
                lum = self.lum(mass_band)[ix[0], ix[1]]
                m = ml * lum
            allsfrs *= (m[:, None] / 1.0e9)

        ts = fits.getdata(self.pca.sfh_fnames[0], 'allts')
        allts = np.repeat(ts[None, :], allsfrs.shape[0], axis=0)
        wts = np.repeat(w_spax[:, None], len(ts), axis=1)

        sfrpctls = np.column_stack(
            [ut.weighted_pctls_single(
                 a=allsfrs[:, i], w=wts[:, i], qtls=[16., 50., 84.])
             for i in range(allsfrs.shape[1])])

        ax.fill_between(ts, sfrpctls[0, :], sfrpctls[2, :], color='k', alpha=.5)
        # 50th pctl
        ax.plot(ts, sfrpctls[1, :], color='k')

        rangemax = 4. * np.nanmax(sfrpctls[1, :])
        rangemin = 1.0e-4 * rangemax

        tbins = np.linspace(0., 13.71, 101)
        sfrbins = np.concatenate([np.array([-rangemin]),
                                  np.linspace(rangemin, rangemax, 25)])
        _hist, *_ = np.histogram2d(
            x=allts.flatten(), y=allsfrs.flatten(), weights=wts.flatten(),
            bins=[tbins, sfrbins], normed=True)
        ax.hist2d(x=allts.flatten(), y=allsfrs.flatten(),
                  weights=wts.flatten(), vmax=_hist[:, 1:].max(),
                  bins=[tbins, sfrbins], normed=True)

        if self.truth is not None:
            if (massnorm == 'mstar') and mass_abs:
                if ml.meta.get('scale', 'linear') == 'log':
                    mtruth = 10.**(self.truth['ML{}'.format(mass_band)] + lum)
                else:
                    mtruth = lum * self.truth['ML{}'.format(mass_band)]
                ax.plot(ts, self.truth_sfh * mtruth / 1.0e9, color='r')
            else:
                ax.plot(ts, self.truth_sfh, color='r')

        ax.set_xlabel('time [Gyr]', size=8)
        if mass_abs:
            ax.set_ylabel(r'SFR $[\frac{M_{\odot}}{\rm yr}]$', size=8)
        else:
            ax.set_ylabel('Normed SFR', size=8)

        ax.tick_params(labelsize=8)
        ax.set_title('All SFHs', size=6)

        fig.tight_layout()

        fname = '{}_allSFHs_{}-{}.png'.format(self.objname, ix[0], ix[1])
        self.savefig(fig, fname, self.figdir, dpi=300)

    @property
    def wcs_header(self):
        return wcs.WCS(self.dered.drp_hdulist['RIMG'].header)

    @property
    def wcs_header_offset(self):
        return figures_tools.linear_offset_coordinates(
            self.wcs_header, coord.SkyCoord(
                *(self.wcs_header.wcs.crval * u.deg)))

    @property
    def dist(self):
        return gal_dist(self.cosmo, self.z)

    @property
    def lamticks(self):
        return mticker.MaxNLocator(nbins=8, integer=True, steps=[1, 2, 5, 10])

    @lru_cache(maxsize=16)
    def pctls_16_50_84_(self, qty):
        '''
        caches result of external call to pca.param_pctl_map
        '''
        return self.pca.param_pct_map(
            qty, P=[16., 50., 84.], W=self.w,
            mask=np.logical_or(self.mask_map, ~self.fit_success))

    def param_cred_intvl(self, qty, factor=None, add=None):
        '''
        wraps around caching method to
        '''

        P = self.pctls_16_50_84_(qty)

        if factor is None:
            factor = np.ones_like(P)

        if add is None:
            add = np.zeros_like(P)

        # get scale for qty, default to linear
        scale = self.pca.metadata[qty].meta.get('scale', 'linear')

        if scale == 'log':
            P += np.log10(factor)
        else:
            P *= factor

        # get uncertainty increase
        unc_incr = self.pca.metadata[qty].meta.get('unc_incr', 0.)

        P16, P50, P84 = tuple(map(np.squeeze, np.split(P, 3, axis=0)))
        if scale == 'log':
            l_unc, u_unc = (np.abs(P50 - P16) + unc_incr,
                            np.abs(P84 - P50) + unc_incr)
        else:
            l_unc, u_unc = (np.abs(P50 - P16) + unc_incr,
                            np.abs(P84 - P50) + unc_incr)

        return P50, l_unc, u_unc, scale

    def write_results(self, qtys='important', pc_info=True):

        # initialize FITS hdulist
        # PrimaryHDU is identical to DRP 0th HDU
        hdulist = fits.HDUList([self.dered.drp_hdulist[0]])

        if qtys == 'all':
            qtys = self.pca.metadata.colnames
        elif qtys == 'important':
            qtys = self.pca.important_params
        elif qtys == 'important+':
            qtys = self.pca.importantplus_params
        elif qtys == 'confident':
            qtys = self.pca.confident_params

        for qty in qtys:
            try:
                # retrieve results
                P50, l_unc, u_unc, scale = self.param_cred_intvl(qty=qty)
                qty_hdu = fits.ImageHDU(np.stack([P50, l_unc, u_unc]))
                qty_hdu.header['LOGSCALE'] = (scale == 'log')
                qty_hdu.header['CHANNEL0'] = 'median'
                qty_hdu.header['CHANNEL1'] = 'lower uncertainty'
                qty_hdu.header['CHANNEL2'] = 'upper uncertainty'
                qty_hdu.header['QTYNAME'] = qty
                qty_hdu.header['EXTNAME'] = qty

                # if ground-truth is available, list it
                if self.truth is not None:
                    qty_hdu.header['TRUTH'] = self.truth[qty]
            except ValueError:
                raise ValueError(
                    'Something wrong with output {} (true value: {})'.format(
                        qty, self.truth[qty]))
            else:
                hdulist.append(qty_hdu)

        # luminosity HDU
        lum_hdu = fits.ImageHDU(np.log10(self.lum(band='i')))
        lum_hdu.header['EXTNAME'] = 'LOG_LUM_I'
        hdulist.append(lum_hdu)

        # make extension with median spectral SNR
        snr_hdu = fits.ImageHDU(self.SNR_med)
        snr_hdu.header['EXTNAME'] = 'SNRMED'
        hdulist.append(snr_hdu)

        # make extensions with mask (True denotes bad fit)
        mask_hdu = fits.ImageHDU(self.mask_map.astype(float))
        mask_hdu.header['EXTNAME'] = 'MASK'
        hdulist.append(mask_hdu)

        # make extension with PDF population statistics
        fracs = np.array([.01, .05, .1, .25, .5, .9])
        goodpdf_hdu = fits.ImageHDU(np.stack(
            [self.sample_diag(f=f_) / len(self.pca.metadata)
             for f_ in fracs]))
        for fi, f_ in enumerate(fracs):
            goodpdf_hdu.header['FRAC{}'.format(fi)] = f_
        goodpdf_hdu.header['EXTNAME'] = 'GOODFRAC'
        goodpdf_hdu.header['NMODELS'] = len(self.pca.metadata)
        hdulist.append(goodpdf_hdu)

        # make extension with fit success
        fit_success_hdu = fits.ImageHDU(self.fit_success.astype(float))
        fit_success_hdu.header['EXTNAME'] = 'SUCCESS'
        hdulist.append(fit_success_hdu)

        # make extension with best-fit model index
        bestmodel_hdu = fits.ImageHDU(np.argmax(self.w, axis=0))
        bestmodel_hdu.header['EXTNAME'] = 'MODELNUM'
        hdulist.append(bestmodel_hdu)

        # make extensions with pc amplitudes and normalization array
        if pc_info:
            pc_hdu = fits.ImageHDU(self.A)
            pc_hdu.header['EXTNAME'] = 'CALPHA'
            hdulist.append(pc_hdu)

            norm_hdu = fits.ImageHDU(self.a_map)
            norm_hdu.header['EXTNAME'] = 'NORM'
            hdulist.append(norm_hdu)

        kld_hdu = fits.ImageHDU(self.kullback_leibler())
        kld_hdu.header['EXTNAME'] = 'KLD'
        hdulist.append(kld_hdu)

        fname = os.path.join(self.figdir, '{}_res.fits'.format(self.objname))

        hdulist.writeto(fname, overwrite=True)


def setup_pca(base_dir, base_fname, fname=None,
              redo=True, pkl=True, q=7, nfiles=5, fre_target=.005,
              pca_kwargs={}, makefigs=True):

    if (fname is None) or (not os.path.isfile(fname)) or (redo):
        run_pca = True
    else:
        run_pca = False

    kspec_fname = os.path.join(
        os.environ['STELLARMASS_PCA_DIR'], 'tremonti_cov/manga_covar_matrix.fit')
    # shrink covariance matrix based on
    K_obs = cov_obs.ShrunkenCov.from_tremonti(kspec_fname, shrinkage=.005)

    if run_pca:
        pca = StellarPop_PCA.from_FSPS(
            K_obs=K_obs, base_dir=base_dir,
            nfiles=nfiles, **pca_kwargs)
        if pkl:
            with open(fname, 'wb') as pk_file:
                pickle.dump(pca, pk_file)

    else:
        with open(fname, 'rb') as pk_file:
            pca = pickle.load(pk_file)

    if q == 'auto':
        q_opt = pca.xval_fromfile(
            fname=os.path.join(base_dir, '{}_validation.fits'.format(base_fname)),
            qmax=50, target=fre_target)
        print('Optimal number of PCs:', q_opt)
        pca.run_pca_models(q_opt)
    else:
        pca.run_pca_models(q)

    if run_pca and makefigs:
        pca.make_PCs_fig()
        pca.make_PC_param_regr_fig()
        pca.make_params_vs_PCs_fig()
        pca.make_PC_param_importance_fig()

    return pca, K_obs


def gal_dist(cosmo, z):
    return cosmo.luminosity_distance(z)

def get_col_metadata(col, k, notfound=''):
    '''
    Retrieve a specific metadata keyword `k` from the given column `col`.
        Specify how to behave when the keyword does not exist
    '''

    try:
        res = col.meta[k]
    except KeyError:
        res = notfound

    return res

def setup_fake(row, pca, K_obs, dered_method='nearest', dered_kwargs={},
               mockspec_ix=None, CSPs_dir='.', mockspec_fname='CSPs_test.fits',
               fakedata_basedir='fakedata', mocksfh_fname='SFHs_test.fits',
               pc_cov_method='full_iter', mpl_v='MPL-6', sky=None):

    plateifu = row['plateifu']
    plate, ifu = plateifu.split('-')

    mockspec_fullpath = os.path.join(CSPs_dir, mockspec_fname)
    mocksfh_fullpath = os.path.join(CSPs_dir, mocksfh_fname)

    nsubpersfh = fits.getval(mocksfh_fullpath, ext=0, keyword='NSUBPER')
    nsfhperfile = fits.getval(mocksfh_fullpath, ext=0, keyword='NSFHPER')
    nspecperfile = nsubpersfh * nsfhperfile

    if mockspec_ix is None:
        if nspecperfile == 1:
            mockspec_ix = 0
        else:
            mockspec_ix = np.random.randint(0, nspecperfile - 1)

    # get SFH data from table
    mock_metadata, subsample_entry_ix = csp.retrieve_meta_table(
        filelist=[mockspec_fullpath], i=mockspec_ix,
        nsfhperfile=nsfhperfile, nsubpersfh=nsubpersfh)
    mock_metadata.keep_columns(csp.req_param_keys)
    mock_metadata_row = mock_metadata[mockspec_ix]

    _, mockspec_row, mocksfh_ix, _ = csp.find_sfh_ixs(
        i=mockspec_ix, nsfhperfile=nsfhperfile, nsubpersfh=nsubpersfh)

    _, truth_sfh, _ = csp.retrieve_SFHs(
        [mocksfh_fullpath], i=0, massnorm='mstar')
    truth_sfh = truth_sfh[mocksfh_ix, :]

    data = FakeData.from_FSPS(
        fname=mockspec_fullpath, i=mockspec_ix, plateifu_base=plateifu, pca=pca, row=row,
        K_obs=K_obs, mpl_v=mpl_v, sky=sky, kind=daptype)

    data.write(fakedata_basedir)

    dered = MaNGA_deredshift.from_fakedata(
        plate=int(plate), ifu=int(ifu), MPL_v=mpl_v,
        basedir=fakedata_basedir, row=row, kind=daptype)
    truth_fname = os.path.join(
        fakedata_basedir, '{}_truth.tab'.format(plateifu))
    truth = t.Table.read(truth_fname, format='ascii')[0]

    return dered, data, truth, truth_sfh

def run_object(row, pca, K_obs, force_redo=False, fake=False, redo_fake=False,
               dered_method='nearest', dered_kwargs={}, mockspec_ix=None,
               results_basedir='.', CSPs_basedir='.', mockspec_fname='CSPs_test.fits',
               mocksfh_fname='SFHs_test.fits', vdisp_wt=False,
               pc_cov_method='full_iter', makefigs=True, mpl_v='MPL-7', sky=None):

    plateifu = row['plateifu']

    if (not force_redo) and (os.path.isdir(plateifu)):
        pass
        return

    plate, ifu = plateifu.split('-')

    if fake:
        dered, data, truth, truth_sfh = setup_fake(
            row, pca, K_obs, dered_method=dered_method, dered_kwargs=dered_kwargs,
            mockspec_ix=mockspec_ix, CSPs_dir=CSPs_basedir, fakedata_basedir=results_basedir,
            mockspec_fname=mockspec_fname, mocksfh_fname=mocksfh_fname, mpl_v=mpl_v,
            sky=sky)
        figdir = os.path.join(results_basedir, 'results', plateifu)

    else:
        dered = MaNGA_deredshift.from_plateifu(
            plate=int(plate), ifu=int(ifu), MPL_v=mpl_v, row=row, kind=daptype)
        figdir = os.path.join(results_basedir, 'results', plateifu)
        truth_sfh = None
        truth = None

    z_dist = row['nsa_zdist']

    pca_res = PCA_Result(
        pca=pca, dered=dered, K_obs=K_obs, z=z_dist,
        cosmo=cosmo, figdir=figdir, truth=truth, truth_sfh=truth_sfh,
        dered_method=dered_method, dered_kwargs=dered_kwargs, pc_cov_method=pc_cov_method)
    pca_res.solve(vdisp_wt=vdisp_wt)
    pca_res.reconstruct()

    if makefigs:
        pca_res.make_full_QA_fig(kde=(False, False))
        pca_res.make_sample_diag_fig()
        pca_res.make_qty_fig(qty_str='MLi')

    if fake:
        # delete intermediate files
        drp_fname = os.path.join(
            fakedata_basedir, '{}_drp.fits'.format(pca_res.objname))
        dap_fname = os.path.join(
            fakedata_basedir, '{}_dap.fits'.format(pca_res.objname))
        truth_fname = os.path.join(
            fakedata_basedir, '{}_truth.tab'.format(pca_res.objname))
        for fn in [drp_fname, dap_fname, truth_fname]:
            os.remove(fn)

    return pca_res


def add_bool_arg(parser, name, default, help_string):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true', help=help_string)
    group.add_argument('--no-' + name, dest=name, action='store_false', 
                       help='do not {}'.format(help_string))
    parser.set_defaults(**{name: default})

def run_ew_comparison(drpall, howmany):
    import maskdiag

    index_name = 'H_beta'

    sample = m.shuffle_table(drpall)[:howmany]

    with ProgressBar(howmany) as bar:
        for i, row in enumerate(sample):
            plateifu = row['plateifu']

            try:
                with catch_warnings():
                    simplefilter(warn_behav)
                    pca_res_orig = run_object(
                        basedir=CSPs_dir, row=row, pca=pca, force_redo=True,
                        fake=False, vdisp_wt=False, dered_method=dered_method,
                        dered_kwargs=dered_kwargs, pc_cov_method=pc_cov_method,
                        K_obs=K_obs, mpl_v=mpl_v, makefigs=False)

                    pca_res_redo_w = maskdiag.redo_pca_analysis(
                        pca_res_orig, replace_type='wtdmean')
                    pca_res_redo_z = maskdiag.redo_pca_analysis(
                        pca_res_orig, replace_type='zero')

                    redo_w_tab_ = maskdiag.index_vs_dap_qtys(
                        pca_res_orig, pca_res_redo_w, index_name)
                    redo_z_tab_ = maskdiag.index_vs_dap_qtys(
                        pca_res_orig, pca_res_redo_z, index_name)

            except Exception:
                exc_info = sys.exc_info()
                print('ERROR: {}'.format(plateifu))
                print_exception(*exc_info)
                continue
            else:
                if i == 0:
                    redo_w_tab, redo_z_tab = redo_w_tab_, redo_z_tab_
                else:
                    redo_w_tab = t.vstack([redo_w_tab, redo_w_tab_])
                    redo_z_tab = t.vstack([redo_z_tab, redo_z_tab_])
            finally:
                bar.update()
                redo_w_tab.write(os.path.join(basedir, 'redo_w.tab'),
                                 format='ascii.ecsv', overwrite=True)
                redo_z_tab.write(os.path.join(basedir, 'redo_z.tab'),
                                 format='ascii.ecsv', overwrite=True)

        return redo_w_tab, redo_z_tab


if __name__ == '__main__':
    import argparse

    #'''
    # parse arguments
    parser = argparse.ArgumentParser(description='Run PCA analysis on MaNGA galaxies')
    parser.add_argument('--csp_basedir', default=csp_basedir, required=False,
                        help='where CSPs live')
    # what types of data to run, and whether to make figs or simply output fit
    add_bool_arg(parser, 'figs', default=True, help_string='make figs')

    add_bool_arg(parser, 'manga', default=True, help_string='run MaNGA galax(y/ies)')
    add_bool_arg(parser, 'clobbermanga', default=False,
                 help_string='re-run MaNGA galax(y/ies) where applicable')
    add_bool_arg(parser, 'ensurenew', default=True, 
                 help_string='ensure all galaxies run are new')
    parser.add_argument('--mangaresultsdest', default=manga_results_basedir, required=False,
                        help='destination for MaNGA results')

    add_bool_arg(parser, 'mock', default=False, help_string='run mock(s)')
    add_bool_arg(parser, 'clobbermock', default=False,
                 help_string='re-run mock(s) where applicable')
    parser.add_argument('--mockresultsdest', default=mocks_results_basedir, required=False,
                        help='destination for mocks results')

    add_bool_arg(parser, 'mockfromresults', default=False,
                 help_string='use results from obs to construct mock')

    rungroup = parser.add_mutually_exclusive_group(required=False)
    rungroup.add_argument('--plateifus', '-p', nargs='+', type=str,
                          help='plateifu designations of galaxies to run')
    rungroup.add_argument('--nrun', '-n', help='number of galaxies to run', type=int)

    argsparsed = parser.parse_args()

    print(argsparsed)
    #'''

    #'''
    howmany = 0
    cosmo = WMAP9
    warn_behav = 'ignore'
    dered_method = 'drizzle'
    dered_kwargs = {'nper': 10}
    pc_cov_method = 'precomp'

    drpall = m.load_drpall(mpl_v, index='plateifu')
    drpall = drpall[drpall['nsa_z'] != -9999]
    drpall = drpall[drpall['ifudesignsize'] > 0.]
    
    if argsparsed.mock or argsparsed.manga:
        lsf = ut.MaNGA_LSF.from_drpall(drpall=drpall, n=2)
        pca_kwargs = {'lllim': 3700. * u.AA, 'lulim': 8800. * u.AA,
                      'lsf': lsf, 'z0_': .04}

        pca_pkl_fname = os.path.join(csp_basedir, 'pca.pkl')
        pca, K_obs = setup_pca(
            fname=pca_pkl_fname, base_dir=argsparsed.csp_basedir, base_fname='CSPs',
            redo=False, pkl=True, q=6, fre_target=.005, nfiles=40,
            pca_kwargs=pca_kwargs, makefigs=True)

        K_obs.precompute_Kpcs(pca.PCs)
        K_obs._init_windows(len(pca.l))

        pca.write_pcs_fits()

    if argsparsed.mock:
        skymodel = SkyContamination.from_mpl_v(mpl_v)

    if argsparsed.plateifus:
        howmany = len(argsparsed.plateifus)
        plateifus = argsparsed.plateifus
    elif argsparsed.nrun:
        howmany = argsparsed.nrun
        plateifus = np.random.permutation(list(drpall['plateifu']))  

    while i < howmany:
        plateifu = plateifus[i]
        row = drpall.loc[plateifu]

        if pca_status.log_file_exists(plateifu):
            if ~argsparsed.ensurenew:
                continue
            else:
                i += 1
        else:
            i += 1


        try:
            with catch_warnings():
                simplefilter(warn_behav)

                if argsparsed.manga:
                    pca_res = run_object(
                        row=row, pca=pca, K_obs=K_obs, force_redo=argsparsed.clobbermanga,
                        fake=False, redo_fake=False, dered_method=dered_method,
                        dered_kwargs=dered_kwargs,
                        results_basedir=argsparsed.mangaresultsdest,
                        CSPs_basedir=csp_basedir, vdisp_wt=False,
                        pc_cov_method=pc_cov_method, mpl_v=mpl_v,
                        makefigs=argsparsed.figs)
                    pca_res.write_results(['MLi'])

                '''
                if argsparsed.mock:
                    pca_res = run_object(
                        row=row, pca=pca, K_obs=K_obs, force_redo=argsparsed.clobbermock,
                        fake=False, redo_fake=argsparsed.clobbermock,
                        dered_method=dered_method, dered_kwargs=dered_kwargs,
                        results_basedir=argsparsed.mockresultsdest,
                        CSPs_basedir=csp_basedir, vdisp_wt=False,
                        pc_cov_method=pc_cov_method, mpl_v=mpl_v,
                        makefigs=argsparsed.figs, sky=skymodel)
                    pca_res_f.write_results('confident')
                '''

        except Exception as e:
            exc_info = sys.exc_info()
            print('ERROR: {}'.format(plateifu))
            print_exception(*exc_info)
            pca_status.write_log_file(plateifu, repr(e))
            continue
        else:
            print('{} completed successfully'.format(plateifu))
            pca_status.write_log_file(plateifu, 'SUCCESS')
        finally:
            pass
    #'''
