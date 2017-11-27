import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib import cm as mplcm
from matplotlib import gridspec
import matplotlib.ticker as mticker
from cycler import cycler

from corner import corner

# astropy ecosystem
from astropy import constants as c, units as u, table as t
from astropy.io import fits
from astropy import wcs
from astropy.utils.console import ProgressBar
from specutils.extinction import reddening
from astropy.cosmology import WMAP9
from astropy import coordinates as coord
from astropy.wcs.utils import pixel_to_skycoord

import os
import sys
from warnings import warn, filterwarnings, catch_warnings, simplefilter
from traceback import print_exception
import multiprocessing as mpc
import ctypes
from functools import lru_cache

# scipy
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.ndimage.filters import gaussian_filter1d

# sklearn
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import sklearn.decomposition as decomp

# statsmodels
from statsmodels.nonparametric.kde import KDEUnivariate

# local
import csp
import cov_obs
import figures_tools
import radial
from spectrophot import (lumspec2lsun, color, C_ML_conv_t as CML,
                         Spec2Phot, absmag_sun_band as Msun)
import utils as ut
import indices
from fakedata import FakeData
from partition import CovCalcPartitioner
from linalg import *

# add manga RC location to path, and import config
if os.environ['MANGA_CONFIG_LOC'] not in sys.path:
    sys.path.append(os.environ['MANGA_CONFIG_LOC'])

import mangarc

if mangarc.tools_loc not in sys.path:
    sys.path.append(mangarc.tools_loc)

# personal
import manga_tools as m
import spec_tools

eps = np.finfo(float).eps


class StellarPop_PCA(object):

    '''
    class for determining PCs of a library of synthetic spectra
    '''

    def __init__(self, l, trn_spectra, gen_dicts, metadata, K_obs, src,
                 sfh_fnames, nsubpersfh, nsfhperfile,
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
        self.gen_dicts = gen_dicts

        self.src = src

        self.sfh_fnames = sfh_fnames

        self.nsubpersfh = nsubpersfh
        self.nsfhperfile = nsfhperfile

        # observational covariance matrix
        if K_obs.__class__ != cov_obs.Cov_Obs:
            raise TypeError('incorrect observational covariance matrix class!')

        if K_obs.dlogl != self.dlogl:
            raise PCAError('non-matching log-lambda spacing ({}, {})'.format(
                           K_obs.dlogl, self.dlogl))

    @classmethod
    def from_FSPS(cls, K_obs, lsf, base_dir, nfiles=None,
                  log_params=['MWA', 'MLr', 'MLi', 'MLz', 'MLV', 'Fstar'],
                  vel_params={}, dlogl=1.0e-4, z0_=.04, **kwargs):
        '''
        Read in FSPS outputs (dicts & metadata + spectra) from some directory
        '''

        from glob import glob
        from utils import pickle_loader, add_losvds
        from itertools import chain

        d_names = glob(os.path.join(base_dir,'CSPs_*.pkl'))
        csp_fnames = glob(os.path.join(base_dir,'CSPs_*.fits'))
        sfh_fnames = glob(os.path.join(base_dir,'SFHs_*.fits'))

        if nfiles is not None:
            d_names = d_names[:nfiles]
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

        meta['Dn4000'] = indices.StellarIndex('Dn4000')(
            l=l.value, flam=spec.T, axis=0).flatten()
        meta['Hdelta_A'] = indices.StellarIndex('Hdelta_A')(
            l=l.value, flam=spec.T, axis=0).flatten()
        meta['Mg_b'] = indices.StellarIndex('Mg_b')(
            l=l.value, flam=spec.T, axis=0).flatten()
        meta['Ca_HK'] = indices.StellarIndex('Ca_HK')(
            l=l.value, flam=spec.T, axis=0).flatten()
        meta['Na_D'] = indices.StellarIndex('Na_D')(
            l=l.value, flam=spec.T, axis=0).flatten()
        meta['tau_V mu'] = meta['tau_V'] * meta['mu']
        meta = meta['sigma', 'logzsol', 'tau_V', 'mu',
                    'tau_V mu', 'theta', 'MWA', 'MLV', 'MLi',
                    'Dn4000', 'Hdelta_A', 'Mg_b',
                    'Ca_HK', 'Na_D', 'mf_1Gyr']

        meta['MWA'].meta['TeX'] = r'MWA'
        meta['Dn4000'].meta['TeX'] = r'D$_{n}$4000'
        meta['Hdelta_A'].meta['TeX'] = r'H$\delta_A$'
        meta['Mg_b'].meta['TeX'] = r'Mg$_b$'
        meta['Ca_HK'].meta['TeX'] = r'CaHK'
        meta['Na_D'].meta['TeX'] = r'Na$_D$'
        meta['logzsol'].meta['TeX'] = r'$\log{\frac{Z}{Z_{\odot}}}$'
        meta['tau_V'].meta['TeX'] = r'$\tau_V$'
        meta['mu'].meta['TeX'] = r'$\mu$'
        meta['tau_V mu'].meta['TeX'] = r'$\tau_V ~ \mu$'
        #meta['MLr'].meta['TeX'] = r'$\Upsilon^*_r$'
        meta['MLi'].meta['TeX'] = r'$\Upsilon^*_i$'
        #meta['MLz'].meta['TeX'] = r'$\Upsilon^*_z$'
        meta['MLV'].meta['TeX'] = r'$\Upsilon^*_V$'
        meta['sigma'].meta['TeX'] = r'$\sigma$'
        meta['mf_1Gyr'].meta['TeX'] = r'$f_m^{\textrm{1~Gyr}}$'
        meta['theta'].meta['TeX'] = r'$\Theta$'

        for n in meta.colnames:
            if n in log_params:
                meta[n] = np.log10(meta[n])
                meta[n].meta['scale'] = 'log'
                if 'ML' in n:
                    meta[n].meta['unc_incr'] = .008

        '''
        # M/L thresholds
        ML_ths = [-10 if meta[n].meta.get('scale', 'linear') == 'log' else 0.
                  for n in ['MLV', 'MLi]]

        # MWA bounds
        if meta['MWA'].meta.get('scale', 'linear') == 'log':
            MWA_bds = [-10, 2] # log Gyr
        else:
            MWA_bds = [0, 14] # Gyr

        models_good = np.all(np.row_stack(
            [(np.array(meta['MLr']) > ML_ths[0]),
             (np.array(meta['MLi']) > ML_ths[1]),
             (np.array(meta['MLz']) > ML_ths[2]),
             (np.array(meta['MWA']) > MWA_bds[0]),
             (np.array(meta['MWA']) < MWA_bds[1])]), axis=0)

        print('not good models:', np.argwhere(~models_good))
        '''

        dicts = list(chain.from_iterable(
            [pickle_loader(f) for (i, f) in enumerate(d_names)]))

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
        spec_lores = spec_lores.astype(np.float32)

        for k in meta.colnames:
            meta[k] = meta[k].astype(np.float32)

        return cls(l=l_final * l.unit, trn_spectra=spec_lores,
                   gen_dicts=dicts, metadata=meta, sfh_fnames=sfh_fnames,
                   K_obs=K_obs, dlogl=None, src='FSPS',
                   nsubpersfh=Nsubsample, nsfhperfile=Nsfhper, **kwargs)

    # =====
    # methods
    # =====

    def xval(self, specs, qmax=30):
        # reconstruction error
        qs = np.arange(1, qmax + 1, 1)
        err = np.empty_like(qs, dtype=float)

        trn = self.trn_spectra

        # normalize mean of each training spectrum to 1
        a = np.mean(trn, axis=1, keepdims=True)
        normed_trn = trn / a

        # find the average spectrum and subtract it from each training spectrum
        M = np.mean(normed_trn, axis=0, keepdims=True)
        S = normed_trn - M

        for i, q in enumerate(qs):
            PCs, PVE = self.PCA(S, q)

            # normalize mean again
            a_ = np.median(specs, axis=1, keepdims=True)
            specs_ = (specs / a_)

            # PC amplitudes
            A = (specs_ - M).dot(PCs.T)
            # reconstructed spectra
            R = a_ * (PCs.T.dot(A.T) + M.T).T

            # fractional reconstruction error
            e = np.mean(np.abs((specs - R) / specs))

            err[i] = e

        return qs, err

    def xval_fromfile(self, fname, qmax=50, target=.01):
        hdulist = fits.open(fname)

        specs_full = hdulist['flam'].data
        l_full = hdulist['lam'].data
        logl_full = np.log10(l_full)

        specs_interp = interp1d(x=logl_full, y=specs_full, kind='linear', axis=-1)
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
        plt.tight_layout()

        # return smallest q value for which target FRE is reached
        q = np.min(qs[err < target])

        hdulist.close()

        fig.savefig('xval_test.png')

        return q

    def run_pca_models(self, q):
        '''
        run PCA on library of model spectra
        '''

        self.scaler = ut.MedianSpecScaler(X=self.trn_spectra)
        self.normed_trn = self.scaler.X_sc
        self.M = np.median(self.normed_trn, axis=0)
        self.S = self.normed_trn - self.M

        R = np.cov(self.S, rowvar=False)
        # calculate evecs & evalse of covariance matrix
        # (use 'eigh' rather than 'eig' since R is symmetric for performance
        evals, evecs = np.linalg.eigh(R)
        # sort eigenvalues and eigenvectors in decreasing order
        idx = np.argsort(evals)[::-1]
        # and select first `q`
        evals, evecs = evals[idx], evecs[:,idx]
        self.PCs = evecs[:, :q].T
        self.PCs /= np.sign(self.PCs.sum(axis=1, keepdims=True))
        # carry out the transformation on the data using eigenvectors
        self.trn_PC_wts = np.dot(self.S, self.PCs.T)

        # reconstruct the best approximation for the spectra from PCs
        self.trn_recon = np.dot(self.trn_PC_wts, self.PCs)
        # calculate covariance using reconstruction residuals
        self.trn_resid = self.normed_trn - self.trn_recon

        # percent variance explained
        self.PVE = (evals / evals.sum())[:q]

        self.cov_th = np.cov(self.trn_resid, rowvar=False)

    def project_cube(self, f, ivar, mask_spax=None, mask_spec=None,
                     mask_cube=None):
        '''
        project real spectra onto principal components, given a
            flux array & inverse-variance array, plus optional
            additional masks

        params:
         - f: flux array (should be regridded to rest, need not be normalized)
            with shape (nl, m, m) (shape of IFU datacube)
         - ivar: inverse-variance (same shape as f)
         - dl: width of wavelength bins (nspec-length array)
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
            ivar *= (~mask_spax).astype(float)
        if mask_spec is not None:
            ivar *= (~mask_spec[:, np.newaxis, np.newaxis]).astype(float)
        if mask_cube is not None:
            ivar *= (~mask_cube).astype(float)

        # account for everywhere-zero ivar
        ivar[ivar == 0.] = eps

        # run through same scaling normalization as training data
        O_norm, a = self.scaler(f)
        ivar_wt = ivar * a**2.
        O_sub = O_norm - self.M[:, None, None]

        A = robust_project_onto_PCs(e=self.PCs, f=O_sub, w=ivar_wt)

        return A, self.M, a, O_sub, O_norm

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

        hdulist.writeto('pc_vecs.fits', overwrite=True)

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

    def build_PC_cov_partitioned(self, z_map, K_obs_, a_map, ivar_cube,
                                 snr_map):
        '''
        build covariance matrix using partitioned linear-algebraic method
        '''
        print('partitioned')

        E = self.PCs
        q, l = E.shape
        mapshape = z_map.shape

        # build an array to hold covariances
        K_PC = 100. * np.ones((q, q) + mapshape)

        # compute starting indices for covariance matrix
        cov_logl = K_obs_.logl
        i0_map = self._compute_i0_map(cov_logl=cov_logl, z_map=z_map)

        covcalc = CovCalcPartitioner(
            kspec_obs=K_obs_.cov, a_map=a_map, i0_map=i0_map,
            E=E, ivar_scaled=ivar_cube)

        K_PC = covcalc.calc_allchunks()
        # set places with bad SNRs to high variance
        np.moveaxis(K_PC, (0, 1, 2, 3), (2, 3, 0, 1))[snr_map < 1] = 10.
        # add theoretical component
        K_PC += (E @ self.cov_th @ E.T)[..., None, None]

        return K_PC

    def build_PC_cov_medix(self, z_map, K_obs_, a_map, ivar_cube,
                           snr_map):
        '''
        build covariance matrix assuming same starting index throughout cube
        '''
        print('medix')

        E = self.PCs
        q, nl = E.shape
        mapshape = z_map.shape

        # compute starting indices for covariance matrix
        cov_logl = K_obs_.logl
        i0_map = self._compute_i0_map(cov_logl=cov_logl, z_map=z_map)

        # fiducial starting index
        i00 = np.round(np.median(i0_map), 0).astype(int)
        # fiducial covariance given starting index
        kpc00 = E @ K_obs_.cov[i00:i00 + nl, i00:i00 + nl] @ E.T

        # propagate ivars into a separate cube
        k_ivar = np.einsum('li,ijk,mi->lmjk',
                           E, np.nan_to_num(1. / ivar_cube), E)

        return (kpc00[..., None, None] * a_map**2.) + k_ivar

    def build_PC_cov_ivaronly(self, z_map, K_obs_, a_map, ivar_cube,
                              snr_map):
        '''
        build PC covariance matrix assuming only main-diagonal
            contribution from ivar
        '''
        print('ivaronly')
        E = self.PCs
        k_ivar = np.einsum('li,ijk,mi->lmjk',
                           E, np.nan_to_num(1. / ivar_cube), E)

        return k_ivar

    def build_PC_cov_full_iter(self, z_map, K_obs_, a_map, ivar_cube,
                               snr_map):
        '''
        for each spaxel, build a PC covariance matrix, and return in array
            of shape (q, q, n, n)

        NOTE: this technique computes each spaxel's covariance matrix
            separately, not all at once! Simultaneous computation requires
            too much memory!

        params:
         - z_map: spaxel redshift map (get this from an dered instance)
         - obs_logl: log-lambda vector (1D) of datacube
         - K_obs_: observational spectral covariance object
        '''
        print('full_iter')

        E = self.PCs
        q, l = E.shape
        mapshape = z_map.shape

        # build an array to hold covariances
        K_PC = 100. * np.ones((q, q) + mapshape)

        cov_logl = K_obs_.logl

        # compute starting indices for covariance matrix
        i0_map = self._compute_i0_map(cov_logl=cov_logl, z_map=z_map)
        # bring ivar cube to same SNR as data

        inds = np.ndindex(mapshape)  # iterator over spatial dims of cube

        k_pc_gen = ut.KPCGen(
            kspec_obs=K_obs_.cov, i0_map=i0_map, E=E, ivar_scaled=ivar_cube,
            a_map=a_map)

        for ind in inds:
            # handle bad (SNR < 1) spaxels
            if snr_map[ind[0], ind[1]] < .1:
                pass
            else:
                K_PC[..., ind[0], ind[1]] = k_pc_gen(ind[0], ind[1])

        K_PC += (E @ self.cov_th @ E.T)[..., None, None]

        return K_PC

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

        chi2 = np.einsum('cixy,ijxy,cjxy->cxy', D, P, D)
        w = np.exp(-0.5 * chi2)

        return w

    def param_pct_map(self, qty, W, P, factor=None, add=None):
        '''
        This is iteration based, which is not awesome.

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

        inds = np.ndindex(*cubeshape)

        A = np.empty((len(P),) + cubeshape)

        for ind in inds:
            q = Q
            w = W[:, ind[0], ind[1]]
            i_ = np.argsort(q, axis=0)
            q, w = q[i_], w[i_]
            A[:, ind[0], ind[1]] = np.interp(
                P, 100. * w.cumsum() / w.sum(), q)

        return (A + add[None, ...]) * factor[None, ...]

    def param_cred_intvl(self, qty, W, factor=None):
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
        P = self.param_pct_map(qty=qty, W=W, P=P, factor=factor, add=add)

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
                pcnum = 'Mean'
            else:
                pcnum = 'PC{}'.format(i)
            ax.set_ylabel(pcnum, size=6)

            loc = mticker.MaxNLocator(nbins=5, prune='upper')
            ax.yaxis.set_major_locator(loc)

            if i != q:
                ax.tick_params(axis='x', labelbottom='off')
            else:
                ax.tick_params(axis='x', color='k', labelsize=8)

            ax.tick_params(axis='y', color='k', labelsize=6)

        # use last axis to give wavelength
        ax.set_xlabel(r'$\lambda~[\textrm{\AA}]$')
        plt.suptitle('Eigenspectra')

        fig.savefig('PCs_{}.png'.format(self.src), dpi=300)

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
            ax.tick_params(axis='x', labelbottom='off')
            ax.tick_params(axis='y', labelleft='off')
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
            ax.tick_params(axis='x', labelbottom='off')
            yloc = mticker.MaxNLocator(nbins=5, prune='upper')
            # tick labels on RHS of hists
            ax.yaxis.set_major_locator(yloc)
            ax.tick_params(axis='y', labelleft='off', labelright='on',
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
                ax.tick_params(axis='x', labelbottom='off')
            else:
                xloc = mticker.MaxNLocator(nbins=5, prune='upper')
                ax.xaxis.set_major_locator(xloc)
                ax.tick_params(axis='x', labelsize=6)
                ax.set_xlabel('PC{}'.format(j + 1), size=8)

            if j != 0:
                ax.tick_params(axis='y', labelleft='off')
            else:
                yloc = mticker.MaxNLocator(nbins=5, prune='upper')
                ax.yaxis.set_major_locator(yloc)
                ax.tick_params(axis='y', labelsize=6)
                ax.set_ylabel(self.metadata_TeX[i], size=8)

        fig.suptitle('PCs vs params')

        plt.savefig('PCs_params_{}.png'.format(self.src), dpi=300)

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

        X = np.stack([np.linalg.lstsq(a=a_, b=b_[:, i])[0]
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

        fig.savefig('param_regr_{}.png'.format(self.src), dpi=300)

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
            TeX = self.metadata[k].meta['TeX']
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
        plt.savefig('PC_param_importance_{}.png'.format(self.src), dpi=300)

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


class MaNGA_deredshift(object):
    '''
    class to deredshift reduced MaNGA data based on velocity info from DAP

    preserves cube information, in general

    also builds in a check on velocity coverage, and computes a mask
    '''

    spaxel_side = 0.5 * u.arcsec

    def __init__(self, drp_hdulist, dap_hdulist, drpall_row,
                 max_vel_unc=500. * u.Unit('km/s'), drp_dlogl=None,
                 MPL_v='MPL-5'):
        self.drp_hdulist = drp_hdulist
        self.dap_hdulist = dap_hdulist
        self.drpall_row = drpall_row
        self.plateifu = self.drp_hdulist[0].header['PLATEIFU']

        self.vel = dap_hdulist['STELLAR_VEL'].data * u.Unit('km/s')
        self.vel_ivar = dap_hdulist['STELLAR_VEL_IVAR'].data * u.Unit(
            'km-2s2')

        self.z = drpall_row['nsa_z']

        # mask all the spaxels that have high stellar velocity uncertainty
        self.vel_ivar_mask = (1. / np.sqrt(self.vel_ivar)) > max_vel_unc
        self.vel_mask = m.mask_from_maskbits(
            self.dap_hdulist['STELLAR_VEL_MASK'].data, [30])

        self.drp_l = drp_hdulist['WAVE'].data
        self.drp_logl = np.log10(self.drp_l)
        if drp_dlogl is None:
            drp_dlogl = spec_tools.determine_dlogl(self.drp_logl)
        self.drp_dlogl = drp_dlogl

        flux_hdu, ivar_hdu, l_hdu = (drp_hdulist['FLUX'], drp_hdulist['IVAR'],
                                     drp_hdulist['WAVE'])

        self.flux = flux_hdu.data
        self.ivar = ivar_hdu.data

        self.units = {'l': u.AA, 'flux': u.Unit('1e-17 erg s-1 cm-2 AA-1')}

        self.S2P = Spec2Phot(lam=(self.drp_l * self.units['l']),
                             flam=(self.flux * self.units['flux']))

        self.DONOTUSE = m.mask_from_maskbits(drp_hdulist['MASK'].data, [10])

        self.ivar *= ~self.DONOTUSE

    @classmethod
    def from_plateifu(cls, plate, ifu, MPL_v, kind='SPX-GAU-MILESHC', row=None,
                      **kwargs):
        '''
        load a MaNGA galaxy from a plateifu specification
        '''

        plate, ifu = str(plate), str(ifu)

        if row is None:
            drpall = m.load_drpall(MPL_v, index='plateifu')
            row = drpall.loc['{}-{}'.format(plate, ifu)]

        drp_hdulist = m.load_drp_logcube(plate, ifu, MPL_v)
        dap_hdulist = m.load_dap_maps(plate, ifu, MPL_v, kind)
        return cls(drp_hdulist, dap_hdulist, row, **kwargs)

    @classmethod
    def from_fakedata(cls, plate, ifu, MPL_v, basedir='fakedata', row=None,
                      kind='SPX-GAU-MILESHC', **kwargs):
        '''
        load fake data based on a particular already-observed galaxy
        '''

        plate, ifu = str(plate), str(ifu)

        if row is None:
            drpall = m.load_drpall(MPL_v, index='plateifu')
            row = drpall.loc['{}-{}'.format(plate, ifu)]

        drp_hdulist = fits.open(
            os.path.join(basedir, '{}-{}_drp.fits'.format(plate, ifu)))
        dap_hdulist = fits.open(
            os.path.join(basedir, '{}-{}_dap.fits'.format(plate, ifu)))

        return cls(drp_hdulist, dap_hdulist, row, **kwargs)

    def correct_and_match(self, template_logl, template_dlogl=None,
                          method='supersample', dered_kwargs={}):
        '''
        gets datacube ready for PCA analysis:
            - take out galactic extinction
            - compute per-spaxel redshifts
            - deredshift observed spectra
            - raise alarms where templates don't cover enough l range
            - return subarrays of flam, ivar

        (this does not perform any fancy interpolation, just "shifting")
        (nor are emission line features masked--that must be done in post-)
        '''
        if template_dlogl is None:
            template_dlogl = spec_tools.determine_dlogl(template_logl)

        if template_dlogl != self.drp_dlogl:
            raise csp.TemplateCoverageError(
                'template and input spectra must have same dlogl: ' +
                'template\'s is {}; input spectra\'s is {}'.format(
                    template_dlogl, self.drp_dlogl))

        # correct for MW extinction
        r_v = 3.1
        EBV = self.drp_hdulist[0].header['EBVGAL']
        f_mwcorr, ivar_mwcorr = ut.extinction_correct(
            l=self.drp_l * u.AA, f=self.flux,
            ivar=self.ivar, r_v=r_v, EBV=EBV)

        # prepare to de-redshift
        # total redshift of each spaxel
        z_map = (1. + self.z) * (1. + (self.vel / c.c).to('').value) - 1.
        z_map[self.vel_mask] = self.z
        self.z_map = z_map

        # shift into restframe
        l_rest, f_rest, ivar_rest = ut.redshift(
            l=self.drp_l, f=f_mwcorr, ivar=ivar_mwcorr, z_in=z_map, z_out=0.)
        # this maintains constant dlogl

        # and make photometric object to reflect rest-frame spectroscopy
        ctr = [i // 2 for i in z_map.shape]
        # approximate rest wavelength of whole cube as rest wavelength
        # of central spaxel
        l_rest_ctr = l_rest[:, ctr[0], ctr[1]]
        self.S2P_rest = Spec2Phot(lam=(l_rest_ctr * self.units['l']),
                                  flam=(f_rest * self.units['flux']))

        self.regrid = ut.Regridder(
            loglgrid=template_logl, loglrest=np.log10(l_rest),
            frest=f_rest, ivarfrest=ivar_rest, dlogl=template_dlogl)

        # cal appropriate regridder method
        self.flux_regr, self.ivar_regr = getattr(
            self.regrid, method)(**dered_kwargs)

        self.spax_mask = np.logical_or.reduce((
            self.vel_mask, self.vel_ivar_mask))

        return self.flux_regr, self.ivar_regr, self.spax_mask

    def compute_eline_mask(self, template_logl, template_dlogl=None, ix_eline=7,
                           half_dv=150. * u.Unit('km/s')):

        from elines import (balmer_low, balmer_high, helium,
                            bright_metal, faint_metal)

        if template_dlogl is None:
            template_dlogl = spec_tools.determine_dlogl(template_logl)

        EW = self.eline_EW(ix=ix_eline)
        # thresholds are set very aggressively for debugging, but should be
        # revisited in the future
        # proposed values... balmer_low: 0, balmer_high: 2, helium: 10
        #                    brightmetal: 0, faintmetal: 10, paschen: 10
        add_balmer_low = (EW >= 0. * u.AA)
        add_balmer_high = (EW >= 2. * u.AA)
        add_helium = (EW >= 10. * u.AA)
        add_brightmetal = (EW >= 0. * u.AA)
        add_faintmetal = (EW >= 10. * u.AA)

        temlogl = template_logl
        teml = 10.**temlogl

        full_mask = np.zeros((len(temlogl),) + EW.shape, dtype=bool)

        for (add_, d) in zip([add_balmer_low, add_balmer_high, add_helium,
                              add_brightmetal, add_faintmetal],
                             [balmer_low, balmer_high,
                              helium, bright_metal, faint_metal]):

            l_el = spec_tools.air2vac(np.array(list(d.values())), u.AA).value
            logl_el = np.log10(l_el)

            # compute mask edges
            hdz = (half_dv / c.c).to('').value

            # use speclite's redshift function
            rules = [dict(name='l', exponent=+1, array_in=l_el)]
            mask_ledges = ut.slrs(rules=rules, z_in=0., z_out=-hdz)['l']
            mask_uedges = ut.slrs(rules=rules, z_in=0., z_out=hdz)['l']

            # is a given wavelength bin within half_dv of a line center?
            mask = np.row_stack(
                [(lo < teml) * (teml < up)
                 for lo, up in zip(mask_ledges, mask_uedges)])
            mask = np.any(mask, axis=0)  # OR along axis 0

            full_mask += (mask[:, np.newaxis, np.newaxis] *
                          add_[np.newaxis, ...])

        full_mask = full_mask > 0.

        return full_mask

    def eline_EW(self, ix):
        return self.dap_hdulist['EMLINE_SEW'].data[ix] * u.Unit('AA')

    def coadd(self, tem_l, good=None):
        '''
        return coadded spectrum and ivar

        params:
         - good: map of good spaxels
        '''

        if good is None:
            good = np.ones_like(self.flux_regr[0, ...])

        ivar = self.ivar_regr * good[None, ...]

        flux, ivar = ut.coadd(f=self.flux_regr, ivar=ivar)
        lam, flux, ivar = (tem_l[:, None, None], flux[:, None, None],
                           ivar[:, None, None])

        return lam, flux, ivar

    # =====
    # properties
    # =====

    @property
    def SB_map(self):
        # RIMG gives nMgy/pix
        return self.drp_hdulist['RIMG'].data * \
            1.0e-9 * m.Mgy / self.spaxel_side**2.

    @property
    def Reff(self):
        r_ang = self.dap_hdulist['SPX_ELLCOO'].data[0, ...]
        Re_ang = self.drpall_row['nsa_elpetro_th50_r']
        return r_ang / Re_ang

    # =====
    # staticmethods
    # =====

    @staticmethod
    def a_map(f, logl, dlogl):
        lllims = 10.**(logl - 0.5 * dlogl)
        lulims = 10.**(logl + 0.5 * dlogl)
        dl = (lulims - lllims)[:, np.newaxis, np.newaxis]
        return np.mean(f * dl, axis=0)


cmap = mplcm.get_cmap('cubehelix')
cmap.set_bad('gray')
cmap.set_under('k')
cmap.set_over('w')

class HistFailedWarning(UserWarning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PCA_Result(object):

    '''
    store results of PCA for one galaxy using this
    '''

    def __init__(self, pca, dered, K_obs, z, cosmo, figdir='.',
                 truth=None, truth_sfh=None, dered_method='nearest',
                 dered_kwargs={}, vdisp_wt=False, pc_cov_method='full_iter',
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

        self.O, self.ivar, mask_spax = dered.correct_and_match(
            template_logl=pca.logl, template_dlogl=self.pca.dlogl,
            method=dered_method, dered_kwargs=dered_kwargs)
        self.mask_cube = dered.compute_eline_mask(
            template_logl=pca.logl, template_dlogl=self.pca.dlogl,
            half_dv=200. * u.km / u.s)

        self.mask_cube = np.logical_or(self.mask_cube, self.ivar == 0.)

        self.SNR_med = np.median(self.O * np.sqrt(self.ivar) + eps,
                                 axis=0)

        self.A, self.M, self.a_map, self.O_sub, self.O_norm = pca.project_cube(
            f=self.O, ivar=self.ivar, mask_spax=mask_spax,
            mask_cube=self.mask_cube)

        # original spectrum
        self.O = np.ma.array(self.O, mask=self.mask_cube)
        self.ivar = np.ma.array(self.ivar * self.a_map**2., mask=self.mask_cube)
        self.O_norm = np.ma.array(self.O_norm, mask=self.mask_cube)

        # how to reconstruct datacube from PC weights cube
        self.O_recon = np.ma.array(pca.reconstruct_normed(self.A),
                                   mask=self.mask_cube)

        self.resid = (self.O_norm - self.O_recon) / self.O_norm

        pc_cov_f = getattr(pca, '_'.join(('build_PC_cov', pc_cov_method)))
        self.pc_cov_method = pc_cov_method
        self.K_PC = pc_cov_f(z_map=dered.z_map, K_obs_=self.K_obs,
            a_map=self.a_map, ivar_cube=self.ivar,
            snr_map=self.SNR_med)

        # increase covariance to account for systematics in model library
        cov_sys = cov_sys_incr * np.diag(np.ones_like(self.pca.l))
        self.K_PC += (self.E @ cov_sys @ self.E.T)[..., None, None]

        try:
            self.P_PC = P_from_K_pinv(self.K_PC)
        except np.linalg.LinAlgError:
            self.P_PC = P_from_K(self.K_PC)

        self.map_shape = self.O.shape[-2:]
        self.ifu_ctr_ix = [s // 2 for s in self.map_shape]

        self.w = pca.compute_model_weights(P=self.P_PC, A=self.A)

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

        # no data
        self.nodata = (dered.drp_hdulist['RIMG'].data == 0.)

        self.mask_map = np.logical_or.reduce(
            (mask_spax, self.nodata))

        # spaxel is bad if < 25 models have weights 1/100 max, and no other problems
        self.badPDF = np.logical_and.reduce(
            ((self.sample_diag(f=.01) < 10), ~self.mask_map))
        self.goodPDF = ~self.badPDF

        self.l = 10.**self.pca.logl

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

        return im, cb

    def comp_plot(self, ax1, ax2, ix=None):
        '''
        make plot illustrating fidelity of PCA decomposition in reproducing
            observed data
        '''

        if ix is None:
            ix = self.ifu_ctr_ix

        # best fitting spectrum
        bestfit = self.pca.normed_trn[np.argmax(self.w[:, ix[0], ix[1]]), :]
        bestfit_ = ax1.plot(self.l, bestfit, drawstyle='steps-mid',
                            c='c', label='Best Trn.', linewidth=0.5, zorder=0)

        # original & reconstructed
        O_norm = self.O_norm[:, ix[0], ix[1]]
        O_recon = self.O_recon[:, ix[0], ix[1]]
        ivar = self.ivar[:, ix[0], ix[1]]

        Onm = np.ma.median(O_norm)
        Orm = np.ma.median(O_recon)
        ivm = np.ma.median(ivar)

        orig_ = ax1.plot(self.l, O_norm, drawstyle='steps-mid',
                         c='b', label='Obs.', linewidth=0.5, zorder=1)
        recon_ = ax1.plot(self.l, O_recon, drawstyle='steps-mid',
                          c='g', label='PCA Fit', linewidth=0.5, zorder=2)
        ax1.axhline(y=0., xmin=self.l.min(), xmax=self.l.max(),
                    c='k', linestyle=':')

        # inverse-variance (weight) plot
        ivar_ = ax1.plot(
            self.l, ivar / ivm, drawstyle='steps-mid', c='m',
            label='IVAR', linewidth=0.5, zorder=0)

        # residual plot
        resid = self.resid[:, ix[0], ix[1]]
        std_err = (1. / np.sqrt(ivar))
        fit_resid_ = ax2.plot(
            self.l.data, resid, drawstyle='steps-mid', c='green',
            linewidth=0.5, alpha=.5)
        model_resid = ax2.plot(
            self.l.data, bestfit - O_norm, drawstyle='steps-mid', c='cyan',
            linewidth=0.5, alpha=.5)
        conf_band_ = ax2.fill_between(
            x=self.l.data, y1=-std_err, y2=std_err,
            linestyle='--', color='salmon', linewidth=0.25, zorder=0)

        ax1.tick_params(axis='y', which='major', labelsize=10,
                        labelbottom='off')
        ax2.tick_params(axis='both', which='major', labelsize=10)

        ax1.xaxis.set_major_locator(self.lamticks)
        ax1.xaxis.set_ticklabels([])
        ax1.legend(loc='best', prop={'size': 6})
        ax1.set_ylabel(r'$F_{\lambda}$ (rel)')
        ax1.set_ylim([-0.1 * Onm, 2.25 * Onm])
        ax1.set_yticks(np.arange(0.0, ax1.get_ylim()[1], 0.5))

        ax2.xaxis.set_major_locator(self.lamticks)
        ax2.set_xlabel(r'$\lambda$ [$\textrm{\AA}$]')
        ax2.set_ylim([-.75, .75])
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
            qty=qty_str, factor=f, W=self.w)

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
            med_TeX = ''.join((r'$\log$', med_TeX))
        elif (scale == 'log'):
            unc = (u_unc + l_unc) / 2.
            med_TeX = ''.join((r'$\log$', med_TeX))
        else:
            unc = (l_unc + u_unc) / 2.

        m = ax1.imshow(
            np.ma.array(P50, mask=self.mask_map),
            aspect='equal', norm=norm[0])

        s = ax2.imshow(
            np.ma.array(unc, mask=self.mask_map),
            aspect='equal', norm=norm[1])

        mcb = plt.colorbar(m, ax=ax1, pad=0.025)
        mcb.set_label(med_TeX, size=8)
        mcb.ax.tick_params(labelsize=8)

        scb = plt.colorbar(s, ax=ax2, pad=0.025)
        scb.set_label(r'$\sigma$', size=8)
        scb.ax.tick_params(labelsize=8)

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
            qty=qty_str, factor=f, W=self.w)

        if scale == 'log':
            return 10.**P50

        return P50

    def Mstar_integrated(self, band='i'):
        '''
        calculate integrated spectrum, and then compute stellar mass from that
        '''

        _, O, ivar = self.dered.coadd(tem_l=self.pca.l, good=~self.mask_map)

        # in the spaxels that were used to coadd, OR-function
        # of mask, over cube
        mask_cube = np.any(self.mask_cube * (~self.mask_map[None, ...]),
                           axis=(1, 2)).astype(bool)[..., None, None]

        A, M, a_map, O_sub, O_norm = pca.project_cube(
            f=O, ivar=ivar, mask_spax=None,
            mask_cube=mask_cube)

        # original spectrum
        O = np.ma.array(O, mask=mask_cube)
        ivar = np.ma.array(ivar, mask=mask_cube)
        O_norm = np.ma.array(O_norm, mask=mask_cube)

        # how to reconstruct datacube from PC weights cube
        O_recon = np.ma.array(pca.reconstruct_normed(A),
                              mask=mask_cube)

        resid = np.abs((O_norm - O_recon) / O_norm)

        # redshift is flux-averaged
        z_map = np.average(self.dered.z_map, weights=self.O.sum(axis=0))
        SNR = np.nanmedian(O.data * np.sqrt(ivar.data))
        z_map, SNR, a_map = (np.atleast_2d(z_map), np.atleast_2d(SNR),
                             np.atleast_2d(a_map))

        pc_cov_f = getattr(pca, '_'.join(('build_PC_cov', pc_cov_method)))
        K_PC = pc_cov_f(z_map=dered.z_map, K_obs_=self.K_obs,
            a_map=self.a_map, ivar_cube=self.ivar,
            snr_map=self.SNR_med)

        P_PC = P_from_K_pinv(K_PC)

        w = pca.compute_model_weights(P=P_PC, A=A)

        lum = np.ma.masked_invalid(self.lum(band=band))

        # this SHOULD and DOES call the method in PCA rather than
        # in self, since we aren't using self.w
        P50, *_, scale = self.pca.param_cred_intvl(
            qty='ML{}'.format(band), factor=lum.sum(), W=w)

        if scale == 'log':
            return 10.**P50

        return P50

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

        TeX_over = r'$M^*_{{{}}}$'.format(band)

        m, s, mcb, scb, scale = self.qty_map(
            ax1=ax1, ax2=ax2, qty_str=qty, f=f, norm=[None, None],
            logify=logify, TeX_over=TeX_over)

        logmstar_tot = np.log10(np.ma.masked_invalid(np.ma.array(
            self.Mstar_tot(band=band), mask=self.mask_map)).sum())

        logmstar_allspec = np.log10(self.Mstar_integrated(band))

        try:
            TeX1 = ''.join((r'$\log{\frac{M_{*}}{M_{\odot}}}$ = ',
                            '{:.2f}'.format(logmstar_tot)))
        except TypeError:
            TeX1 = 'ERROR'

        try:
            TeX2 = ''.join((r'$\log{\frac{M_{*,add}}{M_{\odot}}}$ = ',
                            '{:.2f}'.format(logmstar_allspec)))
        except TypeError:
            TeX2 = 'ERROR'

        ax1xlims, ax1ylims = ax1.get_xlim(), ax1.get_ylim()

        ax1.text(x=tr((0, 1), ax1xlims, 0.05),
                 y=tr((0, 1), ax1ylims, 0.05), s=TeX1, color='r')
        ax1.text(x=tr((0, 1), ax1xlims, 0.5),
                 y=tr((0, 1), ax1ylims, 0.05), s=TeX2, color='r')

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

        TeX = self.pca.metadata[qty].meta['TeX']

        scale = self.pca.metadata[qty].meta.get('scale')
        if scale == 'log':
            TeX = ''.join((r'$\log$', TeX))

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
                ev_ax_.plot(qgrid, log_ev, color='g', linestyle='--',
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

        # over ax objects
        for ax in axs:
            #ax.set_xlabel(r'$\alpha$')
            #ax.set_ylabel(r'$\delta$')

            ax.coords[0].set_ticks(spacing=5. * u.arcsec)
            ax.coords[0].set_format_unit(u.arcsec)

            ax.coords[1].set_ticks(spacing=5. * u.arcsec)
            ax.coords[1].set_format_unit(u.arcsec)

            if bad:
                # figures_tools.annotate_badPDF(ax, self.goodPDF)
                pass

    def __setup_qty_fig__(self):
        fig = plt.figure(figsize=(9, 4), dpi=300)

        gs = gridspec.GridSpec(1, 2, wspace=.175, left=.075, right=.975,
                               bottom=.11, top=.9)
        ax1 = fig.add_subplot(gs[0], projection=self.wcs_header)
        ax2 = fig.add_subplot(gs[1], projection=self.wcs_header)

        # overplot hatches for masks
        # start by defining I & J pixel grid
        II, JJ = np.meshgrid(*(np.linspace(-.5, ms_ - .5, ms_ + 1)
                               for ms_ in self.map_shape))
        IIc, JJc = map(lambda x: 0.5 * (x[:-1, :-1] + x[1:, 1:]), (II, JJ))

        for ax in [ax1, ax2]:
            # badpdf mask
            ax.pcolor(II, JJ,
                      np.ma.array(np.zeros_like(IIc), mask=~self.badPDF),
                      hatch='\\'*8, alpha=0.)
            # dered mask
            ax.pcolor(II, JJ,
                      np.ma.array(np.zeros_like(IIc), mask=~self.mask_map),
                      hatch='/'*8, alpha=0.)

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

        pix_coord = self.wcs_header_offset.all_pix2world(
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

        fig_height = 15
        fig_width = 12

        nparams = len(self.pca.metadata.colnames)
        ncols = 3
        nrows = nparams // ncols + (nparams % ncols != 0)

        plt.close('all')

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)

        # gridspec used for map + spec_compare
        gs1 = gridspec.GridSpec(
            3, 4, bottom=(nrows - 1.) / nrows, top=0.95,
            height_ratios=[3, 1, 1], width_ratios=[2, 0.5, 2, 2],
            hspace=0., wspace=.1, left=.075, right=.95)

        gs2 = gridspec.GridSpec(
            nrows, ncols, bottom=.05, top=(nrows - 1.) / nrows,
            left=.05, right=.95, hspace=.25)

        # put the spectrum and residual here!
        spec_ax = fig.add_subplot(gs1[0, 2:])
        resid_ax = fig.add_subplot(gs1[1, 2:])
        orig_, recon_, bestfit_, ivar_, resid_, resid_avg_, ix_ = \
            self.comp_plot(ax1=spec_ax, ax2=resid_ax, ix=ix)

        TeX_labels = [get_col_metadata(self.pca.metadata[n], 'TeX', n)
                      for n in self.pca.metadata.colnames]

        # image of galaxy in integrated light
        im_ax = fig.add_subplot(gs1[:-1, 0],
                                projection=self.wcs_header_offset)
        lumim, lcb = self.lum_plot(im_ax, ix=ix_, band='r')
        # self.map_add_loc(ix=ix_, ax=im_ax, color='gray', linewidth=1.,
        #                  linestyle=':')

        # loop through parameters of interest, and make a weighted
        # histogram for each parameter
        enum_ = enumerate(zip(gs2, self.pca.metadata.colnames))
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

    def radial_gp_plot(self, qty, TeX_over=None, f=None, ax=None,
                       q_bdy=None, logify=False):
        '''
        make a radial plot of a quantity + uncertainties using GP regression
        '''

        if ax is None:
            ax = plt.gca()

        q, l_unc, u_unc, scale = self.pca.param_cred_intvl(
            qty=qty, factor=f, W=self.w)

        q_unc = np.abs(l_unc + u_unc) / 2.

        if not TeX_over:
            qty_tex = self.pca.metadata[qty].meta.get('TeX', qty)
        else:
            qty_tex = TeX_over

        if scale == 'log':
            qty_tex = ''.join((r'$\log$', qty_tex))

        # throw out spaxels at large Re
        # in future, should evaluate spectrum uncertainties directly
        rlarge = self.dered.Reff > 3.5
        r = np.ma.array(self.dered.Reff,
                        mask=(rlarge | self.mask_map | self.badPDF))

        try:
            # radial gaussian process from sklearn (v0.18 or later)
            gp = radial.radial_gp(r=r, q=q, q_unc=q_unc, q_bdy=q_bdy,
                                  scale=scale)
        except radial.GPFitError:
            # sometimes it fails when solution space is too sparse
            print('GP regr. failed: {}'.format(qty))
        except:
            raise
        else:
            r_pred = np.atleast_2d(np.linspace(0., r.max(), 100)).T
            q_pred, sigma2 = gp.predict(r_pred, return_std=True)
            if scale == 'log':
                (q_pred, sigma) = (np.log10(q_pred),
                                   np.log10(q_pred + np.sqrt(sigma2)) - \
                                       np.log10(q_pred))
            else:
                sigma = np.sqrt(sigma2)
            # plot allowed range
            ax.plot(r_pred, q_pred, c='b', label='Prediction')
            ax.fill(np.concatenate([r_pred, r_pred[::-1]]),
                    np.concatenate([(q_pred - 1.9600 * sigma),
                                    (q_pred + 1.9600 * sigma)[::-1]]),
                    alpha=.3, facecolor='b', edgecolor='None', label='95\% CI')

        # plot data
        sorter = np.argsort(r.flatten())
        ax.errorbar(x=r.flatten()[sorter], y=q.flatten()[sorter],
                    yerr=np.row_stack([l_unc.flatten()[sorter],
                                       u_unc.flatten()[sorter]]),
                    label='PCA Results', linestyle='None', marker='o',
                    markersize=2, c='k', alpha=0.2, capsize=1.5,
                    markevery=10, errorevery=10)

        ax.legend(loc='best', prop={'size': 6})
        ax.set_xlabel(r'$\frac{R}{R_e}$')
        ax.set_ylabel(qty_tex)

        ax.set_ylim([-1., 1.5])

        return ax

    def make_radial_gp_fig(self, qty,TeX_over=None, q_bdy=[-np.inf, np.inf]):
        fig = plt.figure(figsize=(4, 4), dpi=300)

        ax = fig.add_subplot(111)

        self.radial_gp_plot(qty=qty, TeX_over=None, ax=ax,
                            q_bdy=q_bdy)
        ax.set_title(self.objname)
        plt.tight_layout()

        fname = '{}-{}_radGP.png'.format(self.objname, qty)
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

    def cornerplot(self, ix=None):
        '''
        make spaxel parameter corner plot
        '''

        if ix is None:
            ix = self.ifu_ctr_ix

        w_spax = self.w[:, ix[0], ix[1]]

        if self.truth is not None:
            truth = [self.truth[n] for n in self.pca.metadata.colnames]
        else:
            truth = None

        fig = corner(xs=self.pca.metadata_a, weights=w_spax,
                     labels=self.pca.metadata_TeX,
                     truths=truth, truth_color='r')

        fig.tight_layout()

        fname = '{}_cornerplot_{}-{}.png'.format(self.objname, ix[0], ix[1])
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
        return self.pca.param_pct_map(qty, P=[16., 50., 84.], W=self.w)

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

    def write_results(self, qtys='all', pc_info=False):

        # initialize FITS hdulist
        # PrimaryHDU is identical to DRP 0th HDU
        hdulist = fits.HDUList([self.dered.drp_hdulist[0]])

        if qtys == 'all':
            qtys = self.pca.metadata.colnames

        for qty in qtys:
            # retrieve results
            P50, l_unc, u_unc, scale = self.pca.param_cred_intvl(qty=qty, W=self.w)
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

            hdulist.append(qty_hdu)

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
        hdulist.append(goodpdf_hdu)

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

        fname = os.path.join(self.figdir, '{}_res.fits'.format(self.objname))

        hdulist.writeto(fname, overwrite=True)


def setup_pca(base_dir, base_fname, fname=None,
              redo=True, pkl=True, q=7, nfiles=5, fre_target=.005,
              pca_kwargs={}):
    import pickle
    if (fname is None) or (not os.path.isfile(fname)) or (redo):
        run_pca = True
    else:
        run_pca = False

    kspec_fname = 'manga_Kspec_norm_smooth101.fits'

    if run_pca:
        K_obs = cov_obs.Cov_Obs.from_fits(kspec_fname)
        #K_obs = cov_obs.Cov_Obs.from_YMC_BOSS(
        #    'cov_obs_YMC_BOSS.fits')
        pca = StellarPop_PCA.from_FSPS(
            K_obs=K_obs, base_dir=base_dir,
            nfiles=nfiles, **pca_kwargs)
        if pkl:
            pickle.dump(pca, open(fname, 'wb'))

    else:
        K_obs = cov_obs.Cov_Obs.from_fits(kspec_fname)
        pca = pickle.load(open(fname, 'rb'))

    if q == 'auto':
        q_opt = pca.xval_fromfile(
            fname=os.path.join(base_dir, '{}_validation.fits'.format(base_fname)),
            qmax=50, target=fre_target)
        print('Optimal number of PCs:', q_opt)
        pca.run_pca_models(q_opt)
    else:
        pca.run_pca_models(q)

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

def run_object(row, pca, force_redo=False, fake=False, redo_fake=False,
               dered_method='nearest', dered_kwargs={}, mockspec_ix=None, z_new=None,
               CSPs_dir='.', mockspec_fname='CSPs_test.fits',
               mocksfh_fname='SFHs_test.fits', vdisp_wt=False,
               pc_cov_method='full_iter', makefigs=True, mpl_v='MPL-5'):

    plateifu = row['plateifu']

    if (not force_redo) and (os.path.isdir(plateifu)):
        return

    plate, ifu = plateifu.split('-')

    if fake:
        if redo_fake:
            # allow for custom new redshift
            if not z_new:
                plateifu_base = plateifu
            else:
                z_orig, zdist_orig = row['nsa_z'], row['nsa_zdist']
                row['nsa_z'] = row['nsa_zdist'] = z_new
                plateifu_base = '-'.join((plateifu, 'z{}'.format(z_new)))

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
                fname=mockspec_fullpath, i=mockspec_ix,
                plateifu_base=plateifu_base, pca=pca, row=row)

            data.write()

        dered = MaNGA_deredshift.from_fakedata(
            plate=int(plate), ifu=int(ifu), MPL_v=mpl_v,
            basedir='fakedata', row=row)
        figdir = os.path.join('fakedata', 'results', plateifu_base)
        truth_fname = os.path.join(
            'fakedata', '{}_truth.tab'.format(plateifu_base))
        truth = t.Table.read(truth_fname, format='ascii')[0]
    else:
        plateifu_base = plateifu
        dered = MaNGA_deredshift.from_plateifu(
            plate=int(plate), ifu=int(ifu), MPL_v=mpl_v, row=row)
        figdir = os.path.join('results', plateifu_base)
        mock_metadata_row = None
        truth_sfh = None
        truth = None

    z_dist = row['nsa_zdist']

    pca_res = PCA_Result(
        pca=pca, dered=dered, K_obs=K_obs, z=z_dist,
        cosmo=cosmo, figdir=figdir,
        truth=truth, truth_sfh=truth_sfh, vdisp_wt=vdisp_wt,
        dered_method=dered_method, dered_kwargs=dered_kwargs,
        pc_cov_method=pc_cov_method)

    if makefigs:
        pca_res.make_full_QA_fig(kde=(False, False))

        #pca_res.cornerplot()

        pca_res.make_sample_diag_fig()
        #pca_res.make_top_sfhs_fig()
        pca_res.make_all_sfhs_fig(massnorm='mstar', mass_abs=True)

        #pca_res.make_qty_fig(qty_str='MLr')
        pca_res.make_qty_fig(qty_str='MLi')
        #pca_res.make_qty_fig(qty_str='MLz')
        #pca_res.make_qty_fig(qty_str='MLV')

        #pca_res.make_qty_fig(qty_str='MWA')

        #pca_res.make_Mstar_fig(band='r')
        pca_res.make_Mstar_fig(band='i')
        #pca_res.make_Mstar_fig(band='z')

        #pca_res.make_radial_gp_fig(qty='MLr', q_bdy=[.01, 100.])
        #pca_res.make_radial_gp_fig(qty='MLi', q_bdy=[.01, 100.])
        #pca_res.make_radial_gp_fig(qty='MLz', q_bdy=[.01, 100.])
        #pca_res.make_radial_gp_fig(qty='MLV', q_bdy=[.01, 100.])

        #pca_res.make_qty_fig('Dn4000')

        #pca_res.make_color_ML_fig(mlb='i', b1='g', b2='i', colorby='R')
        #pca_res.make_color_ML_fig(mlb='i', b1='g', b2='i', colorby='tau_V mu')

        #pca_res.compare_sigma()

        #pca_res.sigma_vel()

    # reset redshift to original value
    if not z_new:
        pass
    else:
        row['nsa_z'], row['nsa_zdist'] = z_orig, zdist_orig

    return pca_res

if __name__ == '__main__':
    howmany = 10
    cosmo = WMAP9
    warn_behav = 'ignore'
    dered_method = 'nearest'
    dered_kwargs = {'nper': 10}
    pc_cov_method = 'medix'

    CSPs_dir = '/usr/data/minhas2/zpace/CSPs/CSPs_CKC14_MaNGA_20171114-1/'

    mpl_v = 'MPL-5'

    drpall = m.load_drpall(mpl_v, index='plateifu')
    drpall = drpall[drpall['nsa_z'] != -9999]
    lsf = ut.MaNGA_LSF.from_drpall(drpall=drpall, n=2)
    pca_kwargs = {'lllim': 3700. * u.AA, 'lulim': 8800. * u.AA,
                  'lsf': lsf, 'z0_': .04}

    pca_pkl_fname = os.path.join(CSPs_dir, 'pca.pkl')
    pca, K_obs = setup_pca(
        fname=pca_pkl_fname, base_dir=CSPs_dir, base_fname='CSPs',
        redo=True, pkl=True, q=10, fre_target=.005, nfiles=None,
        pca_kwargs=pca_kwargs)

    pca.write_pcs_fits()

    #'''
    row = drpall.loc['8464-1901']

    with catch_warnings():
        simplefilter(warn_behav)

        pca_res = run_object(
            row=row, pca=pca, force_redo=False, fake=False, vdisp_wt=True,
            dered_method=dered_method, dered_kwargs=dered_kwargs,
            pc_cov_method=pc_cov_method)
        pca_res.write_results(pc_info=True)

        pca_res_f = run_object(
            row=row, pca=pca, force_redo=True, fake=True,
            redo_fake=True, mockspec_ix=0, dered_method=dered_method,
            dered_kwargs=dered_kwargs, CSPs_dir=CSPs_dir, vdisp_wt=True,
            mockspec_fname='TestSpecs-5.10.fits', mocksfh_fname='TestSFH-5.10.fits')
        pca_res_f.write_results()
    #'''

    '''
    #drpall = drpall[drpall['ifudesignsize'] > 0.]
    #drpall = drpall[drpall['ifudesignsize'] == 127]
    #drpall = drpall[drpall['nsa_elpetro_ba'] >= 0.4]

    mwolf = t.Table.read('mwolf/gmrt_targets.csv')
    drpall = drpall[[r['plateifu'] in mwolf['plateifu'] for r in drpall]]

    drpall = m.shuffle_table(drpall)
    #drpall = drpall[:howmany]

    with ProgressBar(len(drpall)) as bar:
        for i, row in enumerate(drpall):
            plateifu = row['plateifu']

            try:
                with catch_warnings():
                    simplefilter(warn_behav)

                    pca_res = run_object(
                        row=row, pca=pca, force_redo=False, fake=False, vdisp_wt=True,
                        dered_method=dered_method, dered_kwargs=dered_kwargs)
                    pca_res.write_results()

                    pca_res_f = run_object(
                        row=row, pca=pca, force_redo=True, fake=True,
                        redo_fake=True, mockspec_ix=None, dered_method=dered_method,
                        dered_kwargs=dered_kwargs, CSPs_dir=CSPs_dir, vdisp_wt=True,
                        mockspec_fname='CSPs_0.fits', mocksfh_fname='SFHs_0.fits')
                    pca_res_f.write_results()

            except Exception:
                exc_info = sys.exc_info()
                print('ERROR: {}'.format(plateifu))
                print_exception(*exc_info)
                continue
            finally:
                bar.update()
    '''
