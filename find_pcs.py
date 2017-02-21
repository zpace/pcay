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
from specutils.extinction import reddening
from astropy.cosmology import WMAP9
from astropy import coordinates as coord

import os
import sys
from warnings import warn, filterwarnings, catch_warnings, simplefilter
import multiprocessing as mpc
import ctypes

# scipy
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.integrate import quad

# sklearn
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

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
        self.metadata_TeX = [m for m in metadata_TeX if m is not False]

        # a kludgey conversion from structured array to regular array
        metadata_a = np.array(self.metadata)
        metadata_a = metadata_a.view((metadata_a.dtype[0],
                                      len(metadata_a.dtype.names)))
        self.metadata_a = metadata_a[:, metadata_incl]
        self.gen_dicts = gen_dicts

        self.src = src

        # observational covariance matrix
        if K_obs.__class__ != cov_obs.Cov_Obs:
            raise TypeError('incorrect observational covariance matrix class!')

        if K_obs.dlogl != self.dlogl:
            raise PCAError('non-matching log-lambda spacing ({}, {})'.format(
                           K_obs.dlogl, self.dlogl))

    @classmethod
    def from_YMC(cls, base_dir, lib_para_file, form_file,
                 spec_file_base, K_obs):
        '''
        initialize object from CSP library provided by Y-M Chen, which is
            based on a BC03 model library
        '''

        # first figure out full paths
        form_file_full = os.path.join(base_dir, form_file)
        lib_para_file_full = os.path.join(base_dir, lib_para_file)

        # get wavelength grid, and prepare for interpolation of model
        test_spec_fname = os.path.join(
            base_dir, '{}_1.fits'.format(spec_file_base))
        hdulist = fits.open(test_spec_fname)
        l_raw = hdulist[2].data
        hdulist.close()

        l_raw_good = (3700. <= l_raw) * (l_raw <= 11000.)
        l_raw = l_raw[l_raw_good]
        dlogl_final = 1.0e-4

        l_final = 10.**np.arange(
            np.log10(3700.), np.log10(5500.), dlogl_final)

        l_full = 10.**np.arange(np.log10(3700.), np.log10(10500.),
                                dlogl_final)

        nl_full = len(l_full)

        # load metadata tables
        lib_para = t.Table.read(lib_para_file_full, format='ascii')

        nspec = len(lib_para)
        form_data = t.Table.read(form_file_full, format='ascii')
        form_data_goodcols = ['zmet', 'Tau_v', 'mu']

        for n in form_data_goodcols:
            lib_para.add_column(
                t.Column(data=np.zeros(len(lib_para)), name=n))

        # initialize array that resampled spectra will go into
        spec = np.empty((nspec, nl_full))
        goodspec = np.ones(nspec).astype(bool)
        # and fill it
        for i in range(nspec):
            # create a hypothetical filename
            fname = os.path.join(
                base_dir, '{}_{}.fits'.format(spec_file_base, i))

            # handle file-DNE case exception-less-ly
            if not os.path.exists(fname):
                goodspec[i] = False
                continue

            try:
                hdulist = fits.open(fname)
            # if there's another issue...
            except IOError:
                # print 'bad!', fname
                goodspec[i] = False
            else:
                f_lambda = hdulist[3].data[l_raw_good]
                spec[i, :] = interp1d(l_raw, f_lambda)(l_full)
                for n in form_data_goodcols:
                    lib_para[i - 1][n] = float(form_data[i][n])

            finally:
                hdulist.close()

        metadata = lib_para

        ixs = np.arange(nspec)
        metadata.remove_rows(ixs[~goodspec])
        spec = spec[goodspec, :]
        pca_spec = spec[:, :len(l_final)]

        # compute mass to light ratio
        # convert to Lnu and integrate over bandpass

        L_r = lumspec2lsun(lam=l_full * u.AA, Llam=spec * u.Unit('Lsun/AA'),
                           band='r')
        L_i = lumspec2lsun(lam=l_full * u.AA, Llam=spec * u.Unit('Lsun/AA'),
                           band='i')
        L_z = lumspec2lsun(lam=l_full * u.AA, Llam=spec * u.Unit('Lsun/AA'),
                           band='z')

        MLr = metadata['cspm_star'] / L_r
        MLi = metadata['cspm_star'] / L_i
        MLz = metadata['cspm_star'] / L_z

        metadata['Fstar'] = (metadata['mfb_1e9'] + metadata['mf_1e9'].astype(
            float)) / metadata['mf_all']

        metadata['Dn4000'] = spec_tools.Dn4000_index(
            l=l_full, s=spec.T[..., None]).flatten()
        metadata['Hdelta_A'] = spec_tools.Hdelta_A_index(
            l=l_full, s=spec.T[..., None]).flatten()

        metadata = metadata['MWA', 'Dn4000', 'Hdelta_A', 'Fstar',
                            'zmet', 'Tau_v', 'mu']
        metadata.add_column(t.Column(data=MLr, name='MLr'))
        metadata.add_column(t.Column(data=MLi, name='MLi'))
        metadata.add_column(t.Column(data=MLz, name='MLz'))

        # set metadata to enable plotting later
        metadata['MWA'].meta['TeX'] = r'$\textrm{MWA [Gyr]}$'
        metadata['Dn4000'].meta['TeX'] = r'D$_{n}$4000'
        metadata['Hdelta_A'].meta['TeX'] = r'H$\delta_A$'
        metadata['Fstar'].meta['TeX'] = r'$F^*$'
        metadata['zmet'].meta['TeX'] = r'$\log{\frac{Z}{Z_{\odot}}}$'
        metadata['Tau_v'].meta['TeX'] = r'$\tau_V$'
        metadata['mu'].meta['TeX'] = r'$\mu$'
        metadata['MLr'].meta['TeX'] = r'$\Upsilon^*_r$'
        metadata['MLi'].meta['TeX'] = r'$\Upsilon^*_i$'
        metadata['MLz'].meta['TeX'] = r'$\Upsilon^*_z$'

        return cls(l=l_final * u.AA, trn_spectra=pca_spec,
                   gen_dicts=None, metadata=metadata, dlogl=dlogl_final,
                   K_obs=K_obs, src='YMC')

    @classmethod
    def from_FSPS(cls, K_obs, base_dir='CSPs', base_fname='CSPs', nfiles=None,
                  log_params=['MWA', 'MLr', 'MLi', 'MLz', 'Fstar'],
                  vel_params={}, dlogl=1.0e-4, **kwargs):
        '''
        Read in FSPS outputs (dicts & metadata + spectra) from some directory
        '''

        from glob import glob
        from utils import pickle_loader, add_losvds
        from itertools import chain

        d_names = glob(os.path.join(base_dir,'{}_*.pkl'.format(base_fname)))
        f_names = glob(os.path.join(base_dir,'{}_*.fits'.format(base_fname)))

        # filter out test data
        istest = ['test' in n for n in d_names]
        d_names = [n for n, t in zip(d_names, istest) if not t]
        f_names = [n for n, t in zip(f_names, istest) if not t]

        if nfiles is not None:
            d_names = d_names[:nfiles]
            f_names = f_names[:nfiles]

        hdulists = [fits.open(f) for f in f_names]

        l = hdulists[0][2].data * u.AA
        logl = np.log10(l.value)

        *_, specs = zip(*hdulists)

        meta = t.vstack([t.Table.read(f, format='fits', hdu=1)
                         for f in f_names])
        spec = np.row_stack([s.data for s in specs])

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
        meta = meta['MWA', 'sigma','logzsol',
                    'MLr', 'MLi', 'MLz',
                    'tau_V', 'mu', 'tau_V mu',
                    'Dn4000', 'Hdelta_A', 'Mg_b',
                    'Ca_HK', 'Na_D']

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
        meta['MLr'].meta['TeX'] = r'$\Upsilon^*_r$'
        meta['MLi'].meta['TeX'] = r'$\Upsilon^*_i$'
        meta['MLz'].meta['TeX'] = r'$\Upsilon^*_z$'
        meta['sigma'].meta['TeX'] = r'$\sigma$'

        for n in meta.colnames:
            if n in log_params:
                meta[n] = np.log10(meta[n])
                meta[n].meta['scale'] = 'log'
                if 'ML' in n:
                    meta[n].meta['unc_incr'] = .008

        # M/L thresholds
        ML_ths = [-10 if meta[n].meta.get('scale', 'linear') == 'log' else 0.
                  for n in ['MLr', 'MLi', 'MLz']]

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

        dicts = chain.from_iterable(
            [pickle_loader(f) for (i, f) in enumerate(d_names)
             if models_good[i]])

        spec, meta = spec[models_good, :], meta[models_good]

        # interpolate models to desired l range
        logl_final = np.arange(np.log10(l.value.min()),
                               np.log10(l.value.max()), dlogl)
        l_final = 10.**logl_final
        specs_interp = interp1d(x=logl, y=spec, kind='linear', axis=-1)
        spec = specs_interp(logl_final)

        spec /= spec.max(axis=1)[..., None]
        spec = spec.astype(np.float32)

        for k in meta.colnames:
            meta[k] = meta[k].astype(np.float32)

        return cls(l=l_final * l.unit, trn_spectra=spec, gen_dicts=dicts, metadata=meta,
                   K_obs=K_obs, dlogl=None, src='FSPS', **kwargs)

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
            a_ = np.mean(specs, axis=1, keepdims=True)
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

        fig.savefig('xval_test.png')

        return q

    def run_pca_models(self, q):
        '''
        run PCA on library of model spectra
        '''

        # first run some prep on the model spectra

        # S = (T / a) - M
        # T = a * (M + S)
        # normalize each spectrum by its avg flux density
        self.a = np.mean(self.trn_spectra, axis=1)

        self.normed_trn = self.trn_spectra / self.a[:, np.newaxis]
        self.M = np.mean(self.normed_trn, axis=0)
        self.S = self.normed_trn - self.M

        self.PCs, self.PVE = self.PCA(self.S, dims=q)

        # project back onto the PCs to get the weight vectors
        self.trn_PC_wts = (self.S).dot(self.PCs.T)
        # and reconstruct the best approximation for the spectra from PCs
        self.trn_recon = self.trn_PC_wts.dot(self.PCs)
        # residuals
        self.trn_resid = self.S - self.trn_recon

        self.cov_th = np.cov(self.trn_resid.T)

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

        assert ivar.shape == f.shape, 'cube shapes must be equal'
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

        # normalize by average flux density
        a = np.average(f, weights=ivar, axis=0)
        a[a == 0.] = np.mean(a[a != 0.])
        f = f / a

        # get mean spectrum
        O_sub = f - self.M[..., None, None]

        # need to do some reshaping
        # make f and ivar effectively a list of spectra

        A = self.robust_project_onto_PCs(e=self.PCs, f=O_sub, w=ivar)

        return A, self.M, a, O_sub, f

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

    def compute_spec_cov_full(self, snr_map, z_map, K_obs_):
        '''
        DO NOT USE!! Array output too large, raises MemoryError
        Additionally, this is not up-to-date with the iterative method,
            in that it does not use the MaNGA covariance matrix to set
            the scale of the full matrix

        compute a full PC covariance matrix for each spaxel

        params:
         - a_map: mean flux-density of original spectra (n by n)
         - z_map: spaxel redshift map (get this from an dered instance)
         - obs_logl: log-lambda vector (1D) of datacube
         - K_obs_: observational spectral covariance

        all params with _map as part of the name have to be n by n arrays,
            with the same shape as the last two dimensions of a datacube

        returns:
         - K: array giving the spectral covariance for each spaxel

        shape of K: (nl, nl, n, n) -- that is, each spaxel gets a
            covariance matrix, and final two dimensions have the same
            shape as the final two dimensions of a datacube

        Full cov matrix is q x q for each spaxel

        This raises MemoryError as float64
        '''

        q, nl = self.E.shape

        cov_logl = K_obs_.logl
        I0 = self._compute_i0_map(z_map=z_map, cov_logl=cov_logl)

        nl_ = np.arange(nl)[..., None, None]
        I, J = np.meshgrid(nl_, nl_)

        K_s = KK_l_obs[I0[..., None, None] + J, I0[..., None, None] + I]
        K_s = K_s / a_map[..., None, None] + k_l_th[None, None, ...]

        K_PC = E @ K_s @ E.T
        K_PC = np.moveaxis(K_PC, [0, 1], [-2, -1])

    def build_PC_cov_full_iter(self, z_map, K_obs_, snr_map):
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

        E = self.PCs
        q, l = E.shape
        mapshape = z_map.shape

        # build an array to hold covariances
        K_PC = 10. * np.ones((q, q) + mapshape)

        cov_logl = K_obs_.logl

        # compute starting indices for covariance matrix
        i0_map = self._compute_i0_map(cov_logl=cov_logl, z_map=z_map)

        inds = np.ndindex(mapshape)  # iterator over spatial dims of cube
        nl = self.cov_th.shape[0]

        isnr = 1. / snr_map

        for ind in inds:
            # handle bad (SNR < 1) spaxels
            if snr_map[ind[0], ind[1]] < 1.:
                pass
            else:
                i0_ = i0_map[ind[0], ind[1]]
                sl = [slice(i0_, i0_ + nl) for _ in range(2)]
                K_spec = K_obs_.cov[sl]

                K_PC[..., ind[0], ind[1]] = (E @ K_spec @ E.T)

        K_PC *= isnr[None, None, ...]
        K_PC += (E @ self.cov_th @ E.T)[..., None, None]

        return K_PC

    def build_PC_cov_full_mpc(self, z_map, K_obs_, snr_map, nproc=8):
        '''
        use multiprocessing to build PC cov matrix
        '''

        q, l = self.PCs.shape
        mapshape = z_map.shape

        # build an array to hold covariances
        K_PC = 10. * np.ones((q, q) + mapshape)

        cov_logl = K_obs_.logl

        # compute starting indices for covariance matrix
        i0_map = self._compute_i0_map(cov_logl=cov_logl, z_map=z_map)

        inds = np.ndindex(mapshape)  # iterator over spatial dims of cube
        nl = self.cov_th.shape[0]

        isnr = 1. / snr_map

        def args():
            for ix in inds:
                yield (K_obs.cov, snr_map[ix[0], ix[1]],
                       i0_map[ix[0], ix[1]], self.PCs, nl, q)

        pool = mpc.Pool(processes=nproc)
        K_PC = pool.starmap(ut.PC_cov, args)
        K_PC = np.moveaxis(np.reshape(K_PC, mapshape + (q, q)), [0, 1], [2, 3])

        K_PC *= isnr[None, None, ...]
        K_PC += (E @ self.cov_th @ E.T)[..., None, None]

        return K_PC

    def compute_model_weights(self, P, A, norm='L2', soft=False):
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
        nmodels, q, NX, NY = D.shape
        Xhalf, Yhalf = NX // 2, NY // 2

        if norm == 'L2':
            # this is really mahalanobis distance
            chi2 = np.einsum('cixy,ijxy,cjxy->cxy', D, P, D)
        elif norm == 'L1':
            P_ = np.moveaxis(np.diagonal(P), [0, 1, 2], [1, 2, 0])
            chi2 = (D * P_ * D).sum(axis=1)
        elif norm == 'cos':
            C_norm = np.sqrt((C**2.).sum(axis=1))[..., None, None]
            A_norm = np.sqrt((A**2.).sum(axis=0))[None, ...]
            chi2 = (np.sqrt(np.einsum('cixy,ijxy,cjxy->cxy', D, P, D)) /
                    (A_norm * C_norm))

        if soft:
            f = C.shape[1]
        else:
            f = 1.

        w = np.exp(-chi2 / (2. * f))

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
            ax.text(x=3550., y=np.mean(PCs[i, :]), s=pcnum, size=6)

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
                ahist(self.trn_PC_wts[:, i], bins=50, ax=ax,
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
    # staticmethods
    # =====

    @staticmethod
    def robust_project_onto_PCs(e, f, w=None):
        '''
        project a set of measured spectra with measurement errors onto
            the principal components of the model library

        params:
         - e: (q, l) array of eigenvectors
         - f: (l, x, y) array, where n is the number of spectra, and m
            is the number of spectral wavelength bins. [Flux-density units]
         - w: (l, x, y) array, containing the inverse-variances for each
            of the rows in spec [Flux-density units]
            (this functions like the array w_lam in REF [1])

        REFERENCE:
            [1] Connolly & Szalay (1999, AJ, 117, 2052)
                [particularly eqs. 3, 4, 5]
                (http://iopscience.iop.org/article/10.1086/300839/pdf)

        This should be done on the largest number of spectra possible
            simultaneously, for greatest speed. Rules of thumb to come.
        '''

        if w is None:
            w = np.ones_like(f)
        elif type(w) != np.ndarray:
            raise TypeError('w must be array')
        # make singular system non-singular
        else:
            w[w == 0] = eps

        M = np.einsum('kxy,ik,jk->xyij', w, e, e)
        F = np.einsum('kxy,kxy,jk->xyj', w, f, e)

        #M = np.einsum('sk,ik,jk->sij', w, e, e)
        #F = np.einsum('sk,sk,jk->sj', w, f, e)

        try:
            A = np.linalg.solve(M, F)
        except np.linalg.LinAlgError:
            warn('down-projection is non-exact, due to singular matrix',
                 PCProjectionWarning)
            # try lstsq down-projection
            A = StellarPop_PCA.SVD_downproject(M, F)

        A = np.moveaxis(A, -1, 0)

        #A, resid, rank, S = np.linalg.lstsq(M, F)
        #A = np.moveaxis(A, -1, 0)

        return A

    @staticmethod
    def SVD_downproject(M, F, rcond=1e-10):
        # do lstsq down-projection using SVD

        u, s, v = np.linalg.svd(M, full_matrices=False)
        s_max = s.max(axis=-1, keepdims=True)
        s_min = rcond * s_max
        inv_s = np.zeros_like(s)
        inv_s[s >= s_min] = 1. / s[s >= s_min]

        x = np.einsum('...ji,...j->...i', v,
                      inv_s * np.einsum('...ji,...j->...i', u, F.conj()))

        return np.conj(x, x)

    @staticmethod
    def PCA(data, dims=None):
        '''
        perform pure numpy PCA on array of data, returning `dims`-length
            evals and evecs. This method is covariance-based, not SVD-based.
        '''

        if dims is None:
            dims = data.shape[1]

        # calculate the covariance matrix
        R = np.cov(data.T)

        # calculate eigenvectors & eigenvalues of the covariance matrix
        # use 'eigh' rather than 'eig' since R is symmetric,
        # the performance gain is substantial
        evals, evecs = np.linalg.eigh(R)

        # sort evals and evecs in decreasing order
        idx = np.argsort(evals)[::-1]
        evals, evecs = evals[idx], evecs[:, idx]

        # proportion of variance explained
        PVE = evals[:dims] / evals.sum()

        # select the first n eigenvectors (n is desired dimension
        # of rescaled data array, or dims_rescaled_data)
        evecs = evecs[:, :dims].T

        return evecs, PVE

    @staticmethod
    def P_from_K(K):
        '''
        compute straight inverse of all elements of K_PC [q, q, NX, NY]
        '''
        P_PC = np.moveaxis(
            np.linalg.inv(
                np.moveaxis(K, [0, 1, 2, 3], [2, 3, 0, 1])),
            [0, 1, 2, 3], [2, 3, 0, 1])
        return P_PC

    @staticmethod
    def P_from_K_pinv(K, rcond=1e-15):
        '''
        compute P_PC using Moore-Penrose pseudoinverse (pinv)
        '''

        Kprime = np.moveaxis(K, [0, 1, 2, 3], [2, 3, 0, 1])
        swap = np.arange(Kprime.ndim)
        swap[[-2, -1]] = swap[[-1, -2]]
        u, s, v = np.linalg.svd(Kprime)
        cutoff = np.maximum.reduce(s, axis=-1, keepdims=True) * rcond

        mask = s > cutoff
        s[mask] = 1. / s[mask]
        s[~mask] = 0.

        a = np.einsum('...uv,...vw->...uw',
                      np.transpose(v, swap) * s[..., None, :],
                      np.transpose(u, swap))
        return np.moveaxis(a, [0, 1, 2, 3], [2, 3, 0, 1])

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
        self.vel_mask = self.dap_hdulist['STELLAR_VEL_MASK'].data

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

    def correct_and_match(self, template_logl, template_dlogl=None):
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
            ivar=self.ivar * (~self.DONOTUSE), r_v=r_v, EBV=EBV)

        # prepare to de-redshift
        # total redshift of each spaxel
        z_map = (1. + self.z) * (1. + (self.vel / c.c).to('').value) - 1.
        self.z_map = z_map

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

        # where does template grid start?
        tem_l = 10.**template_logl
        self.tem_l = tem_l
        tem_nl = tem_l.shape[0]
        # what pixel of deredshifted grid is that?
        ix0 = np.argmin(np.abs(l_rest - tem_l[0]), axis=0)

        # test whether wavelength grid extends beyond MaNGA coverage
        # in any spaxels
        bad_logl_extent = np.logical_or.reduce(
            ((l_rest < tem_l[0]), (l_rest > tem_l[-1])))

        # select same num. of elements of f_rest & ivar_rest,
        # starting at ix0
        ixs = (ix0[None, ...] + np.arange(tem_nl)[..., None, None])
        _, I, J = np.ix_(*[range(i) for i in f_rest.shape])

        self.flux_regr = f_rest[ixs, I, J]
        self.ivar_regr = ivar_rest[ixs, I, J]
        self.DONOTUSE_regr = self.DONOTUSE[ixs, I, J]

        # also select from bad_logl_extent
        bad_logl_extent = bad_logl_extent[ixs, I, J]

        # replace out-of-range elements with f = 0 and ivar = 0
        self.ivar_regr[bad_logl_extent] = 0.
        self.flux_regr[bad_logl_extent] = 0.

        self.spax_mask = self.vel_mask

        return self.flux_regr, self.ivar_regr, self.spax_mask

    def compute_eline_mask(self, template_logl, template_dlogl=None, ix_eline=7,
                           half_dv=500. * u.Unit('km/s')):

        from elines import (balmer_low, balmer_high, helium,
                            bright_metal, faint_metal)

        if template_dlogl is None:
            template_dlogl = spec_tools.determine_dlogl(template_logl)

        EW = self.eline_EW(ix=ix_eline)
        # thresholds are set very aggressively for debugging, but should be
        # revisited in the future
        # proposed values... balmer_low: 0, balmer_high: 2, helium: 2
        #                    brightmetal: 0, faintmetal: 5, paschen: 10
        add_balmer_low = (EW >= 0. * u.AA)
        add_balmer_high = (EW >= .5 * u.AA)
        add_helium = (EW >= .5 * u.AA)
        add_brightmetal = (EW >= 0. * u.AA)
        add_faintmetal = (EW >= .5 * u.AA)

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

    def coadd(self, good=None):
        '''
        return coadded spectrum and ivar

        params:
         - good: map of good spaxels
        '''

        if good is None:
            good = np.ones_like(self.flux_regr[0, ...])

        ivar = self.ivar_regr * good[None, ...]
        map_shape = ivar.shape[1:]
        tem_l = np.tile(self.tem_l[:, None, None],
                        (1, ) + map_shape)

        lam, flux, ivar = ut.coadd(
            l=tem_l, f=self.flux_regr, ivar=ivar)
        lam, flux, ivar = (lam[:, None, None], flux[:, None, None],
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

    def __init__(self, pca, dered, K_obs, z, cosmo, norm_params={}, figdir='.',
                 truth=None):
        self.objname = dered.drp_hdulist[0].header['plateifu']
        self.pca = pca
        self.dered = dered
        self.cosmo = cosmo
        self.z = z
        self.norm_params = norm_params
        self.K_obs = K_obs
        self.truth = truth  #  known-truth parameters for fake data

        # where to save all figures
        self.figdir = figdir
        self.__setup_figdir__()

        self.E = pca.PCs

        self.O, self.ivar, mask_spax = dered.correct_and_match(
            template_logl=pca.logl, template_dlogl=None)
        self.mask_cube = dered.compute_eline_mask(
            template_logl=pca.logl, template_dlogl=None) * \
            (~self.dered.DONOTUSE_regr)

        self.SNR_med = np.median(self.O * np.sqrt(self.ivar) + eps,
                                 axis=0)

        self.A, self.M, self.a_map, self.O_sub, self.O_norm = pca.project_cube(
            f=self.O, ivar=self.ivar, mask_spax=mask_spax,
            mask_cube=self.mask_cube)

        # original spectrum
        self.O = np.ma.array(self.O, mask=self.mask_cube)
        self.ivar = np.ma.array(self.ivar, mask=self.mask_cube)
        self.O_norm = np.ma.array(self.O_norm, mask=self.mask_cube)

        # how to reconstruct datacube from PC weights cube
        self.O_recon = np.ma.array(pca.reconstruct_normed(self.A),
                                   mask=self.mask_cube)

        self.resid = np.abs((self.O_norm - self.O_recon) / self.O_norm)

        self.K_PC = pca.build_PC_cov_full_iter(
            z_map=dered.z_map, K_obs_=self.K_obs, snr_map=self.SNR_med)

        self.P_PC = StellarPop_PCA.P_from_K_pinv(self.K_PC)

        self.map_shape = self.O.shape[-2:]
        self.ifu_ctr_ix = [s // 2 for s in self.map_shape]

        self.w = pca.compute_model_weights(P=self.P_PC, A=self.A,
                                           **norm_params)
        # no data
        self.nodata = (dered.drp_hdulist['RIMG'].data == 0.)

        self.mask_map = np.logical_or.reduce(
            (mask_spax, self.nodata))

        # spaxel is bad if < 25 models have weights 1/100 max, and no other problems
        self.badPDF = np.logical_and.reduce(
            ((self.sample_diag(f=.01) < 25), ~self.mask_map))
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
        ABmag = self.dered.S2P.ABmags['-'.join(
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
        ax.axhline(ix[0])
        ax.axvline(ix[1])

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
        bestfit_ = ax1.plot(self.l, bestfit,
                            drawstyle='steps-mid', c='c', label='Best Model',
                            linewidth=0.5)

        # original & reconstructed
        O_norm = self.O_norm[:, ix[0], ix[1]]
        O_recon = self.O_recon[:, ix[0], ix[1]]
        ivar = self.ivar[:, ix[0], ix[1]]

        Onm = np.ma.median(O_norm)
        Orm = np.ma.median(O_recon)
        ivm = np.ma.median(ivar)

        orig_ = ax1.plot(
            self.l, O_norm, drawstyle='steps-mid',
            c='b', label='Orig.', linewidth=0.5)
        recon_ = ax1.plot(
            self.l, O_recon, drawstyle='steps-mid',
            c='g', label='Recon.', linewidth=0.5)

        # inverse-variance (weight) plot
        ivar_ = ax1.plot(
            self.l, (ivar / ivm) * Onm,
            drawstyle='steps-mid', c='m', label='Wt.', linewidth=0.25)

        ax1.legend(loc='best', prop={'size': 6})
        ax1.set_ylabel(r'$F_{\lambda}$ (rel)')
        ax1.set_yticks(np.arange(0.5, 2.5, 0.5))
        ax1.set_ylim([-0.1 * Onm, 2.25 * Onm])
        ax1.axhline(y=0., xmin=self.l.min(), xmax=self.l.max(),
                    c='k', linestyle=':')
        ax1.set_xticklabels([])

        # residual
        resid_ = ax2.plot(
            self.l, self.resid[:, ix[0], ix[1]], drawstyle='steps-mid',
            c='blue', linewidth=0.5)
        resid_avg_ = ax2.axhline(
            self.resid[:, ix[0], ix[1]].mean(), linestyle='--', c='salmon',
            linewidth=0.5)

        try:
            ax2.set_yscale('log')
        except ValueError:
            pass
        ax2.set_xlabel(r'$\lambda$ [$\textrm{\AA}$]')
        ax2.set_ylabel('Resid.')
        ax2.set_ylim([1.0e-3, 1.0])

        return orig_, recon_, bestfit_, ivar_, resid_, resid_avg_, ix

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
            qty=qty_str, W=self.w, factor=f)

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
            qty=qty_str, W=self.w, factor=f)

        if scale == 'log':
            return 10.**P50

        return P50

    def Mstar_integrated(self, band='i'):
        '''
        calculate integrated spectrum, and then compute stellar mass from that
        '''

        _, O, ivar = self.dered.coadd(good=~self.mask_map)

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

        K_PC = pca.build_PC_cov_full_iter(
            z_map=z_map, K_obs_=self.K_obs, snr_map=SNR)

        P_PC = StellarPop_PCA.P_from_K_pinv(
            .1 * K_PC / np.median(np.diag(K_PC[..., 0, 0])))

        w = pca.compute_model_weights(P=P_PC, A=A, **self.norm_params)

        #print('good models:', self.sample_diag(f=.01, w=w))

        lum = np.ma.masked_invalid(self.lum(band=band))

        P50, *_, scale = self.pca.param_cred_intvl(
            qty='ML{}'.format(band), W=w, factor=lum.sum())

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

        logmstar_coadd_tot = np.log10(self.Mstar_integrated(band))

        #print(logmstar_tot, logmstar_coadd_tot)

        try:
            TeX1 = ''.join((r'$\log{\frac{M_{*}}{M_{\odot}}}$ = ',
                            '{:.2f}'.format(logmstar_tot)))
        except TypeError:
            TeX1 = 'ERROR'

        try:
            TeX2 = ''.join((r'$\log{\frac{M_{*,add}}{M_{\odot}}}$ = ',
                            '{:.2f}'.format(logmstar_coadd_tot)))
        except TypeError:
            TeX2 = 'ERROR'

        ax1xlims, ax1ylims = ax1.get_xlim(), ax1.get_ylim()

        ax1.text(x=tr((0, 1), ax1xlims, 0.05),
                 y=tr((0, 1), ax1ylims, 0.05), s=TeX1)
        ax1.text(x=tr((0, 1), ax1xlims, 0.5),
                 y=tr((0, 1), ax1ylims, 0.05), s=TeX2)

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
        q, w = q[np.isfinite(q)], w[np.isfinite(q)]

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
            hprior = ax.plot(qgrid, prigrid, color='orange', linestyle='-',
                             label='prior', linewidth=0.5)
        else:
            hprior = ax_.hist(
                q, bins=bins, normed=True, histtype='step', color='orange',
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

        # value of best-fit spectrum
        ax.axvline(q[np.argmax(w)], color='c', linewidth=0.5, label='best')

        # if we're using fake data, we have some ground truth!
        if self.truth is not None:
            t = self.truth[qty]
            ax.axvline(t, color='r', linewidth=0.5, label='truth')

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
            ax.set_xlabel(' '.join((r'$\Delta$', u.arcsec._repr_latex_())))
            ax.set_ylabel(' '.join((r'$\Delta$', u.arcsec._repr_latex_())))

            # over XOFFSET & YOFFSET
            for i in range(2):
                ax.coords[i].set_major_formatter('x')
                ax.coords[i].set_ticks(spacing=5. * u.arcsec)
                ax.coords[i].set_format_unit(u.arcsec)

            if bad:
                # figures_tools.annotate_badPDF(ax, self.goodPDF)
                pass

    def __setup_qty_fig__(self):
        fig = plt.figure(figsize=(9, 4), dpi=300)

        gs = gridspec.GridSpec(1, 2, wspace=.175, left=.075, right=.975,
                               bottom=.11, top=.9)
        ax1 = fig.add_subplot(gs[0], projection=self.wcs_header_offset)
        ax2 = fig.add_subplot(gs[1], projection=self.wcs_header_offset)

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
        spec_ax.tick_params(axis='y', which='major', labelsize=10)
        resid_ax.tick_params(axis='both', which='major', labelsize=10)
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
            qty=qty, W=self.w, factor=f)

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

        pad = np.maximum(.1 * np.ones_like(sigma), 2.5 * sigma)
        lpred = np.min(q_pred - pad)
        upred = np.max(q_pred + pad)

        ax.set_ylim([lpred, upred])

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
            'ML{}'.format(mlb), W=self.w, factor=None)

        if scale == 'linear':
            ml = np.log10(ml)

        # size of points determined by signal in redder band
        b2_img = self.dered.drp_hdulist['{}img'.format(b2)].data
        s = 10. * np.arctan(0.05 * b2_img / np.median(b2_img[b2_img > 0.]))

        sc = ax.scatter(col.flatten(), ml.flatten(),
                        facecolor=ptcol, edgecolor='None', s=s.flatten(),
                        label=self.objname)

        if type(ptcol) != str:
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
                q, W=self.w, factor=None)[0] for q in colorby], axis=0), axis=0)
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
        sig_dap_corr = self.dered.dap_hdulist['STELLAR_SIGMACORR'].data
        sig_dap = np.sqrt(sig_dap**2. - sig_dap_corr**2.)
        sig_dap_mask = self.dered.dap_hdulist['STELLAR_SIGMA_MASK'].data
        sig_dap = np.ma.array(sig_dap, mask=sig_dap_mask).flatten()
        sig_dap_unc = 1. / np.sqrt(
            self.dered.dap_hdulist['STELLAR_SIGMA_IVAR'].data).flatten()

        sig_pca, sig_pca_lunc, sig_pca_uunc, _ = self.pca.param_cred_intvl(
            qty='sigma', W=self.w)
        sig_pca = np.sqrt(sig_pca**2. - sig_dap_corr**2.)
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


def setup_pca(base_dir, base_fname, fname=None,
              redo=True, pkl=True, q=7, nfiles=5, fre_target=.005,
              pca_kwargs={}):
    import pickle
    if (fname is None):
        redo = True

        if pkl:
            fname = 'pca.pkl'

    if redo:
        #K_obs = cov_obs.Cov_Obs.from_fits('manga_Kspec.fits')
        K_obs = cov_obs.Cov_Obs.from_YMC_BOSS(
            'cov_obs_YMC_BOSS.fits')
        pca = StellarPop_PCA.from_FSPS(
            K_obs=K_obs, base_dir=base_dir, base_fname=base_fname,
            nfiles=nfiles, **pca_kwargs)
        if pkl:
            pickle.dump(pca, open(fname, 'wb'))

    else:
        K_obs = cov_obs.Cov_Obs.from_fits('manga_Kspec.fits')
        pca = pickle.load(open(fname, 'rb'))

    if q == 'auto':
        q_opt = pca.xval_fromfile(
            fname=os.path.join(base_dir, '{}_test.fits'.format(base_fname)),
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

def run_object(row, pca, force_redo=False, fake=False):

    plateifu = row['plateifu']

    if (not force_redo) and (os.path.isdir(plateifu)):
        return

    plate, ifu = plateifu.split('-')

    if fake:
        data = FakeData.from_FSPS(
            fname='CSPs_CKC14_MaNGA/CSPs_test.fits', i=None,
            plateifu_base=plateifu, pca=pca)
        data.write()
        dered = MaNGA_deredshift.from_fakedata(
            plate=int(plate), ifu=int(ifu), MPL_v=mpl_v,
            basedir='fakedata', row=row)
        figdir = os.path.join('fakedata', 'results', plateifu)
        truth_fname = os.path.join(
            'fakedata', '{}_truth.tab'.format(plateifu))
        truth = t.Table.read(truth_fname, format='ascii')[0]
    else:
        dered = MaNGA_deredshift.from_plateifu(
            plate=int(plate), ifu=int(ifu), MPL_v=mpl_v, row=row)
        figdir = os.path.join('results', plateifu)
        truth = None

    z_dist = row['nsa_zdist']

    pca_res = PCA_Result(
        pca=pca, dered=dered, K_obs=K_obs, z=z_dist, cosmo=cosmo,
        norm_params={'norm': 'L2', 'soft': False}, figdir=figdir,
        truth=truth)

    pca_res.make_sample_diag_fig()

    if fake:
        for ix in np.ndindex(pca_res.mask_map.shape):
            if (~pca_res.nodata[ix[0], ix[1]]) and (np.random.rand() < 0.1):
                pca_res.make_full_QA_fig(ix=ix, kde=(False, False))
    else:
        pca_res.make_full_QA_fig(kde=(False, False))

    pca_res.make_comp_fig()

    pca_res.make_qty_fig(qty_str='MLr')
    pca_res.make_qty_fig(qty_str='MLi')
    pca_res.make_qty_fig(qty_str='MLz')

    pca_res.make_qty_fig(qty_str='MWA')

    pca_res.make_Mstar_fig(band='r')
    pca_res.make_Mstar_fig(band='i')
    pca_res.make_Mstar_fig(band='z')

    pca_res.make_radial_gp_fig(qty='MLr', q_bdy=[.01, 100.])
    pca_res.make_radial_gp_fig(qty='MLi', q_bdy=[.01, 100.])
    pca_res.make_radial_gp_fig(qty='MLz', q_bdy=[.01, 100.])

    pca_res.make_qty_fig('Dn4000')

    pca_res.make_color_ML_fig(mlb='i', b1='g', b2='i', colorby='R')
    pca_res.make_color_ML_fig(mlb='i', b1='g', b2='i', colorby='tau_V mu')

    pca_res.compare_sigma()

    pca_res.sigma_vel()

    return pca_res

if __name__ == '__main__':
    howmany = 30
    cosmo = WMAP9
    warn_behav = 'ignore'

    mpl_v = 'MPL-5'

    pca_kwargs = {'lllim': 3800. * u.AA, 'lulim': 8800. * u.AA}

    pca, K_obs = setup_pca(
        fname='pca.pkl', base_dir='CSPs_CKC14_MaNGA', base_fname='CSPs',
        redo=False, pkl=True, q=10, fre_target=.005, nfiles=30,
        pca_kwargs=pca_kwargs)

    drpall = m.load_drpall(mpl_v, index='plateifu')

    row = drpall.loc['8249-12704']

    with catch_warnings():
        simplefilter(warn_behav)
        pca_res = run_object(row=row, pca=pca, force_redo=False,
                             fake=True)
        pca_res = run_object(row=row, pca=pca, force_redo=False,
                             fake=False)

    '''
    drpall = drpall[drpall['ifudesignsize'] > 0.]
    drpall = m.shuffle_table(drpall)
    drpall = drpall[:howmany]

    with ProgressBar(howmany) as bar:
        for i, row in enumerate(drpall):
            plateifu = row['plateifu']

            try:
                with catch_warnings():
                    simplefilter(warn_behav)
                    run_object(row=row, pca=pca, force_redo=False)
            except Exception as e:
                print('ERROR: {}'.format(plateifu))
                print(e)
                continue
            finally:
                bar.update()
    '''
