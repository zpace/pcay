import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as mplcm
from matplotlib.colors import Normalize, LogNorm
from matplotlib import gridspec

from astropy import constants as c, units as u, table as t
from astropy.io import fits
from astropy import wcs
from specutils.extinction import reddening
from astropy.cosmology import WMAP9
import pywcsgrid2 as wg2

import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.integrate import quad
from statsmodels.nonparametric.kde import KDEUnivariate

import spec_tools
import ssp_lib
import manga_tools as m
import cov_obs

from itertools import izip, product
from glob import glob
from copy import copy

eps = np.finfo(float).eps

class StellarPop_PCA(object):
    '''
    class for determining PCs of a library of synthetic spectra
    '''
    def __init__(self, l, trn_spectra, gen_dicts, metadata, K_obs,
                 dlogl=None):
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

        self.l = l
        self.logl = np.log10(l.to('AA').value)
        if dlogl is None:
            dlogl = np.round(np.mean(logl[1:] - logl[:-1]), 8)
        self.dlogl = dlogl

        self.trn_spectra = trn_spectra
        self.metadata = metadata
        self.metadata_a = np.array(
            self.metadata).view(dtype=float).reshape(
            (len(self.metadata), -1))
        self.gen_dicts = gen_dicts

        # observational covariance matrix
        if K_obs.__class__ != cov_obs.Cov_Obs:
            raise TypeError('incorrect observational covariance matrix class!')

        if K_obs.dlogl != self.dlogl:
            raise PCAError('non-matching log-lambda spacing!')

    @classmethod
    def from_YMC(cls, lib_para_file, form_file,
                 spec_file_dir, spec_file_base, K_obs, BP):
        '''
        initialize object from CSP library provided by Y-M Chen, which is
            based on a BC03 model library
        '''

        # get wavelength grid, and prepare for interpolation of model
        hdulist = fits.open(os.path.join(spec_file_dir, '{}_1.fits'.format(
            spec_file_base)))
        l_raw = hdulist[2].data
        hdulist.close()

        l_raw_good = (3700. <= l_raw) * (l_raw <= 5500.)
        l_raw = l_raw[l_raw_good]
        dlogl_final = 1.0e-4
        l_final= 10.**np.arange(np.log10(3700.), np.log10(5500.),
                                dlogl_final)
        dl = 10.**(np.log10(l_final) + dlogl_final/2.) - \
            10.**(np.log10(l_final) - dlogl_final/2.)
        nl_final = len(l_final)

        # load metadata tables
        lib_para = t.Table.read(lib_para_file, format='ascii')

        nspec = len(lib_para)
        form_data = t.Table.read(form_file, format='ascii')
        form_data_goodcols = ['zmet', 'Tau_v', 'mu']

        for n in form_data_goodcols:
            lib_para.add_column(
                t.Column(data=np.zeros(len(lib_para)), name=n))

        # initialize array that resampled spectra will go into
        spec = np.empty((nspec, nl_final))
        goodspec = np.ones(nspec).astype(bool)
        # and fill it
        for i in range(nspec):
            # create a hypothetical filename
            fname = os.path.join(spec_file_dir, '{}_{}.fits'.format(
                spec_file_base, i))

            # handle file-DNE case exception-less-ly
            if not os.path.exists(fname):
                goodspec[i] = False
                continue

            try:
                hdulist = fits.open(fname)
            # if there's another issue...
            except IOError:
                #print 'bad!', fname
                goodspec[i] = False
            else:
                f_lambda = hdulist[3].data[l_raw_good]
                spec[i, :] = interp1d(l_raw, f_lambda)(l_final)
                for n in form_data_goodcols:
                    lib_para[i-1][n] = form_data[i][n]

            finally:
                hdulist.close()

        metadata = lib_para

        ixs = np.arange(nspec)
        metadata.remove_rows(ixs[~goodspec])
        spec = spec[goodspec, :]

        # compute mass to light ratio
        f_r = BP.interps['r'](l_final)
        f_i = BP.interps['i'](l_final)
        f_z = BP.interps['z'](l_final)

        L_r = (f_r * spec).sum(axis=1)
        L_i = (f_i * spec).sum(axis=1)
        L_z = (f_z * spec).sum(axis=1)

        MLr, MLi, MLz = 1./L_r, 1./L_i, 1./L_z

        metadata['Fstar'] = metadata['mfb_1e9'] / metadata['mgalaxy']

        metadata = metadata['MWA', 'LrWA', 'D4000', 'Hdelta_A', 'Fstar',
                            'zmet', 'Tau_v', 'mu']
        metadata.add_column(t.Column(data=MLr, name='MLr'))
        metadata.add_column(t.Column(data=MLi, name='MLi'))
        metadata.add_column(t.Column(data=MLz, name='MLz'))

        return cls(l=l_final*u.Unit('AA'), trn_spectra=spec,
                   gen_dicts=None, metadata=metadata, dlogl=dlogl_final,
                   K_obs=K_obs)

    # =====
    # methods
    # =====

    def run_pca_models(self, mask_half_dv=500.*u.Unit('km/s'),
                       q=None, max_q=None):
        '''
        run PCA on library of model spectra
        '''

        if (q is None) and (max_q is None):
            raise ValueError('must provide either or both `q`/`max_q`!')
        # first run some prep on the model spectra

        # find lower and upper edges of each wavelength bin,
        # and compute width of bins
        dl = self.dl

        # normalize each spectrum to the mean flux density
        self.a = np.mean(self.trn_spectra, axis=1)

        self.normed_trn = self.trn_spectra/self.a[:, np.newaxis]
        self.M = np.mean(self.normed_trn, axis=0)

        # if user asks to test param reconstruction from PCs over range
        # of # of PCs kept
        # this does not keep record of each iteration's output, even at end
        if max_q is not None:
            res_q = [None, ] * max_q
            for dim_pc_subspace in range(1, max_q):
                PCs = self.PCA(
                    self.normed_trn - self.M, dims=dim_pc_subspace)
                # transformation matrix: spectra -> PC amplitudes
                tfm_sp2PC = PCs.T

                # project back onto the PCs to get the weight vectors
                trn_PC_wts = (self.normed_trn - self.M).dot(tfm_sp2PC)
                # and reconstruct the best approximation for the spectra from PCs
                trn_recon = trn_PC_wts.dot(PCs)
                # residuals
                trn_resid = self.normed_trn - (self.M + trn_recon)

                cov_th = np.cov(trn_resid.T)

                # find weights of each PC in determining parameters in metadata
                (n_, p_, q_) = (self.trn_spectra.shape[0],
                                len(self.metadata.colnames),
                                dim_pc_subspace)
                m_dims_ = (n_, p_, q_)

                res_q[dim_pc_subspace] = minimize(
                    self.__obj_fn_Pfit__,
                    x0=np.random.uniform(-1, 1, p_ * (q_ + 1)),
                    args=(self.metadata_a, trn_PC_wts, m_dims_))

                plt.scatter(
                    dim_pc_subspace,
                    res_q[dim_pc_subspace].fun/dim_pc_subspace,
                    c='b', marker='x')

            plt.yscale('log')
            plt.xlabel(r'\# of PCs kept')
            plt.ylabel(r'$\Delta/\delta$ (sum-squared-error over DOF)')
            plt.title('Parameter Estimation Errors')
            plt.savefig('param_est_QA.png')

        # if user asks to run parameter regression of specific # of PCs
        # this also sets self attributes, so that everything is kept
        if q is not None:
            dim_pc_subspace = q
            self.PCs = self.PCA(
                self.normed_trn - self.M, dims=dim_pc_subspace)
            # transformation matrix: spectra -> PC amplitudes
            self.tfm_sp2PC = self.PCs.T

            # project back onto the PCs to get the weight vectors
            self.trn_PC_wts = (self.normed_trn - self.M).dot(self.tfm_sp2PC)
            # and reconstruct the best approximation for the spectra from PCs
            self.trn_recon = self.trn_PC_wts.dot(self.PCs)
            # residuals
            self.trn_resid = self.normed_trn - (self.M + self.trn_recon)

            self.cov_th = np.cov(self.trn_resid.T)

            # find weights of each PC in determining parameters in metadata
            (n_, p_, q_) = (self.trn_spectra.shape[0],
                            len(self.metadata.colnames),
                            dim_pc_subspace)
            m_dims_ = (n_, p_, q_)

            res_q = minimize(
                self.__obj_fn_Pfit__,
                x0=np.random.uniform(-1, 1, p_ * (q_ + 1)),
                args=(self.metadata_a, self.trn_PC_wts, m_dims_))

            self.PC2params_A = res_q.x[:(q_ * p_)].reshape((q_, p_))
            self.PC2params_Z = res_q.x[(q_ * p_):].flatten()

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
        dl = self.dl

        # manage masks
        if mask_spax is not None:
            ivar *= (~mask_spax).astype(float)
        if mask_spec is not None:
            ivar *= (~mask_spec[:, np.newaxis, np.newaxis]).astype(float)
        if mask_cube is not None:
            ivar *= (~mask_cube).astype(float)

        # make f and ivar effectively a list of spectra
        f = np.transpose(f, (1,2,0)).reshape(-1, cube_shape[0])
        ivar = np.transpose(ivar, (1,2,0)).reshape(-1, cube_shape[0])
        ivar[ivar == 0.] = eps

        # normalize by average flux density
        a = np.average(f, weights=ivar, axis=1)
        a[a == 0.] = 1.
        f = f/a[:, np.newaxis]
        # get mean spectrum
        M = np.average(f, weights=ivar, axis=0)
        f = f - M[np.newaxis, :]

        # need to do some reshaping
        A = self.robust_project_onto_PCs(e=self.PCs, f=f, w=ivar)

        A = A.T.reshape((A.shape[1], ) + cube_shape[1:])

        O_norm = f.T.reshape((f.shape[1], ) + cube_shape[1:])

        return A, M, a.reshape(cube_shape[-2:]), O_norm

    def _compute_i0_map(self, logl, z_map):
        '''
        compute the index of some array corresponding to the given
            wavelength at some redshift

        params:
         - tem_logl0: the smallest wavelength of the fixed-grid template
            that will be the destination of the bin-shift
         - logl: the wavelength grid that will be transformed
         - z_map: the 2D array of redshifts used to figure out the offset
        '''

        tem_logl0 = self.logl[0]
        tem_logl0_z = np.log10(
            10.**(tem_logl0) * (1. + z_map))
        cov_logl_tiled = np.tile(
            logl[:, np.newaxis, np.newaxis],
            z_map.shape)

        # find the index for the wavelength that best corresponds to
        # an appropriately redshifted wavelength grid
        logl_diff = tem_logl0_z[np.newaxis, :, :] - cov_logl_tiled
        i0_map = np.argmin(np.abs(logl_diff), axis=0)

        return i0_map

    def compute_spec_cov_full(self, a_map, z_map, SB_map, obs_logl, K_obs_):
        '''
        DO NOT USE!! Array output too large, raises MemoryError

        compute a full PC covariance matrix for each spaxel

        params:
         - a_map: mean flux-density of original spectra (n by n)
         - z_map: spaxel redshift map (get this from an dered instance)
         - SB_map: r-band surface brightness map (nMgy/arcsec2) to scale
            observational covariance matrix by (wrt to avg SB)
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

        K_th = self.cov_th

        # figure out where to start sampling the covariance matrix

        cov_logl = K_obs_.logl
        nlam = K_th.shape[0]
        tem_logl0 = self.logl[0]

        i0_map = self._compute_i0_map(
            tem_logl0=tem_logl0, logl=cov_logl, z_map=z_map)

        N = np.arange(nlam)
        I, J = np.meshgrid(N, N)

        K_obs = K_obs_.cov[
            ix0[..., None, None] + J, ix0[..., None, None] + I]
        SB_BOSS = K_obs_.SB_r_mean

        K_full = a_map * K_th + (SB_map / SB_BOSS).to('').value * K_obs

        return K_full

    def _compute_PC_cov_spax(self, K_spec):
        '''
        compute the PC covariance matrix for one spaxel

        See Chen+'12 (Eq 11) for full expression
        '''

        E = self.PCs

        return E.dot(K_spec).dot(E.T)

    def _compute_spec_cov_spax(self, K_obs_, i0, a, f):
        '''
        compute the spectral covariance matrix for one spaxel

        params:
         - K_obs_: observational covariance object
         - i0: starting index of observational covariance matrix
         - f: surface brightness ratio btwn BOSS objects & spaxel
         - a: mean flux-density of original spectra
        '''

        K_th = self.cov_th
        nspec = K_th.shape[0]
        f = np.ones_like(f)
        K_full = K_obs_.cov[i0:i0+nspec, i0:i0+nspec] * f + K_th * a**2.

        return K_full

    def build_PC_cov_full_iter(self, a_map, z_map, SB_map, obs_logl, K_obs_):
        '''
        for each spaxel, build a PC covariance matrix, and return in array
            of shape (q, q, n, n)

        NOTE: this technique computes each spaxel's covariance matrix
            separately, not all at once! Simultaneous computation requires
            too much memory!

        params:
         - E: eigenvectors of PC basis
         - a_map: mean flux-density of original spectra (n by n)
         - z_map: spaxel redshift map (get this from an dered instance)
         - SB_map: r-band surface brightness map (nMgy/arcsec2) to scale
            observational covariance matrix by (wrt to avg SB)
         - obs_logl: log-lambda vector (1D) of datacube
         - K_obs_: observational spectral covariance object
        '''

        E = self.PCs
        q, l = E.shape
        cubeshape = a_map.shape

        # build an array to hold covariances
        K_PC = np.empty((q, q) + cubeshape)

        # compute starting indices for covariance matrix
        i0_map = self._compute_i0_map(logl=obs_logl, z_map=z_map)

        inds = np.ndindex(cubeshape) # iterator over datacube shape

        for ind, i0, SB, a in izip(inds, i0_map.flatten(), SB_map.flatten(),
                                   a_map.flatten()):
            K_spec = self._compute_spec_cov_spax(
                K_obs_=K_obs_, i0=i0, a=a, f=K_obs_.SB_r_mean/SB)
            K_PC[..., ind[0], ind[1]] = E.dot(K_spec).dot(E.T)

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
        '''

        C = self.trn_PC_wts # shape (nmodel, q)
        D = np.abs(C[..., np.newaxis, np.newaxis] - A[np.newaxis, ...])
        # D goes [MODELNUM, PCNUM, XNUM, YNUM]

        #print D.shape, P.shape

        chi2 = np.einsum('cixy,ijxy,cjxy->cxy', D, P, D)
        w = np.exp(-chi2/2.)

        return w#/w.max(axis=0)

    def param_pct_map(self, qty, W, P, factor=None):
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
        '''

        cubeshape = W.shape[-2:]
        Q = self.metadata[qty][np.isfinite(self.metadata[qty])]

        if factor == None:
            factor = np.ones(cubeshape)

        W = W[np.isfinite(self.metadata[qty])]

        inds = np.ndindex(*cubeshape)

        A = np.empty((len(P),) + cubeshape)

        for ind in inds:
            q = Q
            w = W[:, ind[0], ind[1]]
            i_ = np.argsort(q, axis=0)
            q, w = q[i_], w[i_]
            A[:, ind[0], ind[1]] = np.interp(P, 100.*w.cumsum()/w.sum(), q)

        return A * factor[np.newaxis, ...]

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
        return 10.**(self.logl - self.dlogl/2.)

    @property
    def l_upper(self):
        return 10.**(self.logl + self.dlogl/2.)

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
         - e: n-by-l array of eigenvectors
         - f: n-by-m array, where n is the number of spectra, and m
            is the number of spectral wavelength bins. [Flux-density units]
         - w: n-by-m array, containing the inverse-variances for each
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

        M = np.einsum('sk,ik,jk->sij', w, e, e)
        F = np.einsum('sk,sk,jk->sj', w, f, e)

        A = np.linalg.solve(M, F)

        return A

    @staticmethod
    def __obj_fn_Pfit__(XZ, P, C, dims):
        '''
        objective function for fitting parameters with PCs

        params:
            - P: n-by-p array, representing n spectra having p parameters
                associated with them. P contains the true parameter values
                for all spectra
            - C: n-by-q array, representing principal component amplitudes
                associated with a PCA realization, where n is the number of
                spectra used in the PCA and q is the number of principal
                components kept
            - X: q-by-p array, which takes C.T to P_, an estimate of
                the true P
            - Z: length-p vector, representing the zeropoint associated
                with each parameter P[:, i]

        n: number of spectra
        p: number of galaxy parameters estimated
        q: number of PCs kept (i.e., dimension of PC subspace)
        '''
        (n, p, q) = dims

        X = XZ[:(q * p)].reshape((q, p))
        Z = XZ[(q * p):].flatten()

        P_ = np.dot(C, X) + Z
        P_resid_sq = (((P_ - P).sum(axis=1))**2.).sum()
        return P_resid_sq

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

        # sort eigenvalue in decreasing order
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]

        # sort eigenvectors according to same index
        evals = evals[idx]

        # select the first n eigenvectors (n is desired dimension
        # of rescaled data array, or dims_rescaled_data)
        evecs = evecs[:, :dims].T

        # transformation matrix takes (scaled) data to array of PC weights

        return evecs

    @staticmethod
    def P_from_K(K):
        P_PC = np.moveaxis(
            np.linalg.inv(
                np.moveaxis(K, [0, 1, 2, 3], [2, 3, 0, 1])),
            [0, 1, 2, 3], [2, 3, 0, 1])
        return P_PC

    # =====
    # under the hood
    # =====

    def __str__(self):
        return 'PCA object: q = {0[0]}, nlam = {0[1]}'.format(self.PCs.shape)

class Bandpass(object):
    '''
    class to manage bandpasses for multiple filters
    '''
    def __init__(self):
        self.bands = []
        self.interps = {}

    def add_bandpass(self, name, l, ff):
        self.bands.append(name)
        self.interps[name] = interp1d(
            x=l, y=ff, kind='linear', bounds_error=False, fill_value=0.)

    def add_bandpass_from_ascii(self, fname, band_name):
        table = t.Table.read(fname, format='ascii', names=['l', 'ff'])
        l = np.array(table['l'])
        ff = np.array(table['ff'])
        self.add_bandpass(name=band_name, l=l, ff=ff)

    def __call__(self, flam, l, units=None):
        if units == None:
            units = {}
            units['flam'] = u.Unit('Lsun AA-1')
            units['l'] = u.AA
            units['dl'] = units['l']

        lgood = (l >= 2000.) * (l <= 15000.)
        l, flam = l[lgood], flam[lgood]

        flam_interp = interp1d(
            x=l, y=flam, kind='linear', bounds_error=False, fill_value=0.)
        return {n: quad(
            lambda l: interp(l) * flam_interp(l),
            a=l.min(), b=l.max(), epsrel=1.0e-5, limit=len(l))[0] * \
                (units['flam'] * units['l']).to('Lsun')
                for n, interp in self.interps.iteritems()}

def setup_bandpasses():
    BP = Bandpass()
    BP.add_bandpass_from_ascii(
        fname='filters/r_SDSS.res', band_name='r')
    BP.add_bandpass_from_ascii(
        fname='filters/i_SDSS.res', band_name='i')
    BP.add_bandpass_from_ascii(
        fname='filters/z_SDSS.res', band_name='z')
    return BP

class PCAError(Exception):
    '''
    general error for PCA
    '''
    pass

class MaNGA_deredshift(object):
    '''
    class to deredshift reduced MaNGA data based on velocity info from DAP

    preserves cube information, in general

    also builds in a check on velocity coverage, and computes a mask
    '''

    spaxel_side = 0.5 * u.arcsec

    def __init__(self, drp_hdulist, dap_hdulist,
                 max_vel_unc=500.*u.Unit('km/s'), drp_dlogl=None,
                 MPL_v='MPL-4'):
        self.drp_hdulist = drp_hdulist
        self.dap_hdulist = dap_hdulist
        self.plateifu = self.drp_hdulist[0].header['PLATEIFU']

        self.vel = dap_hdulist['STELLAR_VEL'].data * u.Unit('km/s')
        self.vel_ivar = dap_hdulist['STELLAR_VEL_IVAR'].data * u.Unit(
            'km-2s2')

        self.z = m.get_drpall_val(
            os.path.join(
                m.drpall_loc, 'drpall-{}.fits'.format(m.MPL_versions[MPL_v])),
            ['nsa_redshift'], self.plateifu)[0]['nsa_redshift']

        # mask all the spaxels that have high stellar velocity uncertainty
        self.vel_ivar_mask = 1./np.sqrt(self.vel_ivar) > max_vel_unc
        self.vel_mask = self.dap_hdulist['STELLAR_VEL_MASK'].data

        self.drp_logl = np.log10(drp_hdulist['WAVE'].data)
        if drp_dlogl is None:
            drp_dlogl = spec_tools.determine_dlogl(self.drp_logl)
        self.drp_dlogl = drp_dlogl

        self.flux = self.drp_hdulist['FLUX'].data
        self.ivar = self.drp_hdulist['IVAR'].data

    @classmethod
    def from_filenames(cls, drp_fname, dap_fname):
        drp_hdulist = fits.open(drp_fname)
        dap_hdulist = fits.open(dap_fname)
        return cls(drp_hdulist, dap_hdulist)

    def regrid_to_rest(self, template_logl, template_dlogl=None):
        '''
        regrid flux density measurements from MaNGA DRP logcube results
            to a specified logl grid, essentially picking the pixels that
            fall in the logl grid's range, after being de-redshifted

        also normalizes all spectra to have flux density of mean 1

        (this does not perform any fancy interpolation, just "shifting")
        (nor are emission line features masked--that must be done in post-)
        '''
        if template_dlogl is None:
            template_dlogl = spec_tools.determine_dlogl(template_logl)

        if template_dlogl != self.drp_dlogl:
            raise ssp_lib.TemplateCoverageError(
                'template and input spectra must have same dlogl: ' +\
                'template\'s is {}; input spectra\'s is {}'.format(
                    template_dlogl, self.drp_dlogl))

        # determine starting index for each of the spaxels

        template_logl0 = template_logl[0]
        z_map = (self.vel/c.c).to('').value + self.z
        self.z_map = z_map
        template_logl0_z = np.log10(10.**(template_logl0) * (1. + z_map))
        drp_logl_tiled = np.tile(
            self.drp_logl[:, np.newaxis, np.newaxis],
            self.vel.shape)
        template_logl0_z_ = template_logl0_z[np.newaxis, :, :]

        # find the index for the wavelength that best corresponds to
        # an appropriately redshifted wavelength grid
        logl_diff = template_logl0_z_ - drp_logl_tiled
        ix_logl0_z = np.argmin(logl_diff**2., axis=0)

        # test whether wavelength grid extends beyond MaNGA coverage
        # in any spaxels
        bad_logl_extent = (
            ix_logl0_z + len(template_logl)) >= len(self.drp_logl)

        bad_ = np.logical_or(bad_logl_extent, self.vel_mask)
        # select len(template_logl) values from self.flux, w/ diff starting
        # (see http://stackoverflow.com/questions/37984214/
        # pure-numpy-expression-for-selecting-same-length-
        # subarrays-with-different-startin)
        # (and also filter out spaxels that are bad)

        _, I, J = np.ix_(*[range(i) for i in self.flux.shape])

        self.flux_regr = self.flux[
            ix_logl0_z[None, ...] + np.arange(len(template_logl))[
                :, np.newaxis, np.newaxis], I, J]

        self.ivar_regr = self.ivar[
            ix_logl0_z[None, ...] + np.arange(len(template_logl))[
                :, np.newaxis, np.newaxis], I, J]

        self.spax_mask = bad_

        # finally, compute the MW dust attenuation, and apply the inverse
        r_v = 3.1
        mpt = [i/2 for i in self.z_map.shape]
        ix0_mpt = ix_logl0_z[mpt[0], mpt[1]]
        obs_l = 10.**(template_logl[:len(template_logl)]) * \
            (1. + z_map[mpt[0], mpt[1]])
        atten = reddening(
            wave=obs_l * u.AA, a_v=r_v*self.drp_hdulist[0].header['EBVGAL'],
            r_v=r_v, model='f99')
        atten = atten[:, np.newaxis, np.newaxis]

        self.flux_regr /= atten

        return self.flux_regr, self.ivar_regr, self.spax_mask

    def compute_eline_mask(self, template_logl, template_dlogl, ix_Hb=1,
                           half_dv=500.*u.Unit('km/s')):
        '''
        find where EW(Hb) < -5 AA, and return a mask 500 km/s around
            each of the following emission lines:

             - Ha - H11
             - HeI/II 4388/5048/5016
             - [NeIII] 3868/3968
             - [OIII] 4363/4959/5007
             - [NII] 5754/6549/6583
             - [SII] 6717/6731
             - [NeII] 6583/6548/5754
             - [OI] 5577/6300
             - [SIII] 6313/9071/9533
             - [ArIII] 7137/7753
        '''

        EW_Hb = self.eline_EW(ix=ix_Hb)
        eline_dom = EW_Hb > 5.*u.AA # True if elines dominate

        template_l = 10.**template_logl * u.AA

        line_ctrs = u.Unit('AA') * \
            np.array([
                3727.09, 3729.88,
                #[OII]    [OII]
                3751.22, 3771.70, 3798.98, 3836.48, 3890.15, 3971.19,
                #  H12     H11      H10      H9       H8       He
                4102.92, 4341.69, 4862.69, 6564.61, 5008.24, 4960.30,
                #  Hd      Hg       Hb       Ha      [OIII]   [OIII]
                4364.44, 6549.84, 6585.23, 5756.24, 6718.32, 6732.71,
                # [OIII]  [NII]    [NII]    [NII]    [SII]    [SII]
                6302.04, 6313.8, 6585.23, 6549.84, 5756.24, 7137.8, 7753.2,
                # [OI]   [SIII]   [NeII]   [NeII]   [NeII]  [ArIII] [ArIII]
                9071.1, 9533.2, 3889.75, 5877.30, 6679.996, 5017.08
                #[SIII] [SIII]    HeI      HeI      HeI      HeII
            ])

        # compute mask edges
        mask_ledges = line_ctrs * (1 - (half_dv / c.c).to(''))
        mask_uedges = line_ctrs * (1 + (half_dv / c.c).to(''))

        # find whether each wavelength bin is used in for each eline's mask
        antimask = np.row_stack(
            [~((lo < template_l) * (template_l < up))
                for lo, up in izip(mask_ledges, mask_uedges)])
        antimask = np.prod(antimask, axis=0).astype(bool)

        full_mask = np.zeros(
            (len(template_l),) + (self.flux.shape[1:])).astype(bool)

        for (i, j) in product(*tuple(map(range, eline_dom.shape))):
            if eline_dom[i, j] == True:
                full_mask[:, i, j] = ~antimask

        return full_mask

    def eline_EW(self, ix):
        return self.dap_hdulist['EMLINE_EW'].data[ix] * u.Unit('AA')

    # =====
    # properties
    # =====

    @property
    def SB_map(self):
        # RIMG gives nMgy/pix
        return self.drp_hdulist['RIMG'].data * \
            1.0e-9 * m.Mgy / self.spaxel_side**2.

    # =====
    # staticmethods
    # =====

    @staticmethod
    def a_map(f, logl, dlogl):
        lllims = 10.**(logl - 0.5*dlogl)
        lulims = 10.**(logl + 0.5*dlogl)
        dl = (lulims - lllims)[:, np.newaxis, np.newaxis]
        return np.mean(f*dl, axis=0)

cmap = mplcm.get_cmap('cubehelix')
cmap.set_bad('gray')
cmap.set_under('k')
cmap.set_over('w')

class PCA_Result(object):
    '''
    store results of PCA for one galaxy using this
    '''
    def __init__(self, pca, dered, K_obs):
        self.objname = dered.drp_hdulist[0].header['plateifu']
        self.pca = pca
        self.dered = dered

        self.flux_regr, ivar_regr, mask_spax = dered.regrid_to_rest(
            template_logl=pca.logl, template_dlogl=pca.dlogl)
        self.mask_cube = dered.compute_eline_mask(
            template_logl=pca.logl, template_dlogl=pca.dlogl)

        self.mask_map = np.logical_or(
            mask_spax, (dered.drp_hdulist['RIMG'].data) == 0.)

        self.A, self.M, self.a_map, self.O = pca.project_cube(
            f=self.flux_regr, ivar=ivar_regr,
            mask_spax=mask_spax, mask_cube=self.mask_cube)

        self.K_PC = pca.build_PC_cov_full_iter(
            a_map=self.a_map, z_map=dered.z_map, SB_map=dered.SB_map,
            obs_logl=dered.drp_logl, K_obs_=K_obs)

        self.P_PC = StellarPop_PCA.P_from_K(self.K_PC)

        self.map_shape = self.O.shape[-2:]
        self.ifu_ctr_ix = [s/2 for s in self.map_shape]

        self.w = pca.compute_model_weights(P=self.P_PC, A=self.A)

        self.l = 10.**self.pca.logl

    def Mstar_map(self, ax1, ax2, BP, cosmo, z, band='i'):
        '''
        make two-axes stellar-mass map

        use stellar mass-to-light ratio PDF

        params:
         - ax1, ax2: axes for median and stdev, passed along
         - BP: bandpass object
         - d: distance to galaxy (must include units)
         - band: what bandpass to use
        '''

        f = (BP.interps[band](self.l)[:, np.newaxis, np.newaxis] * \
            self.flux_regr).sum(axis=0)
        d = gal_dist(cosmo, z)
        f *= (1.0e-17 * u.Unit('erg s-1 cm-2') * d**2.).to('Lsun').value

        m, s, mcb, scb = self.qty_map(
            ax1=ax1, ax2=ax2, qty_str='ML{}'.format(band),
            f=f, norm=LogNorm())

        return m, s, mcb, scb

    def make_Mstar_fig(self, BP, cosmo, z, band='i'):
        '''
        make stellar-mass figure
        '''

        qty_str = 'Mstar_{}'.format(band)
        qty_tex = r'$M_{{*,{}}}$'.format(band)

        fig = plt.figure(figsize=(9, 4), dpi=300)
        gs = gridspec.GridSpec(1, 2)
        ax1 = wg2.subplot(gs[0], header=self.wcs_header)
        ax2 = wg2.subplot(gs[1], header=self.wcs_header)

        _ = self.Mstar_map(
            ax1=ax1, ax2=ax2, BP=BP, cosmo=cosmo, z=z, band=band)
        fig.suptitle(self.objname + ': ' + qty_tex)

        self.__fix_im_axs__([ax1, ax2])
        plt.tight_layout()

        fig.savefig('{}-{}.png'.format(self.objname, qty_str), dpi=300)

        return fig

    def comp_plot(self, ax1, ax2, ix=None):
        '''
        make plot illustrating fidelity of PCA decomposition in reproducing
            observed data
        '''

        if ix == None:
            ix = self.ifu_ctr_ix

        orig = self.O[:, ix[0], ix[1]] + self.M
        recon = self.pca.M + self.A[:, ix[0], ix[1]].dot(self.pca.PCs)

        # original & reconstructed
        orig_ = ax1.plot(
            self.l,
            np.ma.array(orig, mask=self.mask_cube[:, ix[0], ix[1]]),
            drawstyle='steps-mid', c='b', label='Orig.')
        recon_ = ax1.plot(
            self.l,
            np.ma.array(recon, mask=self.mask_cube[:, ix[0], ix[1]]),
            drawstyle='steps-mid', c='g', label='Recon.')
        ax1.legend(loc='best')
        ax1.set_ylabel(r'$F_{\lambda}$')
        ax1.set_ylim([-0.1, 2.25])
        ax1.set_xticklabels([])

        # residual
        resid = np.abs((orig - recon)/orig)
        resid_ = ax2.plot(
            self.l,
            np.ma.array(resid,
                        mask=self.mask_cube[:, ix[0], ix[1]].astype(bool)),
            drawstyle='steps-mid', c='blue')
        resid_avg_ = ax2.axhline(np.ma.array(
            resid,
            mask=self.mask_cube[:, ix[0], ix[1]].astype(bool)).mean(),
            linestyle='--', c='salmon')

        ax2.set_yscale('log')
        ax2.set_xlabel(r'$\lambda$ [$\textrm{\AA}$]')
        ax2.set_ylabel('Resid.')
        ax2.set_ylim([1.0e-3, 1.0])

        return orig_, recon_, resid_, resid_avg_, ix

    def make_comp_fig(self, ix=None):
        fig = plt.figure(figsize=(8, 3.5), dpi=300)

        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax = plt.subplot(gs[0])
        ax_res = plt.subplot(gs[1])

        _, _, _, _, ix = self.comp_plot(ax1=ax, ax2=ax_res, ix=ix)

        fig.suptitle('{0}: ({1[0]}, {1[1]})'.format(self.objname, ix))

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        fig.savefig('comp_{0}_{1[0]}-{1[1]}.png'.format(
            self.objname, ix), dpi=300)

        return fig

    def qty_map(self, qty_str, ax1, ax2, f=None, norm=None):
        '''
        make a map of the quantity of interest, based on the constructed
            parameter PDF

        params:
         - qty_str: string designating which quantity from self.metadata
            to access
         - ax1: where median map gets shown
         - ax2: where sigma map gets shown
         - f: factor to multiply percentiles by
        '''

        pct_map = self.pca.param_pct_map(
            qty=qty_str, W=self.w, P=np.array([16., 50., 84.]),
            factor=f)

        m = ax1.imshow(
            np.ma.array(pct_map[1, :, :], mask=self.mask_map),
            aspect='equal', norm=norm)

        s = ax2.imshow(
            np.ma.array(
                np.abs(pct_map[2, :, :] - pct_map[0, :, :])/2.,
                mask=self.mask_map),
            aspect='equal', norm=norm)

        mcb = plt.colorbar(m, ax=ax1, shrink=0.6, label='median')
        scb = plt.colorbar(s, ax=ax2, shrink=0.6, label=r'$\sigma$')

        return m, s, mcb, scb

    def make_qty_fig(self, qty_str, qty_tex, qty_fname=None, f=None):
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

        fig = plt.figure(figsize=(9, 4), dpi=300)

        gs = gridspec.GridSpec(1, 2)
        ax1 = wg2.subplot(gs[0], header=self.wcs_header)
        ax2 = wg2.subplot(gs[1], header=self.wcs_header)

        m, s, mcb, scb = self.qty_map(
            qty_str=qty_str, ax1=ax1, ax2=ax2, f=f)

        fig.suptitle('{}: {}'.format(self.objname, qty_tex))

        self.__fix_im_axs__([ax1, ax2])
        plt.tight_layout()
        fig.savefig('{}-{}.png'.format(self.objname, qty_fname), dpi=300)

        return fig

    def qty_hist(self, qty, qty_tex, ix=None, ax=None, f=None):
        if ix is None:
            ix = self.ifu_ctr_ix

        if ax is None:
            ax = plt.gca()

        if f is None:
            f = np.ones_like(self.pca.metadata[qty])

        h = plt.hist(
            self.pca.metadata[qty], weights=self.w, bins=100)
        ax.set_xlabel(qty_tex)
        return h

    def __fix_im_axs__(self, axs):
        '''
        do all the fixes to make quantity maps look nice in pywcsgrid2
        '''
        if type(axs) is not list:
            axs = [axs]

        for ax in axs:
            ax.set_ticklabel_type('delta', center_pixel=self.ifu_ctr_ix)
            ax.grid()
            ax.set_xlabel('')
            ax.set_ylabel('')

    @property
    def wcs_header(self):
        return wcs.WCS(self.dered.drp_hdulist['RIMG'].header)

def setup_pca(K_obs, BP, fname=None, redo=True, pkl=True):
    import pickle
    if (fname is None):
        redo = True

        if pkl == True:
            fname = 'pca.pkl'

    if redo == True:
        pca = StellarPop_PCA.from_YMC(
            lib_para_file='model_spec_bc03/lib_para',
            form_file='model_spec_bc03/input_model_para_for_paper',
            spec_file_dir='model_spec_bc03',
            spec_file_base='modelspec', K_obs=K_obs, BP=BP)
        pca.run_pca_models(q=7)

        if pkl == True:
            pickle.dump(pca, open(fname, 'w'))

    else:
        pca = pickle.load(open(fname, 'r'))

    return pca

def gal_dist(cosmo, z):
    return cosmo.luminosity_distance(z)

if __name__ == '__main__':
    plateifu = '8083-12704'
    cosmo = WMAP9
    K_obs = cov_obs.Cov_Obs.from_fits('cov.fits')
    BP = setup_bandpasses()

    pca = setup_pca(K_obs, BP, fname='pca.pkl', redo=False, pkl=True)

    dered = MaNGA_deredshift.from_filenames(
        drp_fname='/home/zpace/Downloads/manga-{}-LOGCUBE.fits.gz'.format(
            plateifu),
        dap_fname='/home/zpace/mangadap/default/8083/mangadap-{}-default.fits.gz'.format(plateifu))

    pca_res = PCA_Result(pca=pca, dered=dered, K_obs=K_obs)
    pca_res.make_comp_fig()
    pca_res.make_qty_fig(qty_str='MWA', qty_tex=r'$MWA$')

    z_dist = m.get_drpall_val(
        '{}/drpall-{}.fits'.format(
            m.drpall_loc, m.MPL_versions['MPL-4']),
        'nsa_zdist', plateifu)[0]

    pca_res.make_Mstar_fig(BP=BP, cosmo=cosmo, z=z_dist, band='r')

