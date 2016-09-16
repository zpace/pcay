import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib import cm as mplcm
from matplotlib import gridspec

# astropy ecosystem
from astropy import constants as c, units as u, table as t
from astropy.io import fits
from astropy import wcs
from specutils.extinction import reddening
from astropy.cosmology import WMAP9
from astropy import coordinates as coord

import os
import sys

# scipy
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.integrate import quad

# local
import ssp_lib
import cov_obs
import figures_tools

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
        if not dlogl:
            dlogl = np.round(np.mean(self.logl[1:] - self.logl[:-1]), 8)

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
    def from_YMC(cls, base_dir, lib_para_file, form_file,
                 spec_file_base, K_obs, BP):
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

        l_raw_good = (3700. <= l_raw) * (l_raw <= 8700.)
        l_raw = l_raw[l_raw_good]
        dlogl_final = 1.0e-4

        l_final = 10.**np.arange(
            np.log10(3700.), np.log10(5500.), dlogl_final)

        l_full = 10.**np.arange(np.log10(3700.), np.log10(8700.),
                                dlogl_final)

        dl_full = (10.**(np.log10(l_full) + dlogl_final / 2.) -
                   10.**(np.log10(l_full) - dlogl_final / 2.))
        nl_final = len(l_final)
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

        # compute mass to light ratio
        f_r = BP.interps['r'](l_full)

        L_r = (f_r * spec * dl_full).sum(axis=1)
        # reduce wavelength range
        spec = spec[:, :nl_final]

        MLr = metadata['cspm_star'] / L_r

        metadata['Fstar'] = (metadata['mfb_1e9'] + metadata['mf_1e9'].astype(
            float)) / metadata['mf_all']

        metadata = metadata['MWA', 'D4000', 'Hdelta_A', 'Fstar',
                            'zmet', 'Tau_v', 'mu']
        metadata.add_column(t.Column(data=MLr, name='MLr'))

        # set metadata to enable plotting later
        metadata['MWA'].meta['TeX'] = r'MWA'
        metadata['D4000'].meta['TeX'] = r'D4000'
        metadata['Hdelta_A'].meta['TeX'] = r'H$\delta_A$'
        metadata['Fstar'].meta['TeX'] = r'$F^*$'
        metadata['zmet'].meta['TeX'] = r'$\log{\frac{Z}{Z_{\odot}}}$'
        metadata['Tau_v'].meta['TeX'] = r'$\tau_V$'
        metadata['mu'].meta['TeX'] = r'$\mu$'
        metadata['MLr'].meta['TeX'] = r'$(M/L)^*_r$'

        return cls(l=l_final * u.Unit('AA'), trn_spectra=spec,
                   gen_dicts=None, metadata=metadata, dlogl=dlogl_final,
                   K_obs=K_obs)

    # =====
    # methods
    # =====

    def run_pca_models(self, mask_half_dv=500. * u.Unit('km/s'),
                       q=None, max_q=None):
        '''
        run PCA on library of model spectra
        '''

        if (q is None) and (max_q is None):
            raise ValueError('must provide either or both `q`/`max_q`!')
        # first run some prep on the model spectra

        # find lower and upper edges of each wavelength bin,
        # and compute width of bins

        # normalize each spectrum to the mean flux density
        self.a = np.mean(self.trn_spectra, axis=1)

        self.normed_trn = self.trn_spectra / self.a[:, np.newaxis]
        self.M = np.mean(self.normed_trn, axis=0)
        self.S = self.normed_trn - self.M

        # if user asks to test param reconstruction from PCs over range
        # of # of PCs kept
        # this does not keep record of each iteration's output, even at end
        if max_q is not None:
            res_q = [None, ] * max_q
            for dim_pc_subspace in range(1, max_q):
                PCs = self.PCA(self.S, dims=dim_pc_subspace)
                # transformation matrix: spectra -> PC amplitudes
                tfm_sp2PC = PCs.T

                # project back onto the PCs to get the weight vectors
                trn_PC_wts = (self.S).dot(tfm_sp2PC)
                # and reconstruct the best approximation spectra from PCs
                trn_recon = trn_PC_wts.dot(PCs)
                # residuals
                trn_resid = self.S - trn_recon

                self.cov_th = np.cov(trn_resid.T)

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
                    res_q[dim_pc_subspace].fun / dim_pc_subspace,
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
            self.trn_PC_wts = (self.S).dot(self.tfm_sp2PC)
            # and reconstruct the best approximation for the spectra from PCs
            self.trn_recon = self.trn_PC_wts.dot(self.PCs)
            # residuals
            self.trn_resid = self.S - self.trn_recon

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

        # manage masks
        if mask_spax is not None:
            ivar *= (~mask_spax).astype(float)
        if mask_spec is not None:
            ivar *= (~mask_spec[:, np.newaxis, np.newaxis]).astype(float)
        if mask_cube is not None:
            ivar *= (~mask_cube).astype(float)

        # make f and ivar effectively a list of spectra
        f = np.transpose(f, (1, 2, 0)).reshape(-1, cube_shape[0])
        ivar = np.transpose(ivar, (1, 2, 0)).reshape(-1, cube_shape[0])
        ivar[ivar == 0.] = eps

        # normalize by average flux density
        a = np.average(f, weights=ivar, axis=1)
        a[a == 0.] = np.mean(a[a != 0.])
        f = f / a[:, None]
        # get mean spectrum
        S = f - self.M[np.newaxis, :]

        # need to do some reshaping
        A = self.robust_project_onto_PCs(e=self.PCs, f=S, w=ivar)

        A = A.T.reshape((A.shape[1], ) + cube_shape[1:])

        O_norm = S.T.reshape((f.shape[1], ) + cube_shape[1:])

        return A, self.M, a.reshape(cube_shape[-2:]), O_norm

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

    def compute_spec_cov_full(self, a_map, z_map, obs_logl, K_obs_):
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

        K_th = self.cov_th

        # figure out where to start sampling the covariance matrix

        cov_logl = K_obs_.logl
        nlam = K_th.shape[0]
        tem_logl0 = self.logl[0]

        ix0 = self._compute_i0_map(
            tem_logl0=tem_logl0, logl=cov_logl, z_map=z_map)

        # dummy matrix for figuring out starting indices
        N = np.arange(nlam)
        I, J = np.meshgrid(N, N)

        K_obs = K_obs_.cov[
            ix0[..., None, None] + J, ix0[..., None, None] + I]

        K_full = a_map**2. * K_th + K_obs / a_map**2.

        return K_full

    def _compute_PC_cov_spax(self, K_spec):
        '''
        compute the PC covariance matrix for one spaxel

        See Chen+'12 (Eq 11) for full expression
        '''

        E = self.PCs

        return E.dot(K_spec).dot(E.T)

    def _compute_spec_cov_spax(self, K_obs_, i0, a):
        '''
        compute the spectral covariance matrix for one spaxel

        params:
         - K_obs_: observational covariance object
         - i0: starting index of observational covariance matrix
         - a: mean flux-density of original spectra
        '''

        K_th = self.cov_th
        nspec = K_th.shape[0]
        K_obs = K_obs_.cov[i0:(i0 + nspec), i0:(i0 + nspec)]

        K_full = K_obs / a**2. + K_th * a**2.

        return K_full

    def build_PC_cov_full_iter(self, a_map, z_map, obs_logl, K_obs_):
        '''
        for each spaxel, build a PC covariance matrix, and return in array
            of shape (q, q, n, n)

        NOTE: this technique computes each spaxel's covariance matrix
            separately, not all at once! Simultaneous computation requires
            too much memory!

        params:
         - a_map: mean flux-density of original spectra (n by n)
         - z_map: spaxel redshift map (get this from an dered instance)
         - obs_logl: log-lambda vector (1D) of datacube
         - K_obs_: observational spectral covariance object
        '''

        E = self.PCs
        q, l = E.shape
        mapshape = a_map.shape

        # build an array to hold covariances
        K_PC = np.empty((q, q) + mapshape)

        # compute starting indices for covariance matrix
        i0_map = self._compute_i0_map(logl=obs_logl, z_map=z_map)

        inds = np.ndindex(mapshape)  # iterator over physical shape of cube

        for ind in inds:
            K_spec = self._compute_spec_cov_spax(
                K_obs_=K_obs_, i0=i0_map[ind[0], ind[1]],
                a=a_map[ind[0], ind[1]])
            K_PC[..., ind[0], ind[1]] = E.dot(K_spec).dot(E.T)

        return K_PC

    def build_PC_cov_full_mpc(self, a_map, z_map, obs_logl, K_obs_):
        '''
        same as `build_PC_cov_full_iter`, except uses multiprocessing module
        '''

        import multiprocessing as mpc

        # same prep as above
        E = self.PCs
        q, l = E.shape
        mapshape = a_map.shape

        inds = list(np.ndindex(mapshape))  # over physical shape of cube
        i0_map = self._compute_i0_map(logl=obs_logl, z_map=z_map)

        # set up iterators over the map-like dimensions of the arrays
        i0_map_iter = (i0_map[ind[0], ind[1]] for ind in inds)
        a_map_iter = (a_map[ind[0], ind[1]] for ind in inds)

        args_iter = ((self.cov_th,
                      K_obs.cov[i0:(i0 + l), i0:(i0 + l)], E, a)
                     for a, i0 in zip(a_map_iter, i0_map_iter))

        # now set up pool and dummy function
        MAX_PROCESSES = (mpc.cpu_count() - 1) or (1)
        print('processes: ', MAX_PROCESSES)

        with mpc.Pool(processes=MAX_PROCESSES, maxtasksperchild=1) as p:
            pool_outputs = p.starmap(cov_PC_worker, args_iter)

        # now map pool outputs to corresponding elements of K_PC
        K_PC = np.array(pool_outputs)
        K_PC = K_PC.reshape((q, q) + mapshape)

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

        C = self.trn_PC_wts  # C shape: [MODELNUM, PCNUM]
        D = np.abs(C[..., np.newaxis, np.newaxis] - A[np.newaxis, ...])
        # D shape: [MODELNUM, PCNUM, XNUM, YNUM]

        # print(D.shape, P.shape)

        chi2 = np.einsum('cixy,ijxy,cjxy->cxy', D, P, D)
        w = np.exp(-chi2 / 2.)

        return w  # /w.max(axis=0)

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
        W = W[np.isfinite(self.metadata[qty])]

        if factor is None:
            factor = np.ones(cubeshape)

        inds = np.ndindex(*cubeshape)

        A = np.empty((len(P),) + cubeshape)

        for ind in inds:
            q = Q
            w = W[:, ind[0], ind[1]]
            i_ = np.argsort(q, axis=0)
            q, w = q[i_], w[i_]
            A[:, ind[0], ind[1]] = np.interp(
                P, 100. * w.cumsum() / w.sum(), q)

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
        evecs = evecs[:, idx]

        # sort eigenvectors according to same index
        evals = evals[idx]

        # select the first n eigenvectors (n is desired dimension
        # of rescaled data array, or dims_rescaled_data)
        evecs = evecs[:, :dims].T

        # transformation matrix takes (scaled) data to array of PC weights

        return evecs

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
        return 'PCA object: q = {0[0]}, nlam = {0[1]}'.format(self.PCs.shape)

def cov_PC_worker(Kspec_th, Kspec_obs, E, a):
    '''
    worker function that is passed to a Pool
    '''

    Kspec_full = ((Kspec_obs / a**2.) + (Kspec_th * a**2.))

    return E.dot(Kspec_full).dot(E.T)

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
        if not units:
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
            a=l.min(), b=l.max(), epsrel=1.0e-5,
            limit=len(l))[0] * (units['flam'] * units['l']).to('Lsun')
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
                 max_vel_unc=500. * u.Unit('km/s'), drp_dlogl=None,
                 MPL_v='MPL-5'):
        self.drp_hdulist = drp_hdulist
        self.dap_hdulist = dap_hdulist
        self.plateifu = self.drp_hdulist[0].header['PLATEIFU']

        self.vel = dap_hdulist['STELLAR_VEL'].data * u.Unit('km/s')
        self.vel_ivar = dap_hdulist['STELLAR_VEL_IVAR'].data * u.Unit(
            'km-2s2')

        self.z = m.get_drpall_val(
            os.path.join(
                mangarc.manga_data_loc[MPL_v],
                'drpall-{}.fits'.format(m.MPL_versions[MPL_v])),
            ['nsa_z'], self.plateifu)[0]['nsa_z']

        # mask all the spaxels that have high stellar velocity uncertainty
        self.vel_ivar_mask = (1. / np.sqrt(self.vel_ivar)) > max_vel_unc
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

    @classmethod
    def from_plateifu(cls, plate, ifu, MPL_v, kind='SPX-GAU-MILESHC'):
        '''
        load a MaNGA galaxy from a plateifu specification
        '''
        drp_fname = os.path.join(
            mangarc.manga_data_loc[MPL_v], 'drp/', str(plate), 'stack/',
            '-'.join(('manga', str(plate), str(ifu), 'LOGCUBE.fits.gz')))
        dap_fname = os.path.join(
            mangarc.manga_data_loc[MPL_v], 'dap/', kind, str(plate), str(ifu),
            '-'.join(('manga', str(plate), str(ifu), 'MAPS',
                      '{}.fits.gz'.format(kind))))

        if not os.path.isfile(drp_fname):
            raise m.DRP_IFU_DNE_Error(plate, ifu)
        if not os.path.isfile(dap_fname):
            raise m.DAP_IFU_DNE_Error(plate, ifu)

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
                'template and input spectra must have same dlogl: ' +
                'template\'s is {}; input spectra\'s is {}'.format(
                    template_dlogl, self.drp_dlogl))

        # determine starting index for each of the spaxels

        template_logl0 = template_logl[0]
        z_map = (self.vel / c.c).to('').value + self.z
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
        mpt = [i // 2 for i in self.z_map.shape]
        obs_l = 10.**(
            template_logl[:len(template_logl)]) * (1. + z_map[mpt[0], mpt[1]])
        atten = reddening(
            wave=obs_l * u.AA, a_v=r_v * self.drp_hdulist[0].header['EBVGAL'],
            r_v=r_v, model='f99')
        atten = atten[:, None, None]

        self.flux_regr /= atten
        self.ivar_regr *= atten**2.

        return self.flux_regr, self.ivar_regr, self.spax_mask

    def compute_eline_mask(self, template_logl, template_dlogl, ix_eline=7,
                           half_dv=500. * u.Unit('km/s')):

        from elines import (balmer_low, balmer_high, paschen, helium,
                            bright_metal, faint_metal)

        EW = self.eline_EW(ix=ix_eline)
        # thresholds are set very aggressively for debugging, but should be
        # revisited in the future
        # proposed values... balmer_low: 0, balmer_high: 2, helium: 2
        #                    brightmetal: 0, faintmetal: 5, paschen: 10
        add_balmer_low = EW > 0. * u.AA
        add_balmer_high = EW > 0. * u.AA
        add_helium = EW > 0. * u.AA
        add_brightmetal = EW > 0. * u.AA
        add_faintmetal = EW > 0. * u.AA
        add_paschen = EW > 0. * u.AA

        template_l = 10.**template_logl * u.AA

        full_mask = np.zeros((len(template_l),) + EW.shape, dtype=bool)

        for (add_, d) in zip([add_balmer_low, add_balmer_high, add_helium,
                              add_brightmetal, add_faintmetal,
                              add_paschen],
                             [balmer_low, balmer_high, paschen,
                              helium, bright_metal, faint_metal]):

            line_ctrs = np.array(list(d.values())) * u.AA

            # compute mask edges
            mask_ledges = line_ctrs * (1 - (half_dv / c.c).to(''))
            mask_uedges = line_ctrs * (1 + (half_dv / c.c).to(''))

            # is a given wavelength bin within half_dv of a line center?
            mask = np.row_stack(
                [(lo < template_l) * (template_l < up)
                 for lo, up in zip(mask_ledges, mask_uedges)])
            mask = np.any(mask, axis=0)  # OR along axis 0

            full_mask += (mask[:, np.newaxis, np.newaxis] *
                          add_[np.newaxis, ...])

        return full_mask

    def eline_EW(self, ix):
        return self.dap_hdulist['EMLINE_SEW'].data[ix] * u.Unit('AA')

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
        lllims = 10.**(logl - 0.5 * dlogl)
        lulims = 10.**(logl + 0.5 * dlogl)
        dl = (lulims - lllims)[:, np.newaxis, np.newaxis]
        return np.mean(f * dl, axis=0)

cmap = mplcm.get_cmap('cubehelix')
cmap.set_bad('gray')
cmap.set_under('k')
cmap.set_over('w')


class PCA_Result(object):

    '''
    store results of PCA for one galaxy using this
    '''

    def __init__(self, pca, dered, K_obs, z, cosmo):
        self.objname = dered.drp_hdulist[0].header['plateifu']
        self.pca = pca
        self.dered = dered
        self.cosmo = cosmo
        self.z = z

        self.E = pca.PCs

        self.O, self.ivar, mask_spax = dered.regrid_to_rest(
            template_logl=pca.logl, template_dlogl=pca.dlogl)
        self.mask_cube = dered.compute_eline_mask(
            template_logl=pca.logl, template_dlogl=pca.dlogl)

        self.mask_map = np.logical_or(
            mask_spax, (dered.drp_hdulist['RIMG'].data) == 0.)

        self.A, self.M, self.a_map, O_norm = pca.project_cube(
            f=self.O, ivar=self.ivar, mask_spax=mask_spax,
            mask_cube=self.mask_cube)

        # original spectrum
        self.O = np.ma.array(self.O, mask=self.mask_cube)
        self.ivar = np.ma.array(self.ivar, mask=self.mask_cube)

        # how to reconstruct datacube from PC weights cube and PC
        # ij are IFU index, n is eigenspectrum index, l is wavelength index
        self.O_recon = np.ma.array(
            (self.M[:, np.newaxis, np.newaxis] + np.einsum(
                'nij,nl->lij', self.A, self.E)) * self.a_map[np.newaxis, ...],
            mask=self.mask_cube)

        self.resid = np.abs((self.O - self.O_recon) / self.O)

        self.K_PC = pca.build_PC_cov_full_iter(
            a_map=self.a_map, z_map=dered.z_map,
            obs_logl=dered.drp_logl, K_obs_=K_obs)

        self.P_PC = StellarPop_PCA.P_from_K_pinv(self.K_PC)

        self.map_shape = self.O.shape[-2:]
        self.ifu_ctr_ix = [s // 2 for s in self.map_shape]

        self.w = pca.compute_model_weights(P=self.P_PC, A=self.A)

        self.l = 10.**self.pca.logl

    def flux(self, band='i'):
        '''
        return spaxel map of flux in the specified bandpass
        '''

        l_ctr = {'r': 6231. * u.AA, 'i': 7625. * u.AA, 'z': 9134. * u.AA}

        flux_im = (self.dered.drp_hdulist[
            '{}IMG'.format(band)].data * 3.631e-6 * u.Jy).to('erg s-1 cm-2',
            equivalencies=u.spectral_density(l_ctr[band]))

        return flux_im

    def lum(self, band='i'):
        '''
        return spaxel map estimate of luminosity, in Lsun

        Retrieves the correct bandpass image, and converts to Lsun assuming
            some cosmology and redshift
        '''

        flux = self.flux(band=band)

        stellum = (4 * np.pi * flux * (self.dist)**2.).to('Lsun')

        return stellum.value

    def lum_plot(self, ax, band='i'):

        im = ax.imshow(
            np.log10(np.ma.array(self.lum(band=band), mask=self.mask_map)),
            aspect='equal')

        cb = plt.colorbar(im, ax=ax, pad=0.025)
        cb.set_label(r'$\log{\mathcal{L}}$ [$L_{\odot}$]', size=8)
        cb.ax.tick_params(labelsize=8)

        ax.set_title('{}-band luminosity'.format(band), size=8)

        self.__fix_im_axs__(ax)

        return im, cb

    def Mstar_map(self, ax1, ax2, BP, band='i'):
        '''
        make two-axes stellar-mass map

        use stellar mass-to-light ratio PDF

        params:
         - ax1, ax2: axes for median and stdev, passed along
         - BP: bandpass object
         - band: what bandpass to use
        '''

        f = self.lum(band=band)

        m, s, mcb, scb = self.qty_map(
            ax1=ax1, ax2=ax2, qty_str='ML{}'.format(band),
            f=f, norm=[None, None], log=True)

        mstar_tot = np.ma.masked_invalid(np.ma.array(
            self.Mstar_tot(band=band), mask=self.mask_map)).sum()

        ax1.text(x=0.2, y=0.2,
                 s=''.join((r'$\log{\frac{M_{*}}{M_{\odot}}}$ = ',
                            '{:.2f}'.format(np.log10(mstar_tot)))))

        return m, s, mcb, scb

    def make_Mstar_fig(self, BP, band='i'):
        '''
        make stellar-mass figure
        '''

        qty_str = 'Mstar_{}'.format(band)
        qty_tex = r'$M_{{*,{}}}$'.format(band)

        fig = plt.figure(figsize=(9, 4), dpi=300)
        gs = gridspec.GridSpec(1, 2)
        ax1 = fig.add_subplot(gs[0], projection=self.wcs_header_offset)
        ax2 = fig.add_subplot(gs[1], projection=self.wcs_header_offset)

        self.Mstar_map(
            ax1=ax1, ax2=ax2, BP=BP, band=band)
        fig.suptitle(self.objname + ': ' + qty_tex)

        self.__fix_im_axs__([ax1, ax2])

        fig.savefig('{}-{}.png'.format(self.objname, qty_str), dpi=300)

        return fig

    def comp_plot(self, ax1, ax2, ix=None):
        '''
        make plot illustrating fidelity of PCA decomposition in reproducing
            observed data
        '''

        if ix is None:
            ix = self.ifu_ctr_ix

        # original & reconstructed
        orig_ = ax1.plot(
            self.l, self.O[:, ix[0], ix[1]], drawstyle='steps-mid',
            c='b', label='Orig.')
        recon_ = ax1.plot(
            self.l, self.O_recon[:, ix[0], ix[1]], drawstyle='steps-mid',
            c='g', label='Recon.')
        bestfit = self.pca.trn_spectra[np.argmax(self.w[:, ix[0], ix[1]]), :]
        bestfit_ = ax1.plot(
            self.l, 0.5 * bestfit / bestfit.mean(),
            drawstyle='steps-mid', c='c', label='Best Model')
        ax1.legend(loc='best', prop={'size': 6})
        ax1.set_ylabel(r'$F_{\lambda}$')
        ax1.set_ylim([-0.1 * self.O[:, ix[0], ix[1]].mean(),
                      2.25 * self.O[:, ix[0], ix[1]].mean()])
        ax1.set_xticklabels([])

        # inverse-variance (weight) plot
        ax1_ivar = ax1.twinx()
        ax1_ivar.set_yticklabels([])
        ivar_ = ax1_ivar.plot(
            self.l, self.ivar[:, ix[0], ix[1]], drawstyle='steps-mid',
            c='m', label='Wt.')

        # residual
        resid_ = ax2.plot(
            self.l, self.resid[:, ix[0], ix[1]], drawstyle='steps-mid',
            c='blue')
        resid_avg_ = ax2.axhline(
            self.resid[:, ix[0], ix[1]].mean(), linestyle='--', c='salmon')

        ax2.set_yscale('log')
        ax2.set_xlabel(r'$\lambda$ [$\textrm{\AA}$]')
        ax2.set_ylabel('Resid.')
        ax2.set_ylim([1.0e-3, 1.0])

        return orig_, recon_, bestfit_, ivar_, resid_, resid_avg_, ix

    def make_comp_fig(self, ix=None):
        fig = plt.figure(figsize=(8, 3.5), dpi=300)

        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax = fig.add_subplot(gs[0])
        ax_res = fig.add_subplot(gs[1])

        _, _, _, _, _, ix = self.comp_plot(ax1=ax, ax2=ax_res, ix=ix)

        fig.suptitle('{0}: ({1[0]}, {1[1]})'.format(self.objname, ix))

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        fig.savefig('comp_{0}_{1[0]}-{1[1]}.png'.format(
            self.objname, ix), dpi=300)

        return fig

    def qty_map(self, qty_str, ax1, ax2, f=None, norm=[None, None],
                log=False):
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

        pct_map = self.pca.param_pct_map(
            qty=qty_str, W=self.w, P=np.array([16., 50., 84.]),
            factor=f)

        if log:
            pct_map = np.log10(pct_map)

        m = ax1.imshow(
            np.ma.array(pct_map[1, :, :], mask=self.mask_map),
            aspect='equal', norm=norm[0])

        s = ax2.imshow(
            np.ma.array(
                np.abs(pct_map[2, :, :] - pct_map[0, :, :]) / 2.,
                mask=self.mask_map),
            aspect='equal', norm=norm[1])

        mcb = plt.colorbar(m, ax=ax1, pad=0.)
        mcb.set_label('med.', size=8)
        mcb.ax.tick_params(labelsize=8)

        scb = plt.colorbar(s, ax=ax2, pad=0.)
        scb.set_label('unc.', size=8)
        scb.ax.tick_params(labelsize=8)

        return m, s, mcb, scb

    def map_add_loc(self, ax, ix, **kwargs):
        '''
        add axvline and axhline at the location in the map coresponding to
            some image-frame indices ix
        '''

        from astropy.wcs.utils import pixel_to_skycoord
        from astropy.coordinates import SkyCoord

        pix_coord = self.wcs_header_offset.all_pix2world(
            np.atleast_2d(ix), origin=1)

        ax.axhline(pix_coord[1], **kwargs)
        ax.axvline(pix_coord[0], **kwargs)

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
        ax1 = fig.add_subplot(gs[0], projection=self.wcs_header_offset)
        ax2 = fig.add_subplot(gs[1], projection=self.wcs_header_offset)

        m, s, mcb, scb = self.qty_map(
            qty_str=qty_str, ax1=ax1, ax2=ax2, f=f)

        fig.suptitle('{}: {}'.format(self.objname, qty_tex))

        self.__fix_im_axs__([ax1, ax2])
        # plt.tight_layout()
        fig.savefig('{}-{}.png'.format(self.objname, qty_fname), dpi=300)

        return fig

    def qty_hist(self, qty, qty_tex, ix=None, ax=None, f=None, bins=50,
                 legend=False):
        if ix is None:
            ix = self.ifu_ctr_ix

        if ax is None:
            ax = plt.gca()

        if f is None:
            f = np.ones_like(self.pca.metadata[qty])

        q = self.pca.metadata[qty]
        w = self.w[:, ix[0], ix[1]]
        q, w = q[np.isfinite(q)], w[np.isfinite(q)]

        if len(q) == 0:
            return None

        # marginalized posterior
        h = ax.hist(
            q, weights=w, bins=bins, normed=True, histtype='step',
            color='k', label='posterior')
        # marginalized prior
        hprior = ax.hist(
            q, bins=bins, normed=True, histtype='step', color='b', alpha=0.5,
            label='prior')
        ax.set_xlabel(qty_tex)
        if legend:
            ax.legend(loc='best')
        return h, hprior

    def orig_spax(self, ixx, ixy):
        return self.O[:, ixx, ixy]

    def recon_spax(self, ixx, ixy):
        return self.O_recon[:, ixx, ixy]

    def ivar_spax(self, ixx, ixy):
        return self.ivar[:, ixx, ixy]

    def param_vals_wts(self, ixx, ixy, pname):
        return np.array(self.pca.metadata[pname]), self.w[:, ixx, ixy]

    def __fix_im_axs__(self, axs):
        '''
        do all the fixes to make quantity maps look nice in wcsaxes
        '''
        if type(axs) is not list:
            axs = [axs]

        # over ax objects
        for ax in axs:
            # over XOFFSET & YOFFSET
            for i in range(2):
                ax.coords[i].set_major_formatter('x')
                ax.coords[i].set_ticks(spacing=5.*u.arcsec)
                ax.coords[i].set_format_unit(u.arcsec)

    def make_full_QA_fig(self, BP, ix=None):
        '''
        use matplotlib to make a full map of the IFU grasp, including
            diagnostic spectral fits, and histograms of possible
            parameter values for each spaxel
        '''
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
            hspace=0., wspace=.1, left=.05, right=.95)

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
        lumim, lcb = self.lum_plot(im_ax, band='r')
        # self.map_add_loc(ix=ix_, ax=im_ax, color='gray', linewidth=1.,
        #                  linestyle=':')

        # loop through parameters of interest, and make a weighted
        # histogram for each parameter
        enum_ = enumerate(zip(gs2, self.pca.metadata.colnames, TeX_labels))
        for i, (gs_, q, tex) in enum_:
            ax = fig.add_subplot(gs_)
            if 'ML' in q:
                bins = np.logspace(-1, 6, 50)
                ax.set_xscale('log')
            else:
                bins = 50
            if i == 0:
                legend = True
            else:
                legend = False
            h_, hprior_ = self.qty_hist(
                qty=q, qty_tex=tex, ix=ix, ax=ax, bins=bins, legend=legend)
            ax.tick_params(axis='both', which='major', labelsize=10)

        plt.suptitle('{0}: ({1[0]}-{1[1]})'.format(self.objname, ix_))

        plt.savefig('{0}_fulldiag_{1[0]}-{1[1]}.png'.format(
            self.objname, ix_))

    @property
    def wcs_header(self):
        return wcs.WCS(self.dered.drp_hdulist['RIMG'].header)

    @property
    def wcs_header_offset(self):
        return figures_tools.linear_offset_coordinates(
            self.wcs_header, coord.SkyCoord(
                *(self.wcs_header.wcs.crval * u.deg)))

    def Mstar_tot(self, band='r'):
        qty_str = 'ML{}'.format(band)
        f = self.lum(band=band)

        pct_map = self.pca.param_pct_map(
            qty=qty_str, W=self.w, P=np.array([50.]),
            factor=f)[0, ...]

        return pct_map

    @property
    def dist(self):
        return gal_dist(self.cosmo, self.z)

    def Mstar_surf(self, band='r'):
        spaxel_psize = (self.dered.spaxel_side * self.dist).to(
            'kpc', equivalencies=u.dimensionless_angles())
        # print spaxel_psize
        sig = self.Mstar(band=band) * u.Msun / spaxel_psize**2.
        return sig.to('Msun pc-2').value


def setup_pca(BP, fname=None, redo=False, pkl=True, q=7):
    import pickle
    if (fname is None):
        redo = True

        if pkl:
            fname = 'pca.pkl'

    if redo:
        K_obs = cov_obs.Cov_Obs.from_fits('manga_Kspec.fits')
        pca = StellarPop_PCA.from_YMC(
            base_dir=mangarc.BC03_CSP_loc,
            lib_para_file='lib_para',
            form_file='input_model_para_for_paper',
            spec_file_base='modelspec', K_obs=K_obs, BP=BP)
        pca.run_pca_models(q=q)

        if pkl:
            pickle.dump(pca, open(fname, 'wb'))

    else:
        K_obs = cov_obs.Cov_Obs.from_fits('manga_Kspec.fits')
        pca = pickle.load(open(fname, 'rb'))

    return pca, K_obs


class Test_PCA_Result(PCA_Result):
    def __init__(self, pca, K_obs, cosmo, fake_ifu, objname='', z=0.):
        self.objname = objname
        self.pca = pca
        self.cosmo = cosmo
        self.z = z

        self.E = pca.PCs

        self.O = fake_ifu.make_datacube()
        self.ivar = fake_ifu.ivar

        self.mask_map = np.zeros_like(self.ivar, dtype=bool)
        mask_spax = np.zeros_like(self.ivar[0, ...])

        self.A, self.M, self.a_map, O_norm = pca.project_cube(
            f=self.O, ivar=self.ivar, mask_spax=mask_spax,
            mask_cube=self.mask_cube)

        # original spectrum
        self.O = np.ma.array(self.O, mask=self.mask_cube)
        self.ivar = np.ma.array(self.ivar, mask=self.mask_cube)

        # how to reconstruct datacube from PC weights cube and PC
        # ij are IFU index, n is eigenspectrum index, l is wavelength index
        self.O_recon = np.ma.array(
            (self.M[:, np.newaxis, np.newaxis] + np.einsum(
                'nij,nl->lij', self.A, self.E)) * self.a_map[np.newaxis, ...],
            mask=self.mask_cube)

        self.resid = np.abs((self.O - self.O_recon) / self.O)

        self.K_PC = pca.build_PC_cov_full_iter(
            a_map=self.a_map, z_map=dered.z_map,
            obs_logl=dered.drp_logl, K_obs_=K_obs)

        self.P_PC = StellarPop_PCA.P_from_K_pinv(self.K_PC)

        self.map_shape = self.O.shape[-2:]
        self.ifu_ctr_ix = [s // 2 for s in self.map_shape]

        self.w = pca.compute_model_weights(P=self.P_PC, A=self.A)

        self.l = 10.**self.pca.logl


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

if __name__ == '__main__':
    cosmo = WMAP9
    BP = setup_bandpasses()

    mpl_v = 'MPL-5'

    plateifu = input('What galaxy? ')

    if not plateifu:
        plateifu = '8083-12704'

    pca, K_obs = setup_pca(BP, fname='pca.pkl', redo=False, pkl=True, q=7)

    plate, ifu = plateifu.split('-')

    dered = MaNGA_deredshift.from_plateifu(
        plate=int(plate), ifu=int(ifu), MPL_v=mpl_v)

    z_dist = m.get_drpall_val(
        os.path.join(
            mangarc.manga_data_loc[mpl_v],
            'drpall-{}.fits'.format(m.MPL_versions[mpl_v])),
        ['nsa_zdist'], plateifu)[0]['nsa_zdist']

    pca_res = PCA_Result(
        pca=pca, dered=dered, K_obs=K_obs, z=z_dist, cosmo=cosmo)
    pca_res.make_qty_fig(qty_str='MWA', qty_tex=r'$MWA$')
    pca_res.make_full_QA_fig(BP=BP)

    pca_res.make_Mstar_fig(BP=BP, band='r')
