import numpy as np
import matplotlib.pyplot as plt

from astropy import constants as c, units as u, table as t
from astropy.io import fits

import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize

import spec_tools
import ssp_lib
import manga_tools as m

from itertools import izip, product
from glob import glob

class StellarPop_PCA(object):
    '''
    class for determining PCs of a library of synthetic spectra
    '''
    def __init__(self, l, trn_spectra, gen_dicts, metadata, dlogl=None):
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

    @classmethod
    def from_YMC(cls, lib_para_file, form_file,
                 spec_file_dir, spec_file_base):
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
        nl_final = len(l_final)

        # load metadata tables
        lib_para = t.Table.read(lib_para_file, format='ascii')
        form_data = t.Table.read(form_file, format='ascii')
        form_data_goodcols = ['zmet', 'Tau_v', 'mu']
        for n in form_data_goodcols:
            lib_para[n] = np.zeros(len(lib_para))

        nspec = len(lib_para)

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
                lib_para[form_data_goodcols][i-1] = \
                    form_data[form_data_goodcols][i]
            finally:
                hdulist.close()

            if goodspec[i] == False:
                continue

            spec[i] = interp1d(l_raw, f_lambda)(l_final)

        metadata = lib_para

        ixs = np.arange(nspec)
        metadata.remove_rows(ixs[~goodspec])
        spec = spec[goodspec, :]

        metadata['Fstar'] = metadata['mfb_1e9'] / metadata['mgalaxy']

        metadata = metadata['MWA', 'LrWA', 'D4000', 'Hdelta_A', 'Fstar',
                            'zmet', 'Tau_v', 'mu']

        return cls(l=l_final*u.Unit('AA'), trn_spectra=spec,
                   gen_dicts=None, metadata=metadata, dlogl=dlogl_final)

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

        # scale each spectrum such that the mean flux between
        # 3700 and 5500 AA is unity
        avg_trn_flux = np.mean(self.trn_spectra * dl, axis=1)

        self.normed_trn_spectra = \
            self.trn_spectra/avg_trn_flux[:, np.newaxis]
        self.mean_trn_spectrum = np.mean(
            self.normed_trn_spectra, axis=0)

        # if user asks to test param reconstruction from PCs over range
        # of # of PCs kept
        # this does not keep record of each iteration's output, even at end
        if max_q is not None:
            res_q = [None, ] * max_q
            for dim_pc_subspace in range(1, max_q):
                PCs = self.PCA(
                    self.normed_trn_spectra - self.mean_trn_spectrum,
                    dims=dim_pc_subspace)
                # transformation matrix: spectra -> PC amplitudes
                tfm_sp2PC = PCs.T

                # project back onto the PCs to get the weight vectors
                trn_PC_wts = (self.normed_trn_spectra - \
                    self.mean_trn_spectrum).dot(tfm_sp2PC)
                # and reconstruct the best approximation for the spectra from PCs
                trn_recon = trn_PC_wts.dot(PCs)
                # residuals
                trn_resid = self.normed_trn_spectra - \
                    (self.mean_trn_spectrum + trn_recon)

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
                    res_q[dim_pc_subspace].fun/res_q[dim_pc_subspace].x.size,
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
                self.normed_trn_spectra - self.mean_trn_spectrum,
                dims=dim_pc_subspace)
            # transformation matrix: spectra -> PC amplitudes
            self.tfm_sp2PC = self.PCs.T

            # project back onto the PCs to get the weight vectors
            self.trn_PC_wts = (self.normed_trn_spectra - \
                self.mean_trn_spectrum).dot(self.tfm_sp2PC)
            # and reconstruct the best approximation for the spectra from PCs
            self.trn_recon = self.trn_PC_wts.dot(self.PCs)
            # residuals
            self.trn_resid = self.normed_trn_spectra - \
                (self.mean_trn_spectrum + self.trn_recon)

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

    def project_cube(f, ivar, mask_spax=None, mask_spec=None,
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
        f = np.transpose(f, (1,2,0)).reshape(-1, f.shape[0])
        ivar = np.transpose(ivar, (1,2,0)).reshape(-1, ivar.shape[0])

        # normalize flux array
        f /= np.mean(f*dl[:, np.newaxis, np.newaxis], axis=0)
        f -= np.mean(f, axis=0)

        # need to do some reshaping
        A = self.robust_project_onto_PCs(e=self.PCs, f=f, w=ivar)

        A = A.T.reshape((self.PCs.shape[1],) + cube_shape[1:])

        return A

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
    def __init__(self, drp_hdulist, dap_hdulist,
                 max_vel_unc=90.*u.Unit('km/s'), drp_dlogl=None,
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
        bad_ = (bad_logl_extent | self.vel_ivar_mask)[np.newaxis:, :]
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

if __name__ == '__main__':
    dered = MaNGA_deredshift.from_filenames(
        drp_fname='/home/zpace/Downloads/manga-8083-12704-LOGCUBE.fits.gz',
        dap_fname='/home/zpace/mangadap/default/8083/mangadap-8083-12704-default.fits.gz')
    pca = StellarPop_PCA.from_YMC(
        lib_para_file='model_spec_bc03/lib_para',
        form_file='model_spec_bc03/input_model_para_for_paper',
        spec_file_dir='model_spec_bc03',
        spec_file_base='modelspec')
    pca.run_pca_models(q=7)

    flux_regr, ivar_regr, mask_spax = dered.regrid_to_rest(
        template_logl=pca.logl, template_dlogl=pca.dlogl)
    mask_cube = dered.compute_eline_mask(
        template_logl=pca.logl, template_dlogl=pca.dlogl)

    A = pca.project_cube(f=flux_regr)#, ivar=ivar_regr)
        #f=flux_regr, ivar=ivar_regr)#,
        #mask_spax=mask_spax, mask_cube=mask_cube)
