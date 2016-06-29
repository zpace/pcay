import numpy as np
import matplotlib.pyplot as plt

from astropy import constants as c, units as u, table as t
from astropy.io import fits

import os
from scipy.interpolate import interp1d

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
        self.gen_dicts = gen_dicts

    @classmethod
    def from_YMC(cls, metadata_file, spec_file_dir, spec_file_base):
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

        # load metadata table
        metadata = t.Table.read(metadata_file, format='ascii')
        nspec = len(metadata)

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
            finally:
                hdulist.close()

            if goodspec[i] == False:
                continue

            spec[i] = interp1d(l_raw, f_lambda)(l_final)

        ixs = np.arange(nspec)
        metadata.remove_rows(ixs[~goodspec])
        spec = spec[goodspec, :]

        metadata['Fstar'] = metadata['mfb_1e9'] / metadata['mgalaxy']

        metadata = metadata['MWA', 'LrWA', 'D4000', 'Hdelta_A', 'Fstar']

        return cls(l=l_final*u.Unit('AA'), trn_spectra=spec,
                   gen_dicts=None, metadata=metadata, dlogl=dlogl_final)

    # =====
    # methods
    # =====

    def run_pca_models(self, mask_half_dv=500.*u.Unit('km/s')):
        '''
        run PCA on library of model spectra
        '''
        # first run some prep on the model spectra

        # find lower and upper edges of each wavelength bin,
        # and compute width of bins
        l_lower = 10.**(self.logl - self.dlogl/2.)
        l_upper = 10.**(self.logl + self.dlogl/2.)
        dl = self.dl = l_upper - l_lower

        # scale each spectrum such that the mean flux between
        # 3700 and 5500 AA is unity
        avg_trn_flux = np.mean(self.trn_spectra * dl, axis=1)

        self.normed_trn_spectra = \
            self.trn_spectra/avg_trn_flux[:, np.newaxis]
        self.mean_trn_spectrum = np.mean(
            self.normed_trn_spectra, axis=0)

        self.PCs = self.PCA(
            self.normed_trn_spectra - self.mean_trn_spectrum,
            dims=7)
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



    def robust_project_onto_PCs(self, spec, ivar=None):
        '''
        project a set of measured spectra with measurement errors onto
            the principal components of the model library

        params:
         - spec: n-by-m array, where n is the number of spectra, and m
            is the number of spectral wavelength bins. [Flux-density units]
         - ivar: n-by-m array, containing the inverse-variances for each
            of the rows in spec [Flux-density units]
            (this functions like the array w_lam in REF [1])

        REFERENCE:
            [1] Connolly & Szalay (1999, AJ, 117, 2052)
                [particularly eqs. 3, 4, 5]
                (http://iopscience.iop.org/article/10.1086/300839/pdf)
        '''
        if not hasattr(self, 'PCs'):
            raise PCAError('must run PCA before projecting!')

        e = self.PCs

        if ivar is None:
            ivar = np.ones_like(spec)

        a = np.empty(spec.shape[0])
        for i in range(a):
            M = self.make_M(ivar[i], e)
            F = self.make_F(ivar, spec[i], e)
            a[i] = np.linalg.inv(M)[i,:] * F

    # =====
    # properties
    # =====

    @property
    def model_eline_mask(self):
        from astropy import units as u, constants as c
        half_dv = self.mask_half_dv
        line_ctrs = u.Unit('AA') * \
            np.array([3727.09, 3729.88, 3889.05, 3969.81, 3968.53,
        #              [OII]    [OII]      H8    [NeIII]  [NeIII]
                      4341.69, 4102.92, 4862.69, 5008.24, 4960.30])
        #                Hg       Hd       Hb     [OIII]   [OIII]

        # compute mask edges
        mask_ledges = line_ctrs * (1 - (half_dv / c.c).to(''))
        mask_uedges = line_ctrs * (1 + (half_dv / c.c).to(''))

        # find whether each wavelength bin is used in for each eline's mask
        full_antimask = np.row_stack(
            [~((lo < self.l) * (self.l < up))
                for lo, up in izip(mask_ledges, mask_uedges)])
        antimask = np.prod(full_antimask, axis=0)
        return ~antimask.astype(bool)

    # =====
    # staticmethods
    # =====

    @staticmethod
    def __obj_fn_Pfit__(P, C, X, Z):
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
            - X: q-by-p array, which takes C.T to an estimate of
                the true P, P_
            - Z: length-p vector, representing the zeropoint associated
                with each parameter P[:, i]

        C.T (dot) X must have the same dimensions as P
        '''

        P_ = np.dot(C.T, X) + Z
        P_resid_sq = (((P_ - P).sum(axis=1))**2.).sum()
        return P_resid_sq

    @staticmethod
    def make_M(w, e):
        '''
        take weights `w` and eigenvectors `e`, and create matrix M,
            where element M[i,j] = (e[i] * e[j] * w).sum()
        '''
        dim = (len(e[0]), )*2
        M = np.empty(dim)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M[i, j] = (e[i] * e[j] * w).sum()

        return M

    @staticmethod
    def make_F(w, f, e):
        '''
        take weights `w`, observed spectrum `f`, and eigenvectors `e`
            and create vector F, where F[j] = (w * f * e[j]).sum()
        '''
        dim = e.shape[0]
        F = np.empty(dim)
        for j in range(dim):
            F[j] = (w * f * e[j]).sum()

        return F

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

        self.regridded_cube = self.flux[
            ix_logl0_z[None, ...] + np.arange(len(template_logl))[
                :, np.newaxis, np.newaxis], I, J]

        self.bad_ = bad_

        self.regridded_cube[:, bad_] = 0.

        return self.regridded_cube, self.bad_
