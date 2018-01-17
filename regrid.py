import numpy as np

import utils as ut

from importer import *
import manga_tools as m

from dered_drizzle import drizzle_flux

def _nearest(loglgrid, loglrest, frest, ivarfrest):
    '''
    workhorse integer-pixel function
    '''

    # how to index full cube
    L, I, J = np.ix_(*[range(i) for i in frest.shape])

    # reference index and corresponding loglam **IN GRID**
    # this avoids latching onto endpoints erroneously
    grid_nl = len(loglgrid)
    ix_ref_grid = (grid_nl - 1) // 2
    logl_ref_grid = loglgrid[ix_ref_grid]

    ix_ref_rest = np.argmin(np.abs(loglrest - logl_ref_grid), axis=0)
    logl_ref_rest = loglrest[ix_ref_rest, I, J].squeeze()

    NX, NY = ix_ref_rest.shape
    xctr, yctr = NX // 2, NY // 2

    dlogl_resid = (logl_ref_rest - logl_ref_grid)

    '''
    set up what amounts to a linear mapping from rest to rectified,
        from: pixel u of spaxel I, J of rest-frame cube
        to:   pixel u' of spaxel I', J' of rectified cube

    in reality pixel ix_ref_grid - 1 <==> ix_ref_rest - 1
                     ix_ref_grid     <==> ix_ref_rest
                     ix-ref_grid + 1 <==> ix_ref_rest + 1
                     etc...
    '''

    def rect2rest(i0_rect, i0_rest, i_rect):
        '''
        maps pixel number (rectified frame) to pixel number (rest frame),
            assuming that each pixel is the same width
        '''
        # delta no. pix from reference in rect frame
        d_rect = i_rect - i0_rect
        return i0_rest + d_rect + 0 # should this be -1, 0, or 1?

    # create array of individual wavelength indices in rectified array
    l_ixs_rect = np.linspace(0, grid_nl - 1, grid_nl, dtype=int)[:, None, None]
    l_ixs_rest = rect2rest(ix_ref_grid, ix_ref_rest, l_ixs_rect)
    # where are we out of range?
    badrange = np.logical_or.reduce(((l_ixs_rest >= grid_nl),
                                     (l_ixs_rest < 0)))
    #l_ixs_rest[badrange] = 0

    flux_regr = frest[l_ixs_rest, I, J]
    ivar_regr = ivarfrest[l_ixs_rest, I, J]

    #ivar_regr[badrange] = 0.

    return flux_regr, ivar_regr, dlogl_resid

class Regridder(object):
    '''
    place regularly-sampled array onto another grid
    '''

    methods = ['nearest', 'invdistwt', 'interp', 'supersample', 'drizzle']

    def __init__(self, loglgrid, loglrest, frest, ivarfrest, dlogl=1.0e-4):
        self.loglgrid = loglgrid
        self.loglrest = loglrest
        self.frest = frest
        self.ivarfrest = ivarfrest

        self.dlogl = dlogl

    def nearest(self, **kwargs):
        '''
        integer-pixel deredshifting
        '''

        loglgrid = self.loglgrid
        loglrest = self.loglrest
        frest = self.frest
        ivarfrest = self.ivarfrest

        flux_regr, ivar_regr, *_ = _nearest(
            loglgrid=loglgrid, loglrest=loglrest,
            frest=frest, ivarfrest=ivarfrest)

        return flux_regr, ivar_regr

    def drizzle(self, **kwargs):
        '''
        drizzle flux between pixels (wraps something similar to Carnall 2017)
        '''

        flux_regr, ivar_regr = drizzle_flux(
            grid_ctr=self.loglgrid, rest_ctr=self.loglrest, wave_lin=False,
            flux_cube=self.frest, ivar_cube=self.ivarfrest)

        return flux_regr, ivar_regr

    def interp(self, **kwargs):
        loglgrid = self.loglgrid
        loglrest = self.loglrest
        frest = self.frest
        ivarfrest = self.ivarfrest
        dlogl = self.dlogl

    def supersample(self, nper=2, **kwargs):
        '''
        regrid from rest to fixed frame by supersampling
        '''
        loglgrid = self.loglgrid
        loglrest = self.loglrest
        frest = self.frest
        ivarfrest = self.ivarfrest
        dlogl = self.dlogl
        nlrest, NY, NX = loglrest.shape
        (nlgrid, ) = loglgrid.shape
        mapshape = (NY, NX)

        # dummy offset array
        offset_ = lin_transform(
            x=np.linspace(-.5 + .5 / nper, .5 - .5 / nper, nper,
                          dtype=np.float32),
            r1=[-.5, .5], r2=[-dlogl, dlogl])
        # construct supersampled logl array
        loglrest_super = loglrest.repeat(nper, axis=0) + \
                         offset_.repeat(nlrest, axis=0)[:, None, None]

        lrest_super = 10.**loglrest_super
        dlrest_super = determine_dl(loglrest_super)
        frest_super = frest.repeat(nper, axis=0)
        frest_integ_super = frest_super * dlrest_super

        ivarfrest_super = ivarfrest.repeat(nper, axis=0) / nper

        # now here's the tricky bit...
        # first zero out entries of flam_obs_rest_integ_super that are out of range
        obs_in_range = np.logical_and(
            loglrest_super >= loglgrid.min() - dlogl / 2.,
            loglrest_super <= loglgrid.max() + dlogl / 2.)
        frest_integ_super[~obs_in_range] = 0.

        # set up empty dummy arrays for flux and variance
        flam_obs_rest_integ_regr = np.empty((len(loglgrid), ) + mapshape)
        ivar_obs_rest_regr = np.empty((len(loglgrid), ) + mapshape)

        # set up bin edges for histogramming
        bin_edges = np.concatenate(
            [loglgrid - dlogl / 2., loglgrid[-1:] + dlogl / 2.])
        low_edge = bin_edges[0]

        # figure out what index (in each spaxel of supersampled array)
        # is the first to be assigned
        ix0s = np.argmin(np.abs(loglrest_super - low_edge),
                         axis=0)

        # iterate through indices, and assign fluxes & variances
        for i, j in np.ndindex(*mapshape):
            ix0 = ix0s[i, j]
            if loglrest_super[ix0, i, j] < low_edge:
                ix0 += 1

            # select flux elements in range, reshape, and sum
            _spax = frest_integ_super[
                ix0:ix0 + nlgrid * nper, i, j].reshape(
                    (-1, nper)).sum(axis=1)
            flam_obs_rest_integ_regr[:, i, j] = _spax

            # select ivar elements
            _spax = ivarfrest_super[
                ix0:ix0 + nlgrid * nper, i, j].reshape(
                    (-1, nper)).sum(axis=1)
            ivar_obs_rest_regr[:, i, j] = _spax

        # and divide by dl to get a flux density
        flam_obs_rest_regr = flam_obs_rest_integ_regr / determine_dl(
            loglgrid)[:, None, None]

        return flam_obs_rest_regr, ivar_obs_rest_regr

    def supersample_vec(self, nper=2, **kwargs):
        '''
        regrid from rest to fixed frame by supersampling
        '''
        loglgrid = self.loglgrid
        loglrest = self.loglrest
        frest = self.frest
        ivarfrest = self.ivarfrest
        dlogl = self.dlogl
        nlrest, NY, NX = loglrest.shape
        (nlgrid, ) = loglgrid.shape
        mapshape = (NY, NX)

        # dummy offset array
        offset_ = lin_transform(
            x=np.linspace(-.5 + .5 / nper, .5 - .5 / nper, nper,
                          dtype=np.float32),
            r1=[-.5, .5], r2=[-dlogl, dlogl])
        # construct supersampled logl array
        loglrest_super = loglrest.repeat(nper, axis=0) + \
                         offset_.repeat(nlrest, axis=0)[:, None, None]

        lrest_super = 10.**loglrest_super
        dlrest_super = determine_dl(loglrest_super)
        frest_super = frest.repeat(nper, axis=0)
        frest_integ_super = frest_super * dlrest_super

        ivarfrest_super = ivarfrest.repeat(nper, axis=0) / nper

        # set up bin edges for histogramming
        bin_edges = np.concatenate(
            [loglgrid - dlogl / 2., loglgrid[-1:] + dlogl / 2.])
        low_edge = bin_edges[0]

        # figure out what index (in each spaxel of supersampled array)
        # is the first to be assigned
        ix0s = np.argmin(np.abs(loglrest_super - low_edge),
                         axis=0)

        L, I, J = np.ix_(*[range(i) for i in frest_super.shape])

        start_too_low = (loglrest_super[ix0s, I, J].squeeze() < low_edge)
        ix0s[start_too_low] += 1

        # select correct elements
        # set up empty dummy arrays for flux and variance
        assign = ix0s[None, :, :] + np.linspace(
            0, nlgrid * nper - 1, nlgrid * nper, dtype=int)[:, None, None]
        flam_obs_rest_integ_super = frest_integ_super[assign, I, J].reshape(
            (nlgrid, nper) + mapshape)
        ivar_obs_rest_super = ivarfrest_super[assign, I, J].reshape(
            (nlgrid, nper) + mapshape)

        flam_obs_rest_integ_regr = flam_obs_rest_integ_super.sum(axis=1)
        ivar_obs_rest_regr = ivar_obs_rest_super.sum(axis=1)

        # and divide by dl to get a flux density
        flam_obs_rest_regr = flam_obs_rest_integ_regr / determine_dl(
            loglgrid)[:, None, None]

        return flam_obs_rest_regr, np.sqrt(nper) * ivar_obs_rest_regr
