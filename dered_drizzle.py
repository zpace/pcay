import numpy as np
from numpy.lib.stride_tricks import as_strided

from astropy.io import fits
from astropy import units as u, constants as c
from speclite import redshift as slrs

import os, sys

import utils as ut

from importer import *

import manga_tools as m

def get_cube_lamgrid(plate, ifu, mpl_v='MPL-6'):
    hdulist = m.load_drp_logcube(plate, ifu, mpl_v)
    lamctrgrid = hdulist['WAVE'].data
    hdulist.close()
    return lamctrgrid

def get_evec_lamgrid():
    hdulist = fits.open('pc_vecs.fits')
    lamctrgrid = hdulist['LAM'].data
    hdulist.close()
    return lamctrgrid

def rolling_window(a, window, axis=-1):
    '''
    Return a windowed array.

    Parameters:
        a: A numpy array
        window: The size of the window

    Returns:
        An array that has been windowed
    '''

    if axis == -1:
      axis = a.ndim - 1

    if 0 <= axis <= a.ndim - 1:
        shape = (a.shape[:axis] + (a.shape[axis] - window + 1, window) +
                 a.shape[axis+1:])
        strides = a.strides[:axis] + (a.strides[axis],) + a.strides[axis:]
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    else:
        raise ValueError('rolling_window: axis out of bounds')

def find_redshift_bounds(evec_grid, cube_grid):
    evec_loggrid = np.log(evec_grid)
    cube_loggrid = np.log(cube_grid)

    # find bounds on redshift subject to constraint that
    # min(lcube) < zify(min(lrect), z) < max(lcube)
    # AND
    # min(lcube) < zify(max(lrect), z) < max(lcube)
    # since max > min, this boils down to
    # (1) shift(min(lrect)) > min(lcube) & (2) shift(max(lrect)) < max(lcube)
    zmin = cube_loggrid.min() - evec_loggrid.min()
    zmax = cube_loggrid.max() - evec_loggrid.max()

    return zmin, zmax

def gen_flux(l):
    f0 = 3500. / l
    tau = gen_tau(l)
    return f0 * np.exp(-tau)

def gen_ivar(l):
    return 10. * (l / 5000.)

def gen_tau(l, loc=5000., w=2., tau0=1.):
    tau = tau0 * np.exp(-((l - loc)**2. / (2. * w**2.)))
    return tau

def load_real_restcubes(plate, ifu, MPL_v):
    dered = MaNGA_deredshift.from_plateifu(plate, ifu, MPL_v)
    l_rest, f_rest, ivar_rest = dered.transform_to_restframe(
            l=dered.drp_l * u.AA, f=dered.flux, ivar=dered.ivar)
    return l_rest, f_rest, ivar_rest

def redshift_fullcube(l, f, ivar, z_in, z_out):
    data = np.empty_like(
        f, dtype=[('l', float), ('f', float), ('ivar', float)])
    data['l'] = l
    data['f'] = f
    data['ivar'] = ivar
    rules = [dict(name='l', exponent=+1),
             dict(name='f', exponent=-1),
             dict(name='ivar', exponent=+2)]

    res = slrs(data_in=data, rules=rules, z_in=z_in, z_out=z_out)
    return res['l'], res['f'], res['ivar']

def drizzle_spec(lrest, lgrid):
    logelrest = np.log(lrest)
    loglrest = np.log10(lrest)
    logelgrid = np.log(lgrid)
    loglgrid = np.log10(lgrid)

    dlogel = ut.determine_dlogl(loglgrid)
    dlogl = ut.determine_dlogl(loglgrid)

def gen_logl(logl0, dlogl, nl):
    logl = logl0 + dlogl * np.array(range(nl))
    return logl

def restwithinrect(rect_lhs, rect_rhs, rest_lhs, rest_rhs):
    '''
    verify that each rectified-grid pixel is contained entirely within
        the rest-frame grid
    '''
    within_lhs = rect_lhs >= rest_lhs.min()
    within_rhs = rect_rhs <= rest_rhs.max()
    return np.logical_and(within_lhs, within_rhs)

def get_logl_bin_edges(logl, dlogl):
    ledges, redges = logl - 0.5 * dlogl, logl + 0.5 * dlogl
    return ledges, redges

def find_rest_i0_lhs(grid0_ctr, rest_ctr, dlogl):
    '''
    find the left-most contributor to a rectified bin: this element's
        lhs-edge will be the largest that also is smaller than the corresponding
        rect-grid element's lhs-edge

    args:
        - grid0_ctr: center of first bin in rectified grid
        - rest_ctr: center of rest-frame wavelength cube
        - dlogl
    '''

    grid0_l = grid0_ctr - 0.5 * dlogl
    rest_l = rest_ctr - 0.5 * dlogl

    lt_grid_l = rest_l < grid0_l
    rest_ctr_m = np.ma.masked_where(a=rest_ctr, condition=~lt_grid_l)
    m, am = rest_ctr_m.max(axis=0), rest_ctr_m.argmax(axis=0)

    return m, am

def find_left_fraction(grid0_ctr, rest_ctr, dlogl):
    '''
    find and return the fraction of each rectified grid bin that is covered by
        the nearest rest-frame bin to the left
    '''
    rest_v0, rest_i0 = find_rest_i0_lhs(grid0_ctr, rest_ctr, dlogl)
    grid0_l = grid0_ctr - 0.5 * dlogl
    rest_v0_l = rest_v0 - 0.5 * dlogl
    f = 1. - (grid0_l - rest_v0_l.data) / dlogl

    return f

def covar_offdiag(f_r_or_l):
    return f_r_or_l * (1. - f_r_or_l)

def drizzle_flux(grid_ctr, rest_ctr, wave_lin, flux_cube, ivar_cube):
    '''
    drizzle flux-densities and errors from one large cube into a smaller one.
        This is based on Carnall (2017), plus additional assumption
        of identical bin-width

    args:
        - grid_ctr: 1d, monotonic-increasing array of wavelength centers
                    for rectified grid
        - rest_ctr: 3d cube of wavelength centers for perfectly-deredshifted
                    flux cube
        - wave_lin: is wavelength in linear units?
        - f_cube: 3d cube of flux-density
        - ivar_cube: 3d cube of flux-density inverse-variance
    '''
    ivar0 = 1.0e-8

    # transform
    if wave_lin:
        grid_ctr = np.log10(grid_ctr)
        rest_ctr = np.log10(rest_ctr)

    mapshape = rest_ctr.shape[1:]
    nl_grid = len(grid_ctr)

    dlogl = ut.determine_dlogl(grid_ctr)

    # retrieve wavelength and index of left contributor to 0th rectified bin
    # both are maps
    lval_left, li_left = find_rest_i0_lhs(
        grid0_ctr=grid_ctr[0], rest_ctr=rest_ctr, dlogl=dlogl)

    # fraction of dl contributed by left
    frac_left = find_left_fraction(grid_ctr[0], rest_ctr, dlogl)[None, :, :]
    frac_right = 1. - frac_left

    # construct map indexing arrays
    II, JJ = np.meshgrid(*map(range, mapshape), indexing='ij')
    II, JJ = II[None, ...], JJ[None, ...]
    # all axis-0 indices for left-contributors
    LL_all = np.arange(nl_grid + 1, dtype=int)[:, None, None] + \
             li_left[None, :, :]

    LL_all_split = np.array_split(LL_all, 100, axis=0)

    # contributor arrays: we extract all at once because advanced indexing
    # copies data, and this means we only have to do it once per flux or ivar
    flux_all = np.concatenate([flux_cube[LL_sect, II, JJ]
                               for LL_sect in LL_all_split], axis=0) + ivar0
    ivar_all = np.concatenate([ivar_cube[LL_sect, II, JJ]
                               for LL_sect in LL_all_split], axis=0) + ivar0

    # contributor arrays, with left and right side changing along axis 0
    flux_c = rolling_window(flux_all, nl_grid, axis=0)
    ivar_c = rolling_window(ivar_all, nl_grid, axis=0)
    fracs = np.stack([frac_left, frac_right])
    w = dlogl

    # array-wise ops
    # NOTE: In Carnall's equations, w_i is the width of the origin wavelength grid
    #     which can be dropped, since dlogl for origin and destination are both
    #     uniform and identical. For the sake of being explicit, w is kept.
    flux_wtd = (fracs * w * flux_c).sum(axis=0) / (fracs * w).sum(axis=0)
    var_wtd = (fracs**2. * w**2. / ivar_c).sum(axis=0) / ((fracs * w).sum(axis=0))**2.

    # covariance between adjacent spectral locations in final grid
    # is not accounted for: so increment var by a factor of ~2
    var_wtd *= 2.

    ivar_wtd = 1. / var_wtd

    return flux_wtd, ivar_wtd
