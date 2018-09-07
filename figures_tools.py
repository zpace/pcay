import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from copy import copy

from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel, proj_plane_pixel_scales
import astropy.coordinates as coords

import warnings

try:
    from astropy.wcs.utils import linear_offset_coordinates
except ImportError:
    pass
else:
    warnings.warn('linear_offset_coordinates now available! Use it instead!')

cm = copy(plt.cm.viridis)
cm.set_under(color='gray', alpha=0.5)
cm.set_bad(alpha=1.)

textboxprops = dict(facecolor='wheat', alpha=0.5)

def decide_lims_pctls(a, pctls=[.5, 99.5], bds=[None, None]):
    pctls_vals = np.percentile(a, pctls)
    bds = [b if b is not None else v for b, v in zip(bds, pctls_vals)]

    return bds

def linear_offset_coordinates(wcs, center):
    '''
    return a locally linear offset coordinate system

    does the simplest thing possible and assumes no projection distortions
    '''
    assert isinstance(center, coords.SkyCoord), \
        '`center` must by of type `SkyCoord`'
    assert center.isscalar, '`center` must have length 1'
    # Convert center to pixel coordinates
    xp, yp = skycoord_to_pixel(center, wcs)

    # Set up new WCS
    new_wcs = WCS(naxis=2)
    new_wcs.wcs.crpix = xp + 1, yp + 1
    new_wcs.wcs.crval = 0., 0.
    new_wcs.wcs.cdelt = proj_plane_pixel_scales(wcs)
    new_wcs.wcs.ctype = 'XOFFSET', 'YOFFSET'
    new_wcs.wcs.cunit = 'deg', 'deg'

    return new_wcs

def savefig(fig, fname, fdir, close=True, **kwargs):
    fpath = os.path.join(fdir, fname)
    fig.savefig(fpath, **kwargs)
    if close:
        plt.close(fig)

def annotate_badPDF(ax, mask):
    # place little, red 'x' markers where mask is true

    x = np.array(range(mask.shape[0]))
    y = np.array(range(mask.shape[1]))
    XX, YY = np.meshgrid(x, y)
    XX, YY = XX[mask], YY[mask]

    ax.scatter(XX, YY, facecolor='r', edgecolor='None', s=5, marker='.',
               zorder=10)

def gen_gridspec_fig(N, add_row=False, spsize=(1.75, 1.25),
                     border=(0.3, 0.25, 0.3, 0.25), space=(0.5, 0.5),
                     dpi=100, **kwargs):
    '''
    generate a subplot grid with gridspec

    params:
        - N: # of subplots in the basic grid
        - add_row: add an extra row
        - spsize: dimensions (width, height) of each
                  subplot element in grid (in)
        - border: (l, r, u, d) border widths (in)
        - space: (w, h) spacing between subplots (in)
    '''

    ncols = int(np.sqrt(N))

    if ncols < 2:
        ncols = 2

    n_in_last_row = N % ncols
    nrows = (N // ncols) + ((n_in_last_row > 0) * 1) + (add_row * 1)

    # subplot size (w, h)
    spw, sph = spsize

    # borders
    bl, br, bu, bd = border

    # total figure dimensions
    # includes 2 borders, spacing, subplots
    wsp, hsp = space
    figw = bl + br + ((ncols - 1) * wsp) + (ncols * spw)
    figh = bu + bd + ((nrows - 1) * hsp) + (nrows * sph)

    # compute spacing (gridspec takes in fraction of subplot size)
    wspace = wsp / spw
    hspace = hsp / sph

    # compute borders (gridspec takes in fraction of fig size)
    right = 1. - (br / figw)
    left = bl / figw
    bottom = bd / figh
    top = 1. - (bu / figh)

    # instantiate figure
    fig = plt.figure(figsize=(figw, figh), dpi=dpi)

    gs = gridspec.GridSpec(ncols, nrows, left=left, right=right,
                           bottom=bottom, top=top,
                           wspace=wspace, hspace=hspace, **kwargs)

    return gs, fig
