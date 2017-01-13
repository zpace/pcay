import os

import numpy as np
import matplotlib.pyplot as plt
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
