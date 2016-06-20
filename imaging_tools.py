import numpy as np

from astropy import table as t, constants as c, units as u, \
    coordinates as coords
from astroquery.sdss import SDSSClass

class MaNGA_ImageDownloader(SDSSClass):
    '''
    subclass of SkyView, which allows access to FITS thumbnails of
        a MaNGA object
    '''
