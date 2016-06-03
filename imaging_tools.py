import numpy as np

from astropy import table as t, constants as c, units as u, \
    coordinates as coords
from astroquery.sdss import SDSSClass

class MaNGA_SkyView(SDSSClass):
    '''
    subclass of SkyView, which allows access to FITS thumbnails of
        a MaNGA object
    '''

    def images_from_drpall_row(self, row):
        p = '{}d {}d'.format(row['ifura'], row['ifudec'])
        p = coords.SkyCoord(p, frame='icrs')
        bands = ['u', 'g', 'r', 'i', 'z']
        return self.get_images(
            coordinates=p, band=bands, radius=20.*u.arcsec)

    def images_from_wcs_header(self, header):
        p = '{}d {}d'.format(header['CRVAL1'], header['CRVAL2'])
        p = coords.SkyCoord(p, frame='icrs')
        bands = ['u', 'g', 'r', 'i', 'z']
        return self.get_images(
            coordinates=p, band=bands, radius=20.*u.arcsec)
