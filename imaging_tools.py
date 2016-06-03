import numpy as np

from astroquery.skyview import SkyView

class MaNGA_SkyView(SkyView):
    '''
    subclass of SkyView, which allows access to FITS thumbnails of
        a MaNGA object
    '''
    @classmethod
    def from_drpall_row(cls, row):
        raise NotImplementedError

    @classmethod
    def from_wcs_header(cls, header):
        raise NotImplementedError
