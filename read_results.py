import numpy as np

from astropy.io import fits

class PCAOutput(fits.HDUList):
    '''
    stores output data from PCA that as been written to FITS.
    '''
    @classmethod
    def fromfile(cls, fname, *args, **kwargs):
        ret = super().fromfile(fname, *args, **kwargs)
        return ret

    def getdata(self, extname):
        '''
        get full array in one extension
        '''
        return self[extname].data

    def flattenedmap(self, extname):
        return self.getdata(extname).flatten()

    def cubechannel(self, extname, ch):
        '''
        get one channel (indexed along axis 0) of one extension
        '''
        return self.getdata(extname)[ch]

    def flattenedcubechannel(extname, ch):
        return self.cubechannel(extname, ch)

    def flattenedcubechannels(extname, chs):
        return np.stack([self.flattenedcubechannel(extname, ch)
                         for ch in chs])
