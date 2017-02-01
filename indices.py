import numpy as np
import astropy.table as t

data = t.Table.read('data/indices.dat', format='ascii')
for k in ['band', 'blue', 'red']:
    data[k] = [np.array(list(map(float, r.split()))) for r in data[k]]

data.add_index(['name'])

eps = np.finfo(float).eps

class StellarIndex(object):
    def __init__(self, name):
        '''
        some generic Lick/IDS stellar index
        '''

        row = data.loc[name]

        for k in data.colnames[:-1]:
            setattr(self, k, row[k])

        self._func = ''.join(('_', row['ixtype']))

    def mask(self, rbb, l):
        rbb = getattr(self, rbb)
        return (l >= rbb.min()) * (l <= rbb.max())

    def _EW(self, l, flam, ivar, axis, *args, **kwargs):
        mask_red = self.mask('red', l)
        mask_blue = self.mask('blue', l)
        mask_band = self.mask('band', l)

        l_red = l[mask_red, ...]
        l_blue = l[mask_blue, ...]
        l_band = l[mask_band, ...]
        flam_red = flam[mask_red, ...]
        flam_blue = flam[mask_blue, ...]
        flam_band = flam[mask_band, ...]
        ivar_red = ivar[mask_red, ...]
        ivar_blue = ivar[mask_blue, ...]

        F0_red = np.average(flam_red, weights=ivar_red, axis=0)
        F0_blue = np.average(flam_blue, weights=ivar_blue, axis=0)

        dflam, dl = F0_red - F0_blue, l_red.mean() - l_blue.mean()
        m = dflam / dl
        F0 = lambda l: F0_blue.mean() + m * (l - l_blue.mean())

        return np.trapz(x=l_band, y=(1. - flam_band / F0(l_band)), axis=0)

    def _ratio(self, l, flam, ivar, *args, **kwargs):
        mask_red = self.mask('red', l)
        mask_blue = self.mask('blue', l)
        mask_band = self.mask('band', l)

        l_red = l[mask_red, ...]
        l_blue = l[mask_blue, ...]
        l_band = l[mask_band, ...]
        flam_red = flam[mask_red, ...]
        flam_blue = flam[mask_blue, ...]
        flam_band = flam[mask_band, ...]
        ivar_red = ivar[mask_red, ...]
        ivar_blue = ivar[mask_blue, ...]

        F0_red = np.average(flam_red, weights=ivar_red, axis=0)
        F0_blue = np.average(flam_blue, weights=ivar_blue, axis=0)

        return F0_red / F0_blue

    def _mag(self, l, flam, *args, **kwargs):
        mask_red = self.mask('red', l)
        mask_blue = self.mask('blue', l)
        mask_band = self.mask('band', l)

        l_red = l[mask_red, ...]
        l_blue = l[mask_blue, ...]
        l_band = l[mask_band, ...]
        flam_red = flam[mask_red, ...]
        flam_blue = flam[mask_blue, ...]
        flam_band = flam[mask_band, ...]
        ivar_red = ivar[mask_red, ...]
        ivar_blue = ivar[mask_blue, ...]

        F0_red = np.average(flam_red, weights=ivar_red, axis=0)
        F0_blue = np.average(flam_blue, weights=ivar_blue, axis=0)

        dflam, dl = F0_red - F0_blue, l_red.mean() - l_blue.mean()
        m = dflam / dl
        F0 = lambda l: F0_blue.mean() + m * (l - l_blue.mean())

        r = np.trapz(x=l_band, y=(flam_band / F0(l_band)), axis=0)
        return -2.5 * np.log10(r / (l_band.max() - l_band.min()))

    def __call__(self, l, flam, ivar, *args, **kwargs):
        flam[flam == 0.] = eps
        ivar[ivar == 0.] = eps
        if len(l.shape) > 1:
            raise StandardError('wavelength array (l) must be 1D')
        return getattr(self, self._func)(l, flam, ivar, *args, **kwargs)


class StellarIndices(object):
    def __init__(self):
        self.indices = {k: StellarIndex(k) for k in data['name']}

    def __call__(self, *args, **kwargs):
        return {k: self.indices[k](*args, **kwargs) for k in data['name']}
