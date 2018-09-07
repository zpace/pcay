import numpy as np
import astropy.table as t

import utils as ut

from warnings import simplefilter, catch_warnings

data = t.Table.read('data/indices.dat', format='ascii')
for k in ['band', 'blue', 'red']:
    data[k] = [np.array(list(map(float, r.split()))) for r in data[k]]

data.add_index(['ixname'])

eps = np.finfo(float).eps

class StellarIndex(object):
    def __init__(self, name):
        '''
        some generic Lick/IDS stellar index
        '''

        row = data.loc[name]

        for k in data.colnames:
            setattr(self, k, row[k])

        self._func = ''.join(('_', row['ixtype']))

    def mask(self, rbb, l):
        rbb = getattr(self, rbb)
        return (l >= rbb.min()) * (l <= rbb.max())

    def _EW(self, l, flam, ivar, axis, *args, **kwargs):
        # *Eq 2*, from Trager, Worthey, Faber, Burstein, and Gonzalez
        # (APJs116:1-128, 1998)
        mask_red = self.mask('red', l)
        mask_blue = self.mask('blue', l)
        mask_band = self.mask('band', l)

        l_red = l[..., mask_red]
        l_blue = l[..., mask_blue]
        l_band = l[..., mask_band]
        flam_red = flam[..., mask_red]
        flam_blue = flam[..., mask_blue]
        flam_band = flam[..., mask_band]
        ivar_red = ivar[..., mask_red]
        ivar_blue = ivar[..., mask_blue]

        # mean of red & blue continuum bands
        F0_red = np.average(flam_red, weights=ivar_red, axis=-1)
        F0_blue = np.average(flam_blue, weights=ivar_blue, axis=-1)

        # difference in flam and lam from blue continuum to red
        dflam = F0_red - F0_blue
        dl = l_red.mean() - l_blue.mean()
        m = dflam / dl

        F0 = self._pointslope(x0=l_blue.mean(), y0=F0_blue,
                              slope=m, x=l_band)

        return np.trapz(x=l_band, y=(1. - flam_band / F0), axis=-1)

    def _ratio(self, l, flam, ivar, *args, **kwargs):
        # SS2.2 from Balogh, Morris, Yee, Carlberg, and Ellingson
        # (ApJ 527:54, 1999); and Hamilton (ApJ 297:371, 1985)
        mask_red = self.mask('red', l)
        mask_blue = self.mask('blue', l)

        l_red = l[..., mask_red]
        l_blue = l[..., mask_blue]
        flam_red = flam[..., mask_red]
        flam_blue = flam[..., mask_blue]
        dl = l * ut.determine_dlogl(np.log(l))
        dl_red = dl[..., mask_red]
        dl_blue = dl[..., mask_blue]

        F0_red = np.sum(flam_red * l_red**2. * dl_red, axis=-1)
        F0_blue = np.sum(flam_blue * l_blue**2. * dl_blue, axis=-1)

        return F0_red / F0_blue

    def _mag(self, l, flam, ivar, *args, **kwargs):
        # *Eq 3*, from Trager, Worthey, Faber, Burstein, and Gonzalez
        # (ApJS 116:1-128, 1998)
        mask_red = self.mask('red', l)
        mask_blue = self.mask('blue', l)
        mask_band = self.mask('band', l)

        l_red = l[..., mask_red]
        l_blue = l[..., mask_blue]
        l_band = l[..., mask_band]
        flam_red = flam[..., mask_red]
        flam_blue = flam[..., mask_blue]
        flam_band = flam[..., mask_band]
        ivar_red = ivar[..., mask_red]
        ivar_blue = ivar[..., mask_blue]

        # mean of red & blue continuum bands
        F0_red = np.average(flam_red, weights=ivar_red, axis=-1)
        F0_blue = np.average(flam_blue, weights=ivar_blue, axis=-1)

        # difference in flam and lam from blue continuum to red
        dflam = F0_red - F0_blue
        dl = l_red.mean() - l_blue.mean()
        m = dflam / dl

        F0 = self._pointslope(x0=l_blue.mean(), y0=F0_blue,
                              slope=m, x=l_band)

        r = np.trapz(x=l_band, y=(flam_band / F0), axis=-1)
        return -2.5 * np.log10(r / (l_band.max() - l_band.min()))

    def _pointslope(self, x0, y0, slope, x):
        dx = x - x0
        # ordinal / spatial dimensions
        nspectra = slope.shape
        for _ in nspectra:
            dx = np.expand_dims(dx, -1)
            slope = np.expand_dims(slope, 0)

        newaxes = np.arange(len(nspectra))
        dy = dx * slope
        res = y0 + dy

        # catch warning from axis rearrangement
        with catch_warnings():
            simplefilter('ignore')
            res = np.moveaxis(res, newaxes, -(newaxes + 1))
        return res.squeeze()

    def __call__(self, l, flam, ivar=None, axis=0, *args, **kwargs):
        if len(l.shape) > 1:
            raise ValueError('wavelength array (l) must be 1D')

        flam[flam == 0.] = eps
        if ivar is None:
            ivar = np.ones_like(flam)
        ivar[ivar == 0.] = eps

        # rearrange axes to make broadcasting work
        ivar = np.moveaxis(ivar, axis, -1)
        flam = np.moveaxis(flam, axis, -1)

        return getattr(self, self._func)(l, flam, ivar, axis, *args, **kwargs)


class StellarIndices(object):
    def __init__(self):
        self.indices = {k: StellarIndex(k) for k in data['ixname']}

    def __call__(self, *args, **kwargs):
        return {k: self.indices[k](*args, **kwargs) for k in data['ixname']}
