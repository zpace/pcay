import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import trapz

from astropy import units as u, constants as c, table as t


l_eff_d = {'r': 6166. * u.AA, 'i': 7480. * u.AA, 'z': 8932. * u.AA}
l_wid_d = {'r': 550. * u.AA, 'i': 1300. * u.AA, 'z': 1000. * u.AA}


def lumdens2bbdlum(lam, Llam, band):
    '''
    Convert a spectral luminosity density to a broadband luminosity

    Convolve source spectrum/a with a filter; specific routine derived '
        from MaNGA's ml_mangatosdssimage.pro routine from DRP

    Parameters
    ----------
    lam :
        wavelength array of source spectrum

    Llam :
        Luminosity density, units sim to Lsun/AA

    band :
        chosen band
    '''

    # make sure everything's in the right units
    Llam = Llam.to('erg s-1 AA-1')
    lam = lam.to('AA')
    nu = lam.to('Hz', equivalencies=u.spectral())

    # read in filter table
    band_tab = t.Table.read('filters/{}_SDSS.res'.format(band),
                            names=['lam', 'f'], format='ascii')

    # set up interpolator
    band_interp = interp1d(x=band_tab['lam'].quantity.value,
                           y=band_tab['f'], fill_value=0.,
                           bounds_error=False)

    f = band_interp(lam)

    # convert to Lnu, by multiplying by lam^2/c
    Lnu = (Llam * lam**2. / c.c).to('Lsun Hz-1')

    L = trapz(x=nu.value[::-1], y=f * Lnu.value, axis=-1) * u.Lsun

    return L

def color(hdulist, band1='g', band2='r'):
    '''
    Calculate the color of a MaNGA galaxy, based on two bandpasses

    By convention, C_br = b - r
    '''
    img1 = hdulist['{}IMG'.format(band1)].data
    img2 = hdulist['{}IMG'.format(band2)].data

    color = -2.5 * np.log10(img1 / img2)

    return color
