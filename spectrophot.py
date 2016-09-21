import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import quad

from astropy import units as u, table as t


def spec2phot_source(lam, Llam, band):
    '''
    convolve source spectrum with a filter

    Parameters
    ----------
    lam :
        wavelength array of source spectrum

    Llam :
        Luminosity density, units sim to Lsun/AA

    band :
        chosen band
    '''

    # convert to per unit frequency, with a hack to get around Ldens
    Lnu = (Llam / u.cm**2.).to('Lsun Hz-1 cm-2',
                               equivalencies=u.spectral_density(
                                   wav=lam)) * u.cm**2.
    Lnu = Lnu.to('Lsun Hz-1')
    nu = lam.to('Hz', equivalencies=u.spectral())

    # set up interpolator
    Lnu_interp = interp1d(x=nu.value, y=Lnu.value, fill_value=0.,
                          bounds_error=False)

    # read in filter table
    band_tab = t.Table.read('filters/{}_SDSS.res'.format(band),
                            names=['lam', 'f'], format='ascii')
    # convert to per unit frequency
    band_tab['lam'].unit = u.AA
    band_tab['nu'] = band_tab['lam'].to('Hz', equivalencies=u.spectral())

    # set up interpolator
    band_interp = interp1d(x=band_tab['nu'].quantity.value,
                           y=band_tab['f'], fill_value=0.,
                           bounds_error=False)

    # limits of integration
    nu_ulim = (1000. * u.AA).to('Hz', equivalencies=u.spectral()).value
    nu_llim = (15000. * u.AA).to('Hz', equivalencies=u.spectral()).value

    def bandpassify(freq):
        Lnu_ = Lnu_interp(freq)
        f_ = band_interp(freq)
        return (Lnu_ * f_)

    L, dL = quad(bandpassify, nu_llim, nu_ulim)

    return L
