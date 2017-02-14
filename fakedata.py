import numpy as np
from astropy.io import fits
from astropy import units as u, constants as c

from scipy.interpolate import interp1d

import os
from copy import copy

# local
import utils as ut
from spectrophot import Spec2Phot

# add manga RC location to path, and import config
if os.environ['MANGA_CONFIG_LOC'] not in sys.path:
    sys.path.append(os.environ['MANGA_CONFIG_LOC'])

import mangarc

if mangarc.tools_loc not in sys.path:
    sys.path.append(mangarc.tools_loc)

# personal
import manga_tools as m


class FakeData(object):
    '''
    Make a fake IFU and fake DAP stuff
    '''
    def __init__(self, l_model, s_model, row, drp_base, dap_base):

        # redshift the model spectrum
        z_cosm = row['nsa_zdist']
        z_pec = (dap_base['STELLAR_SIGMA'].data * u.Unit('km/s') / c.c).decompose().value
        z_out = (z_cosm + z_pec)

        # make cube
        cubeshape = drp_bas['FLUX'].data.shape
        nl, *mapshape = cubeshape

        s_norm = s_model.max()
        f = np.tile(s_model[..., None, None] / s_norm, (1, ) + mapshape)
        ivar_dummy = np.ones_like(f)

        # redshift model cube
        l_m_z, s_m_z, _ = ut.redshift(l, f, ivar_dummy, z_in=0., z_out=z_out)
        logl_m_z = np.log10(l_m_z)

        # interpolate redshifted model cube
        logl_f = np.log10(drp_base['WAVE'].data)
        s_m_z_c = np.zeros(cubeshape)

        for ind in np.ndindex(mapshape):
            s_m_z_c[:, ind[0], ind[1]] = self.resample_spaxel(
                logl_m_z[:, ind[0], ind[1]], s_m_z[:, ind[0], ind[1]], logl_f)

        # normalize everything to have the same observed-frame r-band flux
        m_r_drp = Spec2Phot(lam=drp_base['WAVE'].data,
                            flam=drp_base['FLUX'].data).ABmags['sdss2010-r']
        m_r_mzc = Spec2Phot(lam=drp_base['WAVE'].data,
                            flam=s_m_z_c).ABmags['sdss2010-r']
        # flux ratio map
        r = 10.**(0.4 * (m_r_drp - m_r_mcz))
        m_r_mcz *= r[None, ...]

        # add error vector based on ivar
        err = np.random.randn(*cubeshape)
        fakecube = m_r_mcz + (err / np.sqrt(drp_base['IVAR'].data))
        fakecube[~np.isfinite(fakecube)] = 0.

        self.dap_base = dap_base
        self.drp_base = drp_base
        self.fluxcube = fakecube
        self.row = row

    def resample_spaxel(self, logl_in, flam_in, logl_out):
        '''
        resample the given spectrum to the specified logl grid
        '''

        interp = interp1d(x=logl_in, y=flam_in, kind='linear')
        return interp(logl_out)

    @classmethod
    def from_FSPS(cls, fname, i, plateifu_base,
                  mpl_v='MPL-5', kind='SPX-GAU-MILESHC'):

        # load models
        models_hdulist = fits.open(fname)
        models_specs = models_hdulist['flam'].data
        models_lam = models_hdulist['lam'].data
        models_logl = np.log10(models_lam)

        # choose specific model
        model_spec = models_specs[i, :]

        # load data
        plate, ifu = map(str, plateifu_base.split('-'))

        drp_base = m.load_drp_logcube(plate, ifu, mpl_v)
        dap_base = m.load_dap_maps(plate, ifu, mpl_v, kind)
        row = m.load_drpall(mpl_v, index='plateifu').loc[plateifu_base]

        return cls(l_model=models_logl, s_model=model_spec, row=row,
                   drp_base=drp_base, dap_base=dap_base)

    def write(self):
        '''
        write out fake LOGCUBE and DAP
        '''

        fname_base = self.row['plateifu']
        basedir = 'fakedata'
        drp_fname = os.path.join(basedir, '{}_drp.fits'.format(fname_base))
        dap_fname = os.path.join(basedir, '{}_dap.fits'.format(fname_base))

        new_drp_cube = copy(self.drp_base)
        drp_base['FLUX'].data = self.fluxcube

        new_dap_cube = copy(self.dap_base)

        new_drp_cube.writeto(drp_fname)
        new_dap_cube.writeto(dap_fname)
