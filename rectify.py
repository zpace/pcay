import numpy as np

from astropy import units as u, constants as c
from astropy.io import fits

from importer import *
import utils as ut
from spectrophot import Spec2Phot
import regrid

import manga_tools as m
from manga_elines import get_emline_qty
import spec_tools

class MaNGA_deredshift(object):
    '''
    class to deredshift reduced MaNGA data based on velocity info from DAP

    preserves cube information, in general

    also builds in a check on velocity coverage, and computes a mask
    '''

    spaxel_side = 0.5 * u.arcsec

    def __init__(self, drp_hdulist, dap_hdulist, drpall_row,
                 max_vel_unc=500. * u.Unit('km/s'), drp_dlogl=None):
        self.drp_hdulist = drp_hdulist
        self.dap_hdulist = dap_hdulist
        self.drpall_row = drpall_row
        self.plateifu = self.drp_hdulist[0].header['PLATEIFU']

        self.vel = dap_hdulist['STELLAR_VEL'].data * u.Unit('km/s')
        self.vel_ivar = dap_hdulist['STELLAR_VEL_IVAR'].data * u.Unit(
            'km-2s2')

        self.z = drpall_row['nsa_z']

        # mask all the spaxels that have high stellar velocity uncertainty
        self.vel_ivar_mask = (1. / np.sqrt(self.vel_ivar)) > max_vel_unc
        self.vel_mask = m.mask_from_maskbits(
            self.dap_hdulist['STELLAR_VEL_MASK'].data, [30])

        self.drp_l = drp_hdulist['WAVE'].data
        self.drp_logl = np.log10(self.drp_l)
        if drp_dlogl is None:
            drp_dlogl = ut.determine_dlogl(self.drp_logl)
        self.drp_dlogl = drp_dlogl

        flux_hdu, ivar_hdu, l_hdu = (drp_hdulist['FLUX'], drp_hdulist['IVAR'],
                                     drp_hdulist['WAVE'])

        self.flux = flux_hdu.data
        self.ivar = ivar_hdu.data

        self.units = {'l': u.AA, 'flux': u.Unit('1e-17 erg s-1 cm-2 AA-1')}

        self.S2P = Spec2Phot(lam=(self.drp_l * self.units['l']),
                             flam=(self.flux * self.units['flux']))

        self.DONOTUSE = m.mask_from_maskbits(drp_hdulist['MASK'].data, [10])

        self.ivar *= ~self.DONOTUSE

    @classmethod
    def from_plateifu(cls, plate, ifu, MPL_v, kind, row=None,
                      **kwargs):
        '''
        load a MaNGA galaxy from a plateifu specification
        '''

        plate, ifu = str(plate), str(ifu)

        if row is None:
            drpall = m.load_drpall(MPL_v, index='plateifu')
            row = drpall.loc['{}-{}'.format(plate, ifu)]

        drp_hdulist = m.load_drp_logcube(plate, ifu, MPL_v)
        dap_hdulist = m.load_dap_maps(plate, ifu, MPL_v, kind)
        return cls(drp_hdulist, dap_hdulist, row, **kwargs)

    @classmethod
    def from_fakedata(cls, plate, ifu, MPL_v, basedir='fakedata', row=None,
                      kind='SPX-MILESHC-MILESHC', **kwargs):
        '''
        load fake data based on a particular already-observed galaxy
        '''

        plate, ifu = str(plate), str(ifu)

        if row is None:
            drpall = m.load_drpall(MPL_v, index='plateifu')
            row = drpall.loc['{}-{}'.format(plate, ifu)]

        drp_hdulist = fits.open(
            os.path.join(basedir, '{}-{}_drp.fits'.format(plate, ifu)))
        dap_hdulist = fits.open(
            os.path.join(basedir, '{}-{}_dap.fits'.format(plate, ifu)))

        return cls(drp_hdulist, dap_hdulist, row, **kwargs)

    def transform_to_restframe(self, l, f, ivar):
        '''
        bring cube into rest frame
        '''

        # shift into restframe
        l_rest, f_rest, ivar_rest = ut.redshift(
            l=l, f=f, ivar=ivar, z_in=self.z_map, z_out=0.)

        return l_rest, f_rest, ivar_rest

    def correct_and_match(self, template_logl, template_dlogl=None,
                          method='drizzle', dered_kwargs={}):
        '''
        gets datacube ready for PCA analysis:
            - take out galactic extinction
            - compute per-spaxel redshifts
            - deredshift observed spectra
            - raise alarms where templates don't cover enough l range
            - return subarrays of flam, ivar

        (this does not perform any fancy interpolation, just "shifting")
        (nor are emission line features masked--that must be done in post-)
        '''
        if template_dlogl is None:
            template_dlogl = spec_tools.determine_dlogl(template_logl)

        if template_dlogl != self.drp_dlogl:
            raise csp.TemplateCoverageError(
                'template and input spectra must have same dlogl: ' +
                'template\'s is {}; input spectra\'s is {}'.format(
                    template_dlogl, self.drp_dlogl))

        # correct for MW extinction
        r_v = 3.1
        EBV = self.drp_hdulist[0].header['EBVGAL']
        f_mwcorr, ivar_mwcorr = ut.extinction_correct(
            l=self.drp_l * u.AA, f=self.flux,
            ivar=self.ivar, r_v=r_v, EBV=EBV)

        l_rest, f_rest, ivar_rest = self.transform_to_restframe(
            self.drp_l, f_mwcorr, ivar_mwcorr)

        # and make photometric object to reflect rest-frame spectroscopy
        ctr = [i // 2 for i in self.z_map.shape]
        # approximate rest wavelength of whole cube as rest wavelength
        # of central spaxel
        l_rest_ctr = l_rest[:, ctr[0], ctr[1]]
        self.S2P_rest = Spec2Phot(lam=(l_rest_ctr * self.units['l']),
                                  flam=(f_rest * self.units['flux']))

        self.regrid = regrid.Regridder(
            loglgrid=template_logl, loglrest=np.log10(l_rest),
            frest=f_rest, ivarfrest=ivar_rest, dlogl=template_dlogl)

        # call appropriate regridder method
        flux_regr, ivar_regr = getattr(
            self.regrid, method)(**dered_kwargs)

        spax_mask = np.logical_or.reduce((
            self.vel_mask, self.vel_ivar_mask))

        self.flux_regr, self.ivar_regr, self.spax_mask = flux_regr, ivar_regr, spax_mask

        return flux_regr, ivar_regr, spax_mask

    def compute_eline_mask(self, template_logl, template_dlogl=None, ix_eline=7,
                           half_dv=300. * u.Unit('km/s')):

        from elines import (balmer_low, balmer_high, helium,
                            bright_metal, faint_metal)
        from itertools import chain
        el_l_air = [balmer_low, balmer_high, helium, bright_metal, faint_metal]

        # find mask width for all spaxels
        mask_velwidth = determine_eline_mask_dv(
            self.dap_hdulist, minimum_value=half_dv.value, n_times_sigma=1.5)

        if template_dlogl is None:
            template_dlogl = spec_tools.determine_dlogl(template_logl)

        EW = self.eline_EW(ix=ix_eline)

        add_balmer_low = (EW >= 0. * u.AA)
        add_balmer_high = (EW >= 2. * u.AA)
        add_helium = (EW >= 10. * u.AA)
        add_brightmetal = (EW >= 0. * u.AA)
        add_faintmetal = (EW >= 10. * u.AA)
        linelistflags = [add_balmer_low, add_balmer_high, add_helium,
                         add_brightmetal, add_faintmetal]
        # full list of mask flags: one corresponds to each line in each eline dict
        useflags = list(chain(*map(lambda a: [a[0]] * len(a[1]),
                                   zip(linelistflags, el_l_air))))

        temlogl = template_logl
        teml = 10.**temlogl
        temlogel = np.log(teml)

        #full_mask = np.zeros((len(temlogl),) + EW.shape, dtype=bool)

        el_lel_vac = np.concatenate(list(map(
            lambda d: np.log(spec_tools.air2vac(np.array(list(d.values())),
                                                u.AA).value), el_l_air)))

        # iterate through eline types
        full_mask = np.logical_or.reduce(
            [masked_around_line(
                 line_logel=lel, dv_map=mask_velwidth, obs_logel=temlogel) * \
                     flag[None, :, :]
            for flag, lel in zip(useflags, el_lel_vac)])

        return full_mask

    def eline_EW(self, ix):
        return self.dap_hdulist['EMLINE_SEW'].data[ix] * u.Unit('AA')

    def coadd(self, tem_l, good=None):
        '''
        return coadded spectrum and ivar

        params:
         - good: map of good spaxels
        '''

        if good is None:
            good = np.ones_like(self.flux_regr[0, ...])

        ivar = self.ivar_regr * good[None, ...]

        flux, ivar = ut.coadd(f=self.flux_regr, ivar=ivar)
        lam, flux, ivar = (tem_l[:, None, None], flux[:, None, None],
                           ivar[:, None, None])

        return lam, flux, ivar

    # =====
    # properties
    # =====

    @property
    def z_map(self):
        # prepare to de-redshift
        # total redshift of each spaxel
        z_map = (1. + self.z) * (1. + (self.vel / c.c).to('').value) - 1.
        z_map[self.vel_mask] = self.z
        return z_map

    @property
    def SB_map(self):
        # RIMG gives nMgy/pix
        return self.drp_hdulist['RIMG'].data * \
            1.0e-9 * m.Mgy / self.spaxel_side**2.

    @property
    def Reff(self):
        r_ang = self.dap_hdulist['SPX_ELLCOO'].data[0, ...]
        Re_ang = self.drpall_row['nsa_elpetro_th50_r']
        return r_ang / Re_ang

    # =====
    # staticmethods
    # =====

    @staticmethod
    def a_map(f, logl, dlogl):
        lllims = 10.**(logl - 0.5 * dlogl)
        lulims = 10.**(logl + 0.5 * dlogl)
        dl = (lulims - lllims)[:, np.newaxis, np.newaxis]
        return np.mean(f * dl, axis=0)


def determine_eline_mask_dv(dap_hdulist, minimum_value=300., n_times_sigma=2.5):
    '''
    determine the velocity mask width for MaNGA cube
    '''
    sigma = get_emline_qty(dap_hdulist, qty='GSIGMA', key='Ha-6564',
                           sn_th=3., maskbits=range(32))
    # where (sigma < minimum_value) or mask is True, use minimum_value
    dv = (n_times_sigma * sigma.data).clip(min=minimum_value)
    return dv * (u.km / u.s)


def masked_around_line(line_logel, dv_map, obs_logel):
    '''
    make cube mask for a single line
    '''
    dlogel_map = (dv_map / c.c).decompose().value
    logel_mask_l = line_logel - dlogel_map[None, :, :]
    logel_mask_u = line_logel + dlogel_map[None, :, :]
    ismasked = np.logical_and(
        (obs_logel[:, None, None] >= logel_mask_l),
        (obs_logel[:, None, None] <= logel_mask_u))
    return ismasked
