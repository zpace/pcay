'''
tools for working with FSPS stellar population synthesis library
    (and its python bindings)
'''

import os

import numpy as np
from scipy import stats, integrate
from scipy.interpolate import interp1d

from datetime import datetime
import pickle

import matplotlib.pyplot as plt

from astropy.cosmology import WMAP9
from astropy import units as u, constants as c, table as t
from astropy.io import fits

import fsps

import multiprocessing as mpc

from spectrophot import lumspec2lsun
import utils as ut

zsol_padova = .019
zs_padova = t.Table(
    data=[range(23),
          [0., .0002, .0003, .0004, .0005, .0006, .0008, .001, .0012, .0016,
           .002, .0025, .0031, .0039, .0049, .0061, .0077, .0096, .012,
           .015, .019, .024, .03],
          [-np.inf, -1.98, -1.8, -1.68, -1.58, -1.5, -1.38, -1.28, -1.2, -1.07,
           -.98, -.89, -.79, -.69, -.59, -.49, -.39, -.3, -.2, -.1, 0.,
           .1, .2]],
    names=['zmet', 'Z', 'logZ_Zsol'])
# logs are base-10

eps = np.finfo(float).eps


class FSPS_SFHBuilder(object):
    '''
    programmatically generate lots of SSP characteristics

    conventions:
        - all "time"s are in reference to BB, in Gyr
        - all "age"s are in reference to now, in Gyr
    '''

    __version__ = '0.2'

    def __init__(self, max_bursts=5, override={}, min_dt_cont=.03, RS=None, seed=None,
                 subsample_keys=['tau_V', 'mu', 'sigma'], Nsubsample=20, NBB=3.):
        '''
        set up star formation history generation to use with FSPS

        General Form:

        SFR = {
            0                                 : t < tf
            A (t - tf) exp((t - tf)/d1)       : tf < t < tt
            A (tt - tf) exp((tt - tf)/d1) +
                (t - tt) / d2                 : t > tt
        } * Burst

        arguments:

        **override:
            - tf: time (post-BB) that star formation began
            - tt: time of transition to ramp-up or ramp-down
            - d1: e-folding time of declining tau component
            - d2: doubling/halving time of ramp component
            - A [array]: total stellar mass produced in each starburst,
                relative to total produced by continuous model
            - tb [array]: time (post-BB) that each burst began
            - dtb [array]: duration of each burst
            - tau_V: V-band optical depth affecting young stars
            - mu: fraction of V-band optical depth affecting old stars
            - logzsol: metallicity index
        '''

        self.req_param_keys = [
            'tf', 'tt', 'd1', 'd2', 'ud',  #  delayed tau model, ramp
            'A', 'tb', 'dtb',  #  burst properties
            'tau_V', 'mu', 'logzsol']  #  other

        self.FSPS_args = {k: None for k in self.req_param_keys}

        # manage parameter subsampling
        self.subsample_keys = subsample_keys
        self.Nsubsample = Nsubsample

        # in a full SFH of length time0, avg # of bursts (Poisson)
        self.NBB = NBB

        if not RS:
            self.RS = np.random.RandomState()
        else:
            self.RS = RS

        if not seed:
            pass
        else:
            self.RS.seed(seed)

        self.p_cut = .3
        self.max_bursts = max_bursts
        self.min_dt_cont = min_dt_cont

        self.cosmo = WMAP9
        # how long the universe has been around
        self.time0 = self.cosmo.age(0.).to('Gyr').value

        # initialize parameters
        self.override = override
        self.FSPS_args.update(override)

        # initialize fsps.StellarPopulation object
        self._init_sp_()

        # has the sfh been changed?
        self.sfh_changeflag = False

        now = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.fname = '-'.join(['sfh', now])

    @classmethod
    def from_pickle(cls, fname):
        FSPS_args = pickle.load(open(fname))
        return cls(**FSPS_args)

    def gen_FSPS_args(self):
        '''
        generate FSPS arguments
        '''

        # underlying
        self.time_form_gen()
        self.delay_tau_gen()

        # incidentals
        self.sigma_gen()
        self.tau_V_gen()
        self.mu_gen()
        self.anchor_tau_mu()
        self.logzsol_gen()

        # bursts
        self.burst_gen()

    def _init_sp_(self):
        self.sp = fsps.StellarPopulation(
            zcontinuous=1, add_stellar_remnants=True,
            smooth_velocity=True, redshift_colors=False,
            vactoair_flag=False, tage=self.time0, masscut=150.,
            add_neb_continuum=False)

        self.sp.params['imf_type'] = 2
        self.sp.params['tage'] = self.time0
        self.sp.params['sfh'] = 3
        self.sp.params['dust1'] = 0.
        self.sp.params['dust2'] = 0.

    def run_fsps(self):
        '''
        run FSPS with given CSP parameters, using continuous SFH

        returns:
         - spec: flux-density (Lsol/AA)
        '''

        self.sp.params['logzsol'] = self.FSPS_args['logzsol']

        self.sp.set_tabular_sfh(age=self.ts, sfr=self.sfrs)

        spec = [self._run_fsps_newparams(
                    tage=self.time0,
                    d={'dust1': tau, 'dust2': tau * mu,
                       'sigma_smooth': sigma})
                for tau, mu, sigma in zip(
                    self.FSPS_args['tau_V'], self.FSPS_args['mu'],
                    self.FSPS_args['sigma'])]

        spec = np.row_stack(spec)

        return spec

    def _run_fsps_newparams(self, tage, d):
        for k in d:
            self.sp.params[k] = d[k]

        _, spec = self.sp.get_spectrum(tage=tage, peraa=True)

        self.sfh_changeflag = False

        return spec

    def plot_sfh(self, ts=None, sfrs=None, save=False):

        if ts is None:
            ts = self.ts

        if sfrs is None:
            sfrs = self.sfrs

        tf = self.FSPS_args['tf']

        ts, sfrs = ts[ts > tf], sfrs[ts > tf]

        plt.close('all')

        fig = plt.figure(figsize=(3, 2), dpi=300)
        ax = fig.add_subplot(111)
        ax.plot(ts, sfrs, c='b', linewidth=0.5)
        ax.set_xlabel('time [Gyr]', size=8)
        ax.set_ylabel('SFR [sol mass/yr]', size=8)

        ax.tick_params(labelsize=8)

        # compute y-axis limits
        # cover a dynamic range of a few OOM, plus bursts
        ax.set_ylim([0., sfrs.max() + .1])
        ax.set_xlim([0., self.time0])

        fig.tight_layout()
        if save:
            fig.savefig('.'.join([self.fname, 'png']))
        plt.show()

    def dump(self):
        '''
        dump object to a pickle file
        '''

        with open('.'.join([self.fname, 'sfh']), 'wb') as f:
            pickle.dump(self.FSPS_args, f)

    def to_table(self):

        ks = self.subsample_keys
        tab = t.Table(rows=[self.FSPS_args])

        # make way for properly separated columns!
        for k in ks:
            del tab[k]

        tab.add_column(t.Column(data=[self.Fstar], name='Fstar'))
        tab.add_column(t.Column(data=[self.mass_weighted_age], name='MWA'))
        tab.add_column(t.Column(data=[self.mstar], name='mstar'))

        tab = t.vstack([tab, ] * self.Nsubsample)

        for k in ks:
            tab.add_column(t.Column(data=self.FSPS_args[k], name=k))

        return tab

    # =====
    # things to generate parameters
    # (allow master overrides)
    # =====

    def time_form_gen(self):
        # if param already set on object instantiation, leave it alone
        if 'tf' not in self.override:
            self.FSPS_args.update(
                {'tf': self.RS.uniform(low=0.4, high=13.7)})
            self.sfh_changeflag = True
        else:
            pass

    def delay_tau_gen(self):
        '''
        generate delayed tau-model
        '''

        d1 = self._d1_gen()  #  eft of tau component
        tt = self._tt_gen(d1)  #  transition time
        ud = self._ud_gen()  #  does SFH cut off or increase?
        d2 = self._d2_gen(ud)  #  doubling/halving time for ramp portion

        params = {'d1': d1, 'tt': tt, 'ud': ud, 'd2': d2}

        self.FSPS_args.update(params)
        self.sfh_changeflag = True

    def _d1_gen(self):
        if 'd1' not in self.override:
            return 1. / self.RS.uniform(0.1, 5.)

        return self.FSPS_args['d1']

    def _tt_gen(self, d1, d1delay=False, ddt=0.):
        # if param already set on object instantiation, leave it alone
        if 'tt' in self.override:
            return self.override['tt']

        tf = self.FSPS_args['tf']

        # option to delay ramp by d1 + ddt
        dt = d1 * d1delay + ddt

        # transition time can be anytime after tf + .5Gyr
        if tf + dt > self.time0:
            return self.time0
        else:
            return self.RS.uniform(tf + dt, self.time0)

    def _ud_gen(self):
        '''
        does ramp go up or down?
        '''
        r = self.RS.rand()

        if r < 1. / 3.:
            return -1.
        elif r < 2. / 3.:
            return 0.
        else:
            return 1.

    def _d2_gen(self, ud):
        if 'd2' in self.override:
            return self.FSPS_args['d2']

        elif ud < 0.:
            return self.RS.uniform(.15, 1.5)
        elif ud > 0.:
            return self.RS.uniform(1., 10.)
        else:
            return 100.

    @property
    def dt_avail(self):
        '''
        time duration available for bursts
        '''

        # if ramp-up, then anytime after that is good for bursts
        if self.FSPS_args['ud'] > 0:
            return self.time0 - self.FSPS_args['tf']

        # otherwise, bursts must occur before ramp-down
        return self.FSPS_args['tt'] - self.FSPS_args['tf']

    def _time_burst_gen(self, nburst, dt):
        tf = self.FSPS_args['tf']
        t_ = self.RS.uniform(tf, tf + dt, nburst)
        npad = self.max_bursts - nburst
        t_ = np.pad(t_, (0, npad), mode='constant', constant_values=0.)
        return t_

    def _dt_burst_gen(self, nburst):
        dt_ = self.RS.uniform(.01, .1, nburst)
        npad = self.max_bursts - nburst
        dt_ = np.pad(dt_, (0, npad), mode='constant', constant_values=0.)
        return dt_

    def _A_burst_gen(self, nburst):
        A_ = 10.**self.RS.uniform(np.log10(0.5), np.log10(5.), nburst)
        npad = self.max_bursts - nburst
        A_ = np.pad(A_, (0, npad), mode='constant', constant_values=0.)
        return A_

    def burst_gen(self):
        # statistically, `self.NBB` bursts occur in duration time0
        # but forbidden to happen after cutoff

        dt = self.dt_avail

        time_form = self.FSPS_args['tf']

        nburst = self.RS.poisson(self.NBB * dt / self.time0)
        if nburst > self.max_bursts:
            nburst = self.max_bursts

        tb = self._time_burst_gen(nburst, dt)
        dtb = self._dt_burst_gen(nburst)
        A = self._A_burst_gen(nburst)

        self.FSPS_args.update({'tb': tb, 'dtb': dtb, 'A': A, 'nburst': nburst})
        self.sfh_changeflag = True

    def logzsol_gen(self):

        zsol = self.sp.zlegend / zsol_padova

        d_ = stats.truncexpon(loc=0., scale=1., b=1.)
        d_.random_state = self.RS

        #

        if 'logzsol' in self.override:
            self.FSPS_args.update({'logzsol': self.override['logzsol']})
        # 90% chance of linearly-uniform metallicity range
        elif self.RS.rand() < .9:
            self.FSPS_args.update(
                {'logzsol': ut.lin_transform(
                    r1=[0., 1.], r2=[zsol.max(), zsol.min()], x=d_.rvs())})
        # 10% chance of logarithmically-uniform
        else:
            self.FSPS_args.update(
                {'logzsol': self.RS.uniform(
                    np.log10(zsol.min()), np.log10(zsol.max()))})


    def tau_V_gen(self):
        if 'tau_V' in self.override:
            self.FSPS_args.update({'tau_V': self.override['tau_V']})

        mu_tau_V = 1.2
        std_tau_V = 3.
        lclip_tau_V, uclip_tau_V = 0., 7.
        a_tau_V = (lclip_tau_V - mu_tau_V) / std_tau_V
        b_tau_V = (uclip_tau_V - mu_tau_V) / std_tau_V

        pdf_tau_V = stats.truncnorm(
            a=a_tau_V, b=b_tau_V, loc=mu_tau_V, scale=std_tau_V)
        pdf_tau_V.random_state = self.RS

        tau_V = pdf_tau_V.rvs(size=self.Nsubsample)

        self.FSPS_args.update({'tau_V': tau_V})

    def mu_gen(self):
        if 'mu' in self.override:
            self.FSPS_args.update({'mu': self.override['mu']})

        mu_mu = 0.3
        # std_mu = self.RS.uniform(.1, 1)
        std_mu = 0.3
        # 68th percentile range means that stdev is in range .1 - 1
        lclip_mu, uclip_mu = 0., 1.
        a_mu = (lclip_mu - mu_mu) / std_mu
        b_mu = (uclip_mu - mu_mu) / std_mu

        pdf_mu = stats.truncnorm(
            a=a_mu, b=b_mu, loc=mu_mu, scale=std_mu)
        pdf_mu.random_state = self.RS

        mu = pdf_mu.rvs(size=self.Nsubsample)

        self.FSPS_args.update({'mu': mu})

    def anchor_tau_mu(self):
        '''
        enforce that the first subsample must be:
         - 0 <= tau_V <= 0.2
         - 0.15 <= mu <= 0.45
        '''

        self.FSPS_args['tau_V'][0] = self.RS.uniform(0., .1)
        self.FSPS_args['mu'][0] = self.RS.uniform(.15, .45)

    def sigma_gen(self):

        loc, scale = 10., 350.
        trunc_abs = 350.
        pdf_sigma = stats.truncexpon(b=((scale - loc) / trunc_abs),
                                     loc=loc, scale=scale)
        pdf_sigma.random_state = self.RS

        if 'sigma' in self.override:
            self.FSPS_args.update({'sigma': self.override['sigma']})
        elif 'sigma' in self.subsample_keys:
            self.FSPS_args.update({'sigma': pdf_sigma.rvs(size=self.Nsubsample)})
        else:
            self.FSPS_args.update({'sigma': pdf_sigma.rvs()})

    # =====
    # properties
    # =====

    @property
    def mstar(self):
        if self.sfh_changeflag:
            _ = self.run_fsps()

        return self.sp.stellar_mass

    @property
    def all_sf_v(self):
        return np.vectorize(
            self.all_sf)

    @property
    def ts(self):
        discont = self.disconts
        # ages starting at 1Myr, and going to start of SF
        ages = 10.**np.linspace(-3., np.log10(self.time0), 300)
        ts = self.time0 - ages
        ts = np.unique(np.concatenate(
            [np.array([0, self.FSPS_args['tf']]), ts, discont]))
        ts = ts[ts >= 0]
        ts.sort()
        return ts

    @property
    def sfrs(self):
        return self.all_sf_v(self.ts)

    @property
    def disconts(self):
        burst_starts = self.FSPS_args['tb']
        burst_ends = burst_starts + self.FSPS_args['dtb']
        dt = .01
        burst_starts_m = burst_starts - dt
        burst_ends_p = burst_ends + dt
        sf_starts = self.FSPS_args['tf'] + np.array([-dt, dt])
        points = np.concatenate([sf_starts, burst_starts, burst_starts_m,
                                 burst_ends, burst_ends_p])
        points = points[(0. < points) * (points < self.time0)]
        return points

    @property
    def mformed_integration(self):
        FSPS_args = self.FSPS_args
        mf, mfe = integrate.quad(
            self.all_sf, 0., self.time0,
            points=self.disconts, epsrel=5.0e-3, limit=100)
        return mf * 1.0e9

    @property
    def mass_weighted_age(self):

        ts = self.ts
        ages = self.time0 - ts
        sfrs = self.sfrs

        args = np.argsort(ages)
        ages, sfrs = ages[args], sfrs[args]

        # integrating tau * SFR(time0 - tau) wrt tau from 0 to time0
        num = np.trapz(x=ages, y=ages * sfrs)
        denom = np.trapz(x=ages, y=sfrs)
        return num / denom

    @property
    def Fstar(self):
        '''
        mass fraction formed in last Gyr
        '''
        disconts = self.disconts
        disconts = disconts[disconts > self.time0 - 1.]
        mf, mfe = integrate.quad(
            self.all_sf, self.time0 - 1., self.time0,
            points=disconts, epsrel=5.0e-3, limit=100)
        F = mf * 1.0e9 / self.mformed_integration
        return F

    # =====
    # utility methods
    # =====

    def delay_tau_model(self, t):
        '''
        eval the continuous portion of the SFR at some time

        Note: units are Msun/yr, while the time units are Gyr
        '''
        tf = self.FSPS_args['tf']
        tt = self.FSPS_args['tt']
        d1 = self.FSPS_args['d1']

        return (t - tf) * np.exp(-(t - tf) / d1)

    def ramp(self, t):
        '''
        additive ramp
        '''

        # transition time
        tt = self.FSPS_args['tt']

        # evaluate SFR at transition time
        sfrb = self.delay_tau_model(t=tt)

        # does the ramp rise or fall?
        ud = self.FSPS_args['ud']

        # doubling/halving time
        d2 = self.FSPS_args['d2']

        ramp = sfrb * ((t - tt) / d2)
        if type(t) is np.ndarray:
            ramp[t < tt] = 0.
        else:
            if t < tt:
                ramp = 0.

        return ramp * ud

    def burst_modifier(self, t):
        '''
        evaluate the SFR augmentation at some time
        '''
        nburst = self.FSPS_args['nburst']
        burst_starts = self.FSPS_args['tb'][:nburst]
        burst_ends = (self.FSPS_args['tb'] +
                      self.FSPS_args['dtb'])[:nburst]
        A = self.FSPS_args['A'][:nburst]

        in_burst = ((t >= burst_starts) * (t <= burst_ends))[:nburst]
        return A.dot(in_burst)

    def all_sf(self, t):
        '''
        eval the full SF at some time
        '''

        dtm = self.delay_tau_model(t)
        ramp = self.ramp(t)
        cont = dtm + ramp
        if cont < 0.:
            cont = 0.

        burst = self.burst_modifier(t)

        return cont * (1. + burst)

    def all_sf_a(self, t):
        dtm = self.delay_tau_model(t)
        ramp = self.ramp(t)
        cont = dtm + ramp
        cont[cont < 0.] = 0.

        burst = self.burst_modifier(t)

        return cont * (1. + burst)

    # =====
    # utility methods
    # =====

    def __repr__(self):
        return '\n'.join(
            ['{}: {}'.format(k, v) for k, v in self.FSPS_args.items()])


def make_csp(sfh):
    sfh.gen_FSPS_args()
    spec = sfh.run_fsps()
    tab = sfh.to_table()

    return spec, tab, sfh.FSPS_args


def make_spectral_library(fname, sfh, loc='CSPs', n=1, lllim=3700., lulim=8900.,
                          dlogl=1.0e-4):

    l_full = sfh.sp.wavelengths
    l_final = 10.**np.arange(np.log10(lllim), np.log10(lulim), dlogl)

    specs, metadata, dicts = zip(*[make_csp(sfh) for _ in range(n)])

    # write out dict to pickle
    with open(os.path.join(loc, '{}.pkl'.format(fname)), 'wb') as f:
        pickle.dump(dicts, f)

    # assemble the full table & spectra
    metadata = t.vstack(metadata)
    specs = np.row_stack(specs)

    # find luminosity
    Lr = lumspec2lsun(lam=l_full * u.AA,
                      Llam=specs * u.Unit('Lsun/AA'), band='r')
    Li = lumspec2lsun(lam=l_full * u.AA,
                      Llam=specs * u.Unit('Lsun/AA'), band='i')
    Lz = lumspec2lsun(lam=l_full * u.AA,
                      Llam=specs * u.Unit('Lsun/AA'), band='z')

    specs_interp = interp1d(x=l_full, y=specs, kind='linear', axis=-1)
    specs_reduced = specs_interp(l_final)

    MLr, MLi, MLz = (metadata['mstar'] / Lr,
                     metadata['mstar'] / Li,
                     metadata['mstar'] / Lz)
    ML = t.Table(data=[MLr, MLi, MLz], names=['MLr', 'MLi', 'MLz'])
    metadata = t.hstack([metadata, ML])

    # initialize FITS HDUList
    hdulist = fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU(np.array(metadata)),
         fits.ImageHDU(l_final), fits.ImageHDU(specs_reduced)])
    hdulist[1].header['EXTNAME'] = 'meta'
    hdulist[2].header['EXTNAME'] = 'lam'
    hdulist[2].header['DLOGL'] = dlogl
    hdulist[3].header['EXTNAME'] = 'flam'
    '''
    extension list:
     - [1], 'meta': FITS table equal to `metadata`
     - [2], 'lam': array with lambda, and dlogl header keyword
     - [3], 'flam': array with lum-densities for each CSP on same
        wavelength grid
    '''

    hdulist.writeto(os.path.join(loc, '{}.fits'.format(fname)), overwrite=True)

    return sfh

def random_SFH_plots(n=10, save=False):
    from time import time

    fig = plt.figure(figsize=(3, 2), dpi=300)
    ax = fig.add_subplot(111)

    # make color cycle
    c_ix = np.linspace(0., 1., n)
    RSs = [np.random.RandomState() for _ in range(n)]

    for c, RS in zip(c_ix, RSs):
        sfh = FSPS_SFHBuilder(RS=RS)

        ts = sfh.ts
        sfrs = sfh.sfrs

        tf = sfh.FSPS_args['tf']
        ts, sfrs = ts[ts > tf], sfrs[ts > tf]

        ax.plot(ts, sfrs, color=plt.cm.viridis(c),
                linewidth=0.5)

        del sfh

    ax.set_xlabel('time [Gyr]', size=8)
    ax.set_ylabel('SFR [sol mass/yr]', size=8)

    ax.tick_params(labelsize=8)

    # compute y-axis limits
    # cover a dynamic range of a few OOM, plus bursts
    ax.set_xlim([0., 13.7])

    fig.tight_layout()
    if save:
        fig.savefig('.'.join([self.fname, 'png']))
    else:
        plt.show()


# my hobby: needlessly subclassing exceptions

class TemplateError(Exception):
    '''
    when there's something wrong with a template
    '''


class TemplateCoverageError(TemplateError):
    '''
    when there's something wrong with the wavelength coverage of a template
    '''


if __name__ == '__main__':

    nfiles, nper, Nsubsample = 50, 100, 10
    name_ix0 = 0
    name_ixf = name_ix0 + nfiles

    RS = np.random.RandomState()
    sfh = FSPS_SFHBuilder(RS=RS, Nsubsample=Nsubsample, max_bursts=10)

    for i in range(name_ix0, name_ixf):
        sfh = make_spectral_library(
            sfh=sfh, fname='CSPs_{}'.format(i), loc='CSPs_CKC14_MaNGA_new',
            n=nper, lllim=3800., lulim=9400., dlogl=1.0e-4)
        print('Done with {} of {}'.format(i + 1, name_ixf))
