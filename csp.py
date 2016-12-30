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

    def __init__(self, max_bursts=5, override={}, min_dt_cont=.03, RS=None, seed=None):
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
            - zmet: metallicity index
        '''

        self.req_param_keys = [
            'tf', 'tt', 'd1', 'd2', 'ud',  #  delayed tau model, ramp
            'A', 'tb', 'dtb',  #  burst properties
            'tau_V', 'mu', 'zmet', 'sigma']  #  other

        self.FSPS_args = {k: None for k in self.req_param_keys}

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

        # randomize non-overridden parameters
        self.gen_FSPS_args()

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
        self.zmet_gen()

        # bursts
        self.burst_gen()

    def run_fsps(self):
        '''
        run FSPS with given CSP parameters, using continuous SFH

        returns:
         - l: wavelength grid
         - spec: flux-density (Lsol/AA)
         - MLr, MLi, MLz: mass-to-light ratios, in r-, i-, and z-band
        '''

        sp = fsps.StellarPopulation(zcontinuous=1, add_stellar_remnants=True,
                                    smooth_velocity=True, redshift_colors=False,
                                    vactoair_flag=False, tage=self.time0, masscut=150.)
        sp.params['imf_type'] = 2
        sp.params['tage'] = self.time0
        sp.params['sfh'] = 3
        sp.params['logzsol'] = self.FSPS_args['zmet']
        sp.params['sigma_smooth'] = self.FSPS_args['sigma']
        sp.params['dust1'] = self.FSPS_args['tau_V']
        sp.params['dust2'] = self.FSPS_args['tau_V'] * self.FSPS_args['mu']

        sp.set_tabular_sfh(age=self.ts, sfr=self.sfrs)

        l, spec = sp.get_spectrum(tage=self.time0, peraa=True)
        mstar = sp.stellar_mass

        return l, spec, mstar

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
        tab = t.Table(rows=[self.FSPS_args])
        tab.add_column(t.Column(data=[self.Fstar], name='Fstar'))
        tab.add_column(t.Column(data=[self.mass_weighted_age], name='MWA'))

        return tab

    # =====
    # things to generate parameters
    # (allow master overrides)
    # =====

    def time_form_gen(self):
        # if param already set on object instantiation, leave it alone
        if not self.FSPS_args['tf']:
            self.FSPS_args.update(
                {'tf': self.RS.uniform(low=1.0, high=8.)})
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

    def _d1_gen(self):
        if self.FSPS_args['d1']:
            return self.FSPS_args['d1']

        return 1. / self.RS.rand()

    def _tt_gen(self, d1):
        # if param already set on object instantiation, leave it alone
        if self.FSPS_args['tt']:
            return self.FSPS_args['tt']

        tf = self.FSPS_args['tf']

        # transition time can be anytime after tf + 1Gyr
        if tf + d1 > self.time0:
            return self.time0
        else:
            return self.RS.uniform(tf + .5, self.time0)

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
        if self.FSPS_args['d2']:
            return self.FSPS_args['d2']

        if ud < 0.:
            return self.RS.uniform(.1, 2.)
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
        # statistically, one burst occurs in duration time0
        # but forbidden to happen after cutoff

        dt = self.dt_avail

        time_form = self.FSPS_args['tf']

        nburst = self.RS.poisson(dt / self.time0)
        if nburst > self.max_bursts:
            nburst = self.max_bursts

        tb = self._time_burst_gen(nburst, dt)
        dtb = self._dt_burst_gen(nburst)
        A = self._A_burst_gen(nburst)

        self.FSPS_args.update({'tb': tb, 'dtb': dtb, 'A': A, 'nburst': nburst})

    def zmet_gen(self, zsol=zsol_padova):
        if 'zmet' in self.override.keys():
            self.FSPS_args.update({'zmet': self.override['zmet']})

        if self.RS.rand() < .95:
            self.FSPS_args.update(
                {'zmet': np.log10(self.RS.uniform(0.2, 2.5))})

        self.FSPS_args.update({'zmet': np.log10(self.RS.uniform(.02, .2))})

    def tau_V_gen(self):
        if 'tau_V' in self.override.keys():
            self.FSPS_args.update({'tau_V': self.override['tau_V']})

        mu_tau_V = 1.2
        std_tau_V = 1.272  # set to ensure 68% of prob mass lies < 2
        lclip_tau_V, uclip_tau_V = 0., 6.
        a_tau_V = (lclip_tau_V - mu_tau_V) / std_tau_V
        b_tau_V = (uclip_tau_V - mu_tau_V) / std_tau_V

        pdf_tau_V = stats.truncnorm(
            a=a_tau_V, b=b_tau_V, loc=mu_tau_V, scale=std_tau_V)

        tau_V = pdf_tau_V.rvs()

        self.FSPS_args.update({'tau_V': tau_V})

    def mu_gen(self):
        if 'mu' in self.override.keys():
            self.FSPS_args.update({'mu': self.override['mu']})

        mu_mu = 0.3
        std_mu = self.RS.uniform(.1, 1)
        # 68th percentile range means that stdev is in range .1 - 1
        lclip_mu, uclip_mu = 0., 1.
        a_mu = (lclip_mu - mu_mu) / std_mu
        b_mu = (uclip_mu - mu_mu) / std_mu

        pdf_mu = stats.truncnorm(
            a=a_mu, b=b_mu, loc=mu_mu, scale=std_mu)

        mu = pdf_mu.rvs()

        self.FSPS_args.update({'mu': mu})

    def sigma_gen(self):
        if 'sigma' in self.override.keys():
            self.FSPS_args.update({'sigma': self.override['sigma']})

        self.FSPS_args.update({'sigma': self.RS.uniform(10., 400.)})

    # =====
    # properties
    # =====

    @property
    def all_sf_v(self):
        return np.vectorize(
            self.all_sf)

    @property
    def ts(self):
        nburst = self.FSPS_args['nburst']
        burst_starts = self.FSPS_args['tb'][:nburst]
        burst_ends = (self.FSPS_args['tb'] +
                      self.FSPS_args['dtb'])[:nburst]

        discont = self.disconts
        # ages starting at 1Myr, and going to start of SF
        ages = 10.**np.linspace(-3., np.log10(self.time0), 300)
        ts = self.time0 - ages
        ts = np.unique(np.append(ts, discont))
        ts.sort()
        if ts[0] < 0:
            ts[0] = 0.
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
        sfrs = self.sfrs
        disconts = self.disconts

        # integrating tau * SFR(time0 - tau) wrt tau from 0 to time0
        num, numerr = integrate.quad(
            lambda tau: tau * 1.0e9 * self.all_sf(self.time0 - tau), 0., self.time0,
            points=disconts, epsrel=5.0e-3, limit=100)
        denom, denomerr = integrate.quad(
            lambda tau: 1.0e9 * self.all_sf(self.time0 - tau), 0., self.time0,
            points=disconts, epsrel=5.0e-3, limit=100)
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


def make_csp(params={}, return_l=False):
    #print(os.getpid())
    sfh = FSPS_SFHBuilder(max_bursts=5, override=params)

    tab = sfh.to_table()
    l, spec, mstar = sfh.run_fsps()
    mstar = t.Table(
        rows=np.atleast_2d(mstar), names=['mstar'])
    tab = t.hstack([tab, mstar])

    if return_l:
        return spec, tab, l
    else:
        return spec, tab


def make_spectral_library(fname, loc='CSPs', n=1, pkl=True,
                          lllim=3700., lulim=8900., dlogl=1.0e-4,
                          multiproc=False, nproc=8):

    if not pkl:
        if n is None:
            n = 1
        RSs = [np.random.RandomState() for _ in range(n)]
        # generate CSPs and cache them
        CSPs = [FSPS_SFHBuilder(max_bursts=5, RS=rs).FSPS_args
                for rs in RSs]
        with open(os.path.join(loc, '{}.pkl'.format(fname)), 'wb') as f:
            pickle.dump(CSPs, f)
    else:
        with open(os.path.join(loc, '{}.pkl'.format(fname)), 'rb') as f:
            CSPs = pickle.load(f)
        if n is None:
            n = len(CSPs)

    l_final = 10.**np.arange(np.log10(lllim), np.log10(lulim), dlogl)
    # dummy full-lambda-range array
    spec0, meta0, l_full = make_csp(params=CSPs[0], return_l=True)

    if multiproc:
        # poolify the boring stuff
        p = mpc.Pool(processes=nproc, maxtasksperchild=1)
        res = p.map(make_csp, CSPs[1:n], chunksize=1)

        p.close()
        p.join()

        res = [(spec0, meta0), ] + res

        # now build specs and metadata
        specs, metadata = zip(*res)

    else:
        # otherwise, build each entry individually
        # initialize array to hold the spectra
        specs = np.nan * np.ones((n, len(l_full)))

        # initialize list to hold all metadata
        metadata = [None for _ in range(n)]

        metadata[0] = meta0
        specs[0, :] = spec0

        for i in range(1, n):
            specs[i, :], metadata[i] = make_csp(params=CSPs[i])

    # assemble the full table
    metadata = t.vstack(metadata)

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

    hdulist.writeto(os.path.join(loc, '{}.fits'.format(fname)), clobber=True)

# my hobby: needlessly subclassing exceptions

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


class TemplateError(Exception):
    '''
    when there's something wrong with a template
    '''


class TemplateCoverageError(TemplateError):
    '''
    when there's something wrong with the wavelength coverage of a template
    '''
