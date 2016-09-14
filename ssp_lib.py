'''
tools for working with FSPS stellar population synthesis library
    (and its python bindings)
'''

import numpy as np
from scipy import stats, integrate

from datetime import datetime
import pickle

import matplotlib.pyplot as plt

from astropy.cosmology import WMAP9
from astropy import units as u, constants as c, table as t
from astropy.io import fits

# import fsps

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


class FSPS_SFHBuilder(object):
    '''
    programmatically generate lots of SSP characteristics

    conventions:
        - all "time"s are in reference to BB, in Gyr
        - all "age"s are in reference to now, in Gyr
    '''

    __version__ = '0.1'

    def __init__(self, **kwargs):
        '''
        set up star formation history generation to use with FSPS

        arguments:

        **kwargs:
            - time_form: time (post-BB) that star formation began
            - eftu: e-folding time [Gyr] for underlying continuous model
            - A [array]: total stellar mass produced in each starburst,
                relative to total produced by continuous model
            - time_burst [array]: time (post-BB) that each burst began
            - dt_burst [array]: duration of each burst
            - time_cut: time (post-BB) of an exponential cutoff in SFH
            - eftc: e-folding time of exponential cutoff
            - tau_V: V-band optical depth affecting young stars
            - mu: fraction of V-band optical depth affecting old stars
            - zmet: metallicity index
        '''

        self.p_cut = .3

        self.cosmo = WMAP9
        # how long the universe has been around
        self.time0 = self.cosmo.age(0.).value

        # for each of the parameters equal to None, reset the associated
        # instance attribute to the callable function that generates the
        # default distribution

        # pack into dict for easy stringification, and update with new values

        self.FSPS_args = {
            'time_form': np.nan, 'eftu': np.nan, 'A': np.nan,
            'time_burst': np.nan, 'dt_burst': np.nan,
            'time_cut': np.nan, 'eftc': np.nan, 'tau_V': np.nan,
            'mu': np.nan, 'zmet': np.nan, 'sigma': np.nan}
        self.FSPS_args.update(kwargs)

        self.FSPS_args['time_form'] = self.time_form_gen(
            override=self.FSPS_args['time_form'])

        self.FSPS_args['eftu'] = self.eftu_gen(
            override=self.FSPS_args['eftu'])

        self.FSPS_args['time_cut'], self.FSPS_args['eftc'] = \
            self.cut_gen(
                self.FSPS_args['time_form'], self.time0,
                override=(self.FSPS_args['time_cut'],
                          self.FSPS_args['eftc']))

        self.FSPS_args['time_burst'], self.FSPS_args['dt_burst'], \
            self.FSPS_args['A'] = self.burst_gen(
                self.FSPS_args['time_form'], self.time0,
                self.FSPS_args['time_cut'],
                override=(self.FSPS_args['time_burst'],
                          self.FSPS_args['dt_burst'],
                          self.FSPS_args['A']))
        self.nburst = len(self.FSPS_args['time_burst'])

        self.FSPS_args['tau_V'] = self.tau_V_gen(
            override=self.FSPS_args['tau_V'])

        self.FSPS_args['mu'] = self.mu_gen(
            override=self.FSPS_args['mu'])

        self.FSPS_args['zmet'] = self.zmet_gen(
            override=self.FSPS_args['zmet'])

        self.FSPS_args['sigma'] = self.sigma_gen(
            override=self.FSPS_args['sigma'])

        now = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.fname = '-'.join(['sfh', now])

    @classmethod
    def from_pickle(cls, fname):
        FSPS_args = pickle.load(open(fname))
        return cls(**FSPS_args)

    def run_fsps(self):
        '''
        run FSPS with given CSP parameters, using continuous SFH

        returns:
         - l: wavelength grid
         - spec: flux-density (Lsol/AA)
         - MLr, MLi, MLz: mass-to-light ratios, in r-, i-, and z-band
        '''
        sp = fsps.StellarPopulation(zcontinuous=1)
        sp.params['imf_type'] = 2
        sp.params['tage'] = self.time0
        sp.params['sfh'] = 3
        sp.params['logzsol'] = self.FSPS_args['zmet']
        sp.params['sigma_smooth'] = self.FSPS_args['sigma']
        sp.params['dust1'] = self.FSPS_args['tau_V']
        sp.params['dust2'] = self.FSPS_args['tau_V'] * self.FSPS_args['mu']
        sp.set_tabular_sfh(age=self.ts, sfr=self.sfrs)

        l, spec = sp.get_spectrum(tage=self.time0, peraa=True)
        mstar = sp.stellar_mass * u.Msun
        Ms = sp.get_mags(
            bands=['sdss_i', 'sdss_r', 'sdss_z'], tage=self.time0)
        Ls = 10.**(-0.4 * (Ms - 4.85)) * u.Lsun

        MLs = mstar / Ls
        return l, spec, MLs.value

    def calc_sfh(self, plot=False, saveplot=False, mformed_compare=False):
        '''
        return an array of times (rel to BB) and SFRs (Msol/yr)
        '''

        FSPS_args = self.FSPS_args
        mtot_cont = self.mtot_cont(
            FSPS_args['time_form'], self.time0, FSPS_args['eftu'],
            FSPS_args['time_cut'], FSPS_args['eftc'])

        # calculate the total mass formed in bursts
        mtot_burst = FSPS_args['A'].sum() * mtot_cont
        mtot = mtot_burst + mtot_cont

        self.mformed_tot = mtot
        self.mformed_cont = mtot_cont
        self.mformed_burst = mtot_burst

        sfrs = self.sfrs
        ts = self.ts

        if mformed_compare:
            print('Total mass formed')
            print('\tintegral:', self.mformed_integration)
            print('\tsummation:', self.sfrs.sum() * (ts[1] - ts[0]))

        if plot:
            self.plot_sfh(ts=ts, sfrs=sfrs, save=saveplot)

    def plot_sfh(self, ts=None, sfrs=None, save=False):

        if ts is None:
            ts = self.ts

        if sfrs is None:
            sfrs = self.sfrs

        plt.close('all')

        plt.figure(figsize=(3, 2), dpi=300)
        ax = plt.subplot(111)
        ax.plot(self.ts, self.sfrs, c='b', linewidth=0.5)
        ax.set_xlabel('time [Gyr]', size=8)
        ax.set_ylabel('SFR [sol mass/Gyr]', size=8)
        ax.set_yscale('log')
        ax.tick_params(labelsize=8)

        # compute y-axis limits
        # cover a dynamic range of a few OOM, plus bursts
        ylim_cont = [5.0e-3, 1.25]
        ylim = ylim_cont
        if self.nburst > 0:
            ylim[1] = 1.25 * sfrs.max()

        ax.set_ylim(ylim)

        plt.tight_layout()
        if save:
            plt.savefig('.'.join([self.fname, 'png']))
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

        goodcolumns = [n for n in tab.colnames if tab[n].dtype is float]

        return tab[goodcolumns]

    # =====
    # properties
    # =====

    @property
    def all_sf_v(self):
        return np.vectorize(
            self.all_sf,
            excluded=set(self.FSPS_args.keys() + ['time0', ]),
            cache=True)

    @property
    def ts(self):
        burst_ends = self.FSPS_args['time_burst'] + self.FSPS_args['dt_burst']
        discont = np.append(self.FSPS_args['time_burst'], burst_ends)
        ts = np.linspace(0., self.time0, 100)
        ts = np.append(ts, discont)
        ts.sort()
        return ts

    @property
    def sfrs(self):
        return self.all_sf_v(self.ts, time0=self.time0, **self.FSPS_args)

    @property
    def disconts(self):
        burst_ends = self.FSPS_args['time_burst'] + self.FSPS_args['dt_burst']
        points = np.append(self.FSPS_args['time_burst'], burst_ends)
        return points

    @property
    def mformed_integration(self):
        FSPS_args = self.FSPS_args
        mf = integrate.quad(
            self.all_sf, 0., self.time0, args=(
                FSPS_args['time_form'], FSPS_args['eftu'],
                FSPS_args['time_cut'], FSPS_args['eftc'],
                FSPS_args['time_burst'], FSPS_args['dt_burst'],
                FSPS_args['A'], self.time0),
            points=self.disconts, epsrel=5.0e-3)[0]
        return mf

    @property
    def mass_weighted_age(self):

        disconts = self.disconts
        mtot_cont = self.mtot_cont(
            self.FSPS_args['time_form'], self.time0, self.FSPS_args['eftu'],
            self.FSPS_args['time_cut'], self.FSPS_args['eftc'])

        def num_integrand(tau, mtot_cont):
            return tau * self.all_sf(
                self.time0 - tau, mtot_cont=mtot_cont,
                time0=self.time0, **self.FSPS_args)

        def denom_integrand(tau, mtot_cont):
            return self.all_sf(
                self.time0 - tau, mtot_cont=mtot_cont,
                time0=self.time0, **self.FSPS_args)

        # integrating tau * SFR(time0 - tau) wrt tau from 0 to time0
        num = integrate.quad(
            num_integrand, 0., self.time0, args=(mtot_cont),
            points=disconts, epsrel=5.0e-3)[0]
        denom = integrate.quad(
            denom_integrand, 0., self.time0, args=(mtot_cont),
            points=disconts, epsrel=5.0e-3)[0]
        return num / denom

    @property
    def Fstar(self):
        '''
        mass fraction formed in last Gyr
        '''
        FSPS_args = self.FSPS_args
        disconts = self.disconts[
            (self.time0 - 1. < self.disconts) * (self.disconts < self.time0)]
        mformed_lastGyr = integrate.quad(
            self.all_sf, self.time0 - 1., self.time0, args=(
                FSPS_args['time_form'], FSPS_args['eftu'],
                FSPS_args['time_cut'], FSPS_args['eftc'],
                FSPS_args['time_burst'], FSPS_args['dt_burst'],
                FSPS_args['A'], self.time0),
            points=disconts, epsrel=5.0e-3)[0]
        F = mformed_lastGyr / self.mformed_integration
        return F

    # =====
    # static methods
    # (allow master overrides)
    # =====

    @staticmethod
    def time_form_gen(override):
        if not np.isnan(override):
            return override

        return np.random.uniform(low=1.5, high=13.5)

    @staticmethod
    def eftu_gen(override):
        if not np.isnan(override):
            return override

        return 1. / np.random.rand()

    @staticmethod
    def time_cut_gen(time_form, time0, override):
        if not np.isnan(override):
            return override

        return np.random.uniform(time_form, time0)

    @staticmethod
    def eftc_gen(override):
        if not np.isnan(override):
            return override

        return 10.**np.random.uniform(-2, 0)

    @staticmethod
    def cut_gen(time_form, time0, override):
        # does cutoff occur?
        if None in override:
            return None, None
        if np.isnan(override).sum() < 2:
            # if any override has been given, there is a cutoff
            cut_yn = True
        else:
            cut_yn = np.random.rand() < .3

        if not cut_yn:
            return (None, None)

        time_cut = FSPS_SFHBuilder.time_cut_gen(
            time_form, time0, override[0])
        eftc = FSPS_SFHBuilder.eftc_gen(override[1])

        return time_cut, eftc

    @staticmethod
    def time_burst_gen(time_form, dt, nburst):
        return np.random.uniform(time_form, time_form + dt, nburst)

    @staticmethod
    def dt_burst_gen(nburst):
        return np.random.uniform(.01, .1, nburst)

    @staticmethod
    def A_burst_gen(nburst):
        return 10.**np.random.uniform(np.log10(.03), np.log10(4.), nburst)

    @staticmethod
    def burst_gen(time_form, time0, time_cut, override):
        '''
        allow bursts to be partially specified...

        `override` has 3 elements: time_burst, dt_burst, and A (mass ampl.)

        IMPORTANT: if there's no burst at all, then len-0 arrays should
        be specified for each of the elements of `override`
        '''

        time_burst, dt_burst, A = override

        # statistically, one burst occurs in duration time0
        # but forbidden to happen after cutoff
        if time_cut is not None:
            dt = time_cut - time_form
        else:
            dt = time0 - time_form

        # =====

        # handle case that a burst is forbidden, i.e., override is 3 Nones
        if time_burst is None and dt_burst is None and A is None:
            return (np.array([]), ) * 3

        # =====

        # handle the case that no burst information is given (3 NaNs)
        if ((np.isnan(time_burst).sum() *
             np.isnan(dt_burst).sum() *
             np.isnan(A).sum()) == 1) and \
                (np.ndarray not in map(type, override)):

            # number of bursts
            nburst = stats.poisson.rvs(dt / time0)
            # when they occur
            time_burst = FSPS_SFHBuilder.time_burst_gen(time_form, dt, nburst)
            # how long they last
            dt_burst = FSPS_SFHBuilder.dt_burst_gen(nburst)
            # how much mass each forms
            A = FSPS_SFHBuilder.A_burst_gen(nburst)

            return time_burst, dt_burst, A

        # =====

        # handle case that single burst is specified, and move forward
        if type(time_burst) is not np.ndarray:
            time_burst = np.array([time_burst])
        if type(dt_burst) is not np.ndarray:
            dt_burst = np.array([dt_burst])
        if type(A) is not np.ndarray:
            A = np.array([A])

        # =====

        # handle case of arbitrary number of bursts

        # make sure that lengths of all provided burst parameter
        # arrays are equal. Otherwise iterating over them will fail.
        assert len(time_burst) == len(dt_burst) == len(A), \
            'override elements have differing lengths ({}, {}, {})'.format(
                len(time_burst), len(dt_burst), len(A))

        # iterate through expressions for each burst
        # IMPORTANT: if there's no burst, then len-0 arrays
        # (i.e., np.array([])) should be specified for each
        # of the components of `override`
        for i in range(len(time_burst)):
            # if no burst time has been specified, generate one
            if np.isnan(time_burst[i]):
                time_burst[i] = FSPS_SFHBuilder.time_burst_gen(
                    time_form, dt, 1)
            # if no burst duration has been specified, generate one
            if np.isnan(dt_burst[i]):
                dt_burst[i] = FSPS_SFHBuilder.dt_burst_gen(1)
            # if no burst amplitude (mass) has been specified, generate one
            if np.isnan(A[i]):
                A[i] = FSPS_SFHBuilder.A_burst_gen(1)

        return time_burst, dt_burst, A

    @staticmethod
    def zmet_gen(override, zsol=zsol_padova):
        if not np.isnan(override):
            return override

        if np.random.rand() < .95:
            return np.random.uniform(0.2 * zsol, 2.5 * zsol)

        return np.random.uniform(.02 * zsol, .2 * zsol)

    @staticmethod
    def tau_V_gen(override):
        if not np.isnan(override):
            return override

        mu_tau_V = 1.2
        std_tau_V = 1.272  # set to ensure 68% of prob mass lies < 2
        lclip_tau_V, uclip_tau_V = 0., 6.
        a_tau_V = (lclip_tau_V - mu_tau_V) / std_tau_V
        b_tau_V = (uclip_tau_V - mu_tau_V) / std_tau_V

        pdf_tau_V = stats.truncnorm(
            a=a_tau_V, b=b_tau_V, loc=mu_tau_V, scale=std_tau_V)

        tau_V = pdf_tau_V.rvs()

        return tau_V

    @staticmethod
    def mu_gen(override):
        if not np.isnan(override):
            return override

        mu_mu = 0.3
        std_mu = np.random.uniform(.1, 1)
        # 68th percentile range means that stdev is in range .1 - 1
        lclip_mu, uclip_mu = 0., 1.
        a_mu = (lclip_mu - mu_mu) / std_mu
        b_mu = (uclip_mu - mu_mu) / std_mu

        pdf_mu = stats.truncnorm(
            a=a_mu, b=b_mu, loc=mu_mu, scale=std_mu)

        mu = pdf_mu.rvs()

        return mu

    @staticmethod
    def sigma_gen(override):
        if not np.isnan(override):
            return override

        return np.random.uniform(50., 400.)

    @staticmethod
    def continuous_sf(t, time_form, eftu, time_cut, eftc):
        '''
        eval the continuous portion of the SFR at some time
        '''
        if t < time_form:
            return 0.

        return FSPS_SFHBuilder.cut_modifier(
            t, time_cut, eftc) * np.exp(-(t - time_form) / eftu)

    @staticmethod
    def cut_modifier(t, time_cut, eftc):
        '''
        eval the SF cut's contribution to the overall SFR at some time
        '''
        if time_cut is None:
            return 1.
        if t < time_cut:
            return 1.

        return np.exp(-(t - time_cut) / eftc)

    @staticmethod
    def mtot_cont(time_form, time0, eftu, time_cut, eftc):
        # calculate total stellar mass formed in the continuous bit
        mtot_cont = integrate.quad(
            FSPS_SFHBuilder.continuous_sf, 0, time0,
            args=(time_form, eftu, time_cut, eftc))
        return mtot_cont[0]

    @staticmethod
    def all_sf(t, time_form, eftu, time_cut, eftc, time_burst,
               dt_burst, A, time0, mtot_cont=None, **kwargs):
        '''
        eval the full SF at some time

        **kwargs are ignored
        '''

        if not mtot_cont:
            mtot_cont = FSPS_SFHBuilder.mtot_cont(
                time_form, time0, eftu, time_cut, eftc)

        continuous = FSPS_SFHBuilder.continuous_sf(
            t, time_form, eftu, time_cut, eftc)

        if len(time_burst) == 0:
            return continuous

        t_a = t * np.ones_like(time_burst)
        in_burst = (t_a > time_burst) * (t_a < time_burst + dt_burst)

        burst = (mtot_cont * A / dt_burst).dot(in_burst)

        return continuous + burst

    # =====
    # utility methods
    # =====

    def __repr__(self):
        return '\n'.join(
            ['{}: {}'.format(k, v) for k, v in self.FSPS_args.iteritems()])


def make_csp():
    sfh = FSPS_SFHBuilder()
    tab = sfh.to_table()
    del tab['A']
    del tab['dt_burst']
    del tab['time_burst']
    del tab['eftc']
    del tab['time_cut']
    print(tab.dtype)
    l, spec, MLs = sfh.run_fsps()
    MLs = t.Table(
        rows=np.atleast_2d(MLs), names=['MLr', 'MLi', 'MLz'])
    tab = t.hstack([tab, MLs])
    return l, spec, tab


def make_spectral_library(n=1, pkl=False, lllim=3700., lulim=4700.):

    if not pkl:
        # generate CSPs and cache them
        CSPs = [FSPS_SFHBuilder().FSPS_args for _ in range(n)]
        pickle.dump(CSPs, open('csps.pkl', 'wb'))
    else:
        CSPs = pickle.load(open('csps.pkl', 'wb'))
        n = len(CSPs)

    l_final = 10.**np.arange(np.log10(lllim), np.log10(lulim), 1.0e-4)

    # initialize array to hold the spectra
    specs = np.nan * np.ones((n, len(l_final)))

    # initialize list to hold all metadata
    metadata = [None for _ in range(n)]

    # build each entry individually
    for i in range(n):
        sfh = FSPS_SFHBuilder(**CSPs[i])
        l, spec, tab_ = make_csp()
        metadata[i] = tab_
        specs[i, :] = np.interp(x=l_final, xp=l, fp=spec)

    # assemble the full table
    metadata = t.vstack(metadata)

    # initialize FITS HDUList
    hdulist = fits.HDUList(
        [fits.BinTableHDU(np.array(metadata)), fits.ImageHDU(l),
         fits.ImageHDU(specs)])
    '''
    extension list:
     - [0], 'meta': FITS table equal to `metadata`
     - [1], 'loglam': array with log-lambda, and dlogl header keyword
     - [2], 'flam': array with flux-densities for each CSP on same
        wavelength grid
    '''

    hdulist.writeto('CSP_spectra.fits', clobber=True)

# my hobby: needlessly subclassing exceptions


class TemplateError(Exception):
    '''
    when there's something wrong with a template
    '''


class TemplateCoverageError(TemplateError):
    '''
    when there's something wrong with the wavelength coverage of a template
    '''

'''
to do:
 - r-band luminosity-weighted age
 - mass-weighted age
 - i- and z-band mass-to-light ratio
'''
