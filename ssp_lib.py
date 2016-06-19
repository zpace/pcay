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

import fsps

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
            'time_form': None, 'eftu': None, 'A': np.array([]),
            'time_burst': np.array([]), 'dt_burst': np.array([]),
            'time_cut': None, 'eftc': None, 'tau_V': None, 'mu': None,
            'zmet': None, 'sigma': None}
        self.FSPS_args.update(kwargs)

        self.FSPS_args['time_form'] = self.time_form_gen(
            override = self.FSPS_args['time_form'])

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
                override=(self.FSPS_args['time_burst'], \
                          self.FSPS_args['dt_burst'], \
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

        burst_ends = FSPS_args['time_burst'] + FSPS_args['dt_burst']
        discont = np.append(FSPS_args['time_burst'], burst_ends)

        mtot_integration = integrate.quad(
            self.all_sf, 0., self.time0, args=(
                FSPS_args['time_form'], FSPS_args['eftu'],
                FSPS_args['time_cut'], FSPS_args['eftc'],
                FSPS_args['time_burst'], FSPS_args['dt_burst'],
                FSPS_args['A'], self.time0),
            points=discont, epsrel=5.0e-3)[0]

        sfrs = self.sfrs
        ts = self.ts

        if mformed_compare:
            print 'Total mass formed'
            print '\tintegral:', self.mformed_tot
            print '\tsummation:', self.sfrs.sum() * (ts[1] - ts[0])

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
            ylim[1] = 1.25*sfrs.max()

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
            pickle.dump(self, f)

    def write_sfh(self, mformed_tot_force=1.0e6):
        '''
        translate computed SFH into array of constant SFHs
        '''

        # load star formation rate, and turn it into Msol/yr,
        # assuming total mass formed of 1.0e6 Msol
        sfrs = self.sfrs / 1.0e9 * (mformed_tot_force / self.mformed_tot)
        ts = self.ts
        # star formation starts at self.FSPS_params['time_start']
        # and effectively ends when the SFR drops below .01 Msol/yr
        sf_start = self.FSPS_args['time_start']
        sf_end = ts[sfrs < .01][0]

        #inject bursts articificially
        burst_start = self.FSPS_args['time_burst']
        burst_end = self.FSPS_args['dt_burst']
        burst_points = np.column_stack(burst_start, burst_end).flatten()

    def generate_full_spectrum(self, lllim=3700., lulim=10200.):
        # if there are no bursts, then just run a single underlying
        # continuous SFH

        # compute mass fractions from each burst and continuous portion
        mfrac_cont = 1./(1. + self.FSPS_args['A'].sum())
        mfrac_b = self.FSPS_args['A'] / (1. + self.FSPS_args['A'].sum())

        # start out with continuous, underlying portion of SFH
        h_cont = self.run_fsps_continuous(
            time_form=self.FSPS_args['time_form'],
            eftu=self.FSPS_args['eftu'], time0=self.time0,
            tau_V=self.FSPS_args['tau_V'], mu=self.FSPS_args['mu'],
            zmet=self.FSPS_args['zmet'], sigma=self.FSPS_args['sigma'])
        l_cont, s_cont = h_cont.get_spectrum(tage=self.time0, peraa=True)
        mstar_cont = h_cont.stellar_mass

        l_cont, s_cont = l_cont[(l_cont > lllim) * (l_cont < lulim)], \
                         s_cont[(l_cont > lllim) * (l_cont < lulim)]

        if self.nburst == 0:
            l, s = l_cont, s_cont
            return l, s, mstar_cont

        # manage starbursts
        s_bursts = [None, ] * self.nburst
        mstar_bursts = [None, ] * self.nburst
        for i in range(self.nburst):
            h_b = self.run_fsps_burst(
                time_form=self.FSPS_args['time_form'],
                eftu=self.FSPS_args['eftu'], time0=self.time0,
                tau_V=self.FSPS_args['tau_V'], mu=self.FSPS_args['mu'],
                zmet=self.FSPS_args['zmet'], sigma=self.FSPS_args['sigma'],
                time_burst=self.FSPS_args['time_burst'])
            l_b, s_b = h_b.get_spectrum(tage=self.time0, peraa=True)
            mstar_b = h_b.stellar_mass

            mstar_bursts[i] = mfrac_cont * mfrac_b[i]

            s_b *= (mstar_bursts[i] / mstar_b)
            l_b, s_b = l_b[(l_b > lllim) * (l_b < lulim)], \
                       s_b[(l_b > lllim) * (l_b < lulim)]
            s_bursts[i] = s_b

        s = np.array(s_bursts).sum(axis=0) + s_cont
        l = l_cont

        mtot = mstar_cont + np.array(mstar_bursts).sum()

        return l, s, mtot


    @property
    def all_sf_v(self):
        return np.vectorize(
            self.all_sf,
            excluded=set(self.FSPS_args.keys() + ['time0',]),
            cache=True)

    @property
    def ts(self):
        return np.linspace(0., self.time0, 10000)

    @property
    def sfrs(self):
        return self.all_sf_v(self.ts, time0=self.time0, **self.FSPS_args)

    #=====
    # static methods
    # (allow master overrides)
    #=====

    @staticmethod
    def time_form_gen(override):
        if override is not None:
            return override

        return np.random.uniform(low=1.5, high=13.5)

    @staticmethod
    def eftu_gen(override):
        if override is not None:
            return override

        return 1./np.random.rand()

    @staticmethod
    def cut_gen(time_form, time0, override):
        if override != (None, None):
            return override

        # does cutoff occur?
        cut_yn = np.random.rand() < .3
        if not cut_yn:
            return (None, None)

        time_cut = np.random.uniform(time_form, time0)
        eftc = 10.**np.random.uniform(-2, 0)

        return time_cut, eftc

    @staticmethod
    def burst_gen(time_form, time0, time_cut, override):
        if map(len, override) != [0, 0, 0]:
            return override

        # statistically, one burst occurs in duration time0
        # but forbidden to happen after cutoff
        if time_cut is not None:
            dt = time_cut - time_form
        else:
            dt = time0 - time_form

        # number of bursts
        nburst = stats.poisson.rvs(dt/time0)
        # when they occur

        time_burst = np.random.uniform(time_form, time_form + dt, nburst)
        # how long they last
        dt_burst = np.random.uniform(.01, .1, nburst)
        # how much mass each forms
        A = 10.**np.random.uniform(np.log10(.03), np.log10(4.), nburst)

        return time_burst, dt_burst, A

    @staticmethod
    def zmet_gen(override, zsol=zsol_padova):
        if override is not None:
            return override

        if np.random.rand() < .95:
            return np.random.uniform(0.2*zsol, 2.5*zsol)

        return np.random.uniform(.02*zsol, .2*zsol)

    @staticmethod
    def tau_V_gen(override):
        if override is not None:
            return override

        mu_tau_V = 1.2
        std_tau_V = 1.272 # set to ensure 68% of prob mass lies < 2
        lclip_tau_V, uclip_tau_V = 0., 6.
        a_tau_V = (lclip_tau_V - mu_tau_V) / std_tau_V
        b_tau_V = (uclip_tau_V - mu_tau_V) / std_tau_V

        pdf_tau_V = stats.truncnorm(
            a=a_tau_V, b=b_tau_V, loc=mu_tau_V, scale=std_tau_V)

        tau_V = pdf_tau_V.rvs()

        return tau_V

    @staticmethod
    def mu_gen(override):
        if override is not None:
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
        if override != None:
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
            t, time_cut, eftc) * np.exp(-(t - time_form)/eftu)

    @staticmethod
    def cut_modifier(t, time_cut, eftc):
        '''
        eval the SF cut's contribution to the overall SFR at some time
        '''
        if time_cut is None:
            return 1.
        if t < time_cut:
            return 1.

        return np.exp(-(t - time_cut)/eftc)

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

        t_a = t*np.ones_like(time_burst)
        in_burst = (t_a > time_burst) * (t_a < time_burst + dt_burst)

        burst = (mtot_cont * A / dt_burst).dot(in_burst)

        return continuous + burst

    @staticmethod
    def run_fsps_continuous(time_form, eftu, time0, tau_V, mu, zmet, sigma):
        '''
        models underlying continuous SFH. DO NOT USE FOR BURSTS
        '''
        sp = fsps.StellarPopulation(
            compute_vega_mags=False, vactoair_flag=False, smooth_velocity=True,
            add_stellar_remnants=True, imf_type=2, # Kroupa IMF
            tau=eftu, const=0, tage=time0, fburst=0., tburst=999., # no burst
            dust1=tau_V, dust2=tau_V*mu, zcontinuous=1,
            logzsol=np.log10(zmet), sf_start=time_form, sf_trunc=0.,
            sf_slope=0., masscut=150., sfh=1, # five-parameter SFH
            sigma_smooth=sigma)
        return sp

    @staticmethod
    def run_fsps_burst(time_form, eftu, time0, tau_V, mu, zmet, sigma,
                       time_burst):
        '''
        models single SF burst only
        '''
        sp = fsps.StellarPopulation(
            compute_vega_mags=False, vactoair_flag=False, smooth_velocity=True,
            add_stellar_remnants=True, imf_type=2, # Kroupa IMF
            tau=eftu, const=0, tage=time0, fburst=1., tburst=time_burst,
            dust1=tau_V, dust2=tau_V*mu, zcontinuous=1,
            logzsol=np.log10(zmet), sf_start=time_form, sf_trunc=0.,
            sf_slope=0., masscut=150., sfh=1, # five-parameter SFH
            sigma_smooth=sigma)
        return sp



    #=====
    # utility methods
    #=====

    def __repr__(self):
        return '\n'.join(
            ['{}: {}'.format(k, v) for k, v in self.FSPS_args.iteritems()])
