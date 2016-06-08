'''
tools for working with FSPS stellar population synthesis library
    (and its python bindings)
'''

import numpy as np
from scipy import stats, integrate

import matplotlib.pyplot as plt

from astropy.cosmology import WMAP9
from astropy import units as u, constants as c, table as t

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

        self.FSPS_args['tau_V'] = self.tau_V_gen(
            override=self.FSPS_args['tau_V'])

        self.FSPS_args['mu'] = self.mu_gen(
            override=self.FSPS_args['mu'])

        self.FSPS_args['zmet'] = self.zmet_gen(
            override=self.FSPS_args['zmet'])

        self.FSPS_args['sigma'] = self.sigma_gen(
            override=self.FSPS_args['sigma'])

    def calc_sfh(self):
        '''
        return an array of times (rel to BB) and SFRs (Msol/yr)
        '''

        FSPS_args = self.FSPS_args
        mtot_cont = self.mtot_cont(
            FSPS_args['time_form'], self.time0, FSPS_args['eftu'],
            FSPS_args['time_cut'], FSPS_args['eftc'])

        ts = np.linspace(0., self.time0, 1000)

        # calculate the total mass formed in bursts
        mtot_burst = FSPS_args['A'].sum() * mtot_cont
        mtot = mtot_burst + mtot_cont

        all_sf_v = np.vectorize(
            self.all_sf,
            excluded=set(FSPS_args.keys() + ['time0',]),
            cache=False)

        mtot_integration = integrate.quad(
            self.all_sf, 0., self.time0, args=(
                FSPS_args['time_form'], FSPS_args['eftu'],
                FSPS_args['time_cut'], FSPS_args['eftc'],
                FSPS_args['time_burst'], FSPS_args['dt_burst'],
                FSPS_args['A'], self.time0, mtot_cont))[0]

        print '\n'
        print '\n'.join(
            ['{}: {}'.format(*l) for l in zip(
                ['mtot_cont', 'mtot_burst', 'mtot', 'mtot_integration'],
                [mtot_cont, mtot_burst, mtot, mtot_integration])])

        #sfrs = np.array(
        #    [self.all_sf(t, time0=self.time0, **FSPS_args)
        #     for t in ts])

        sfrs = all_sf_v(ts, time0=self.time0, **FSPS_args)
        print 'array-summed SFR:', sfrs.sum() * (ts[1] - ts[0])

        '''plt.figure(figsize=(3, 2), dpi=300)
                                plt.plot(ts, sfrs)
                                plt.xlabel('time [Gyr]')
                                plt.ylabel('SFR [sol mass/Gyr]')
                                plt.yscale('log')
                                plt.show()'''


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

    #=====
    # utility methods
    #=====

    def __repr__(self):
        return '\n'.join(
            ['{}: {}'.format(k, v) for k, v in self.FSPS_args.iteritems()])
