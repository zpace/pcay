'''
tools for working with FSPS stellar population synthesis library
    (and its python bindings)
'''

import numpy as np
from scipy import stats

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

    def __init__(self, ict_b, **kwargs):

        '''
        set up star formation history generation to use with FSPS

        arguments:
            - ict_b: inverse characteristic time of starburst [1/Gyr]
                (tells how often starbursts occur)
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
            'time_form': None, 'g': None, 'A': np.array([]),
            'time_burst': np.array([]), 'dt_burst': np.array([]),
            'time_cut': None, 'eftc': None, 'tau_V': None, 'mu': None,
            'zmet': None}
        self.FSPS_args.update(kwargs)

        for k, v in self.FSPS_args.iteritems():
            # leave as-is if there's a single value given
            pass

    def calc_sfh(self):
        '''
        return an array of times (rel to BB) and SFRs (Msol/yr)
        '''
        pass

    #=====
    # static methods
    #=====

    @staticmethod
    def time_form_gen():
        return np.random.uniform(low=1.5, high=13.5)

    @staticmethod
    def eftu_gen():
        return 1./np.random.rand()

    @staticmethod
    def cut_gen(time_form, time0):
        # does cutoff occur?
        cut_yn = np.random.rand() < .3
        if not cut_yn:
            return (None, None)

        time_cut = np.random.uniform(time_form, time0)
        eftc = 10.**np.random.uniform(7., 9.)

        return time_cut, eftc

    @staticmethod
    def burst_gen(time_form, time0, time_cut):
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
        dt_burst = np.random.uniform(3.0e7, 3.0e8, nburst)
        # how much mass each forms
        A = 10.**np.random.uniform(np.log10(.03), np.log10(4.), nburst)

        return time_burst, dt_burst, A

def test_SFHBuilder():
    time0 = WMAP9.age(0.).value

    time_form = FSPS_SFHBuilder.time_form_gen()
    eftu = FSPS_SFHBuilder.eftu_gen()
    time_cut, eftc = FSPS_SFHBuilder.cut_gen(time_form, time0)
    time_burst, dt_burst, A = FSPS_SFHBuilder.burst_gen(
        time_form, time0, time_cut)

    print locals()

