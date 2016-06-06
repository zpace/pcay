'''
tools for working with FSPS stellar population synthesis library
    (and its python bindings)
'''

import numpy as np

class FSPS_ParamBuilder(object):
    '''
    programmatically generate lots of SSP characteristics
    '''

    __version__ = '0.1'

    def __init__(self, t_form=None, g=None, A=None, t_burst=None,
                 dt_cut=None, tau_V=None, mu=None, sigma=None):

        self.p_cut = .3

        # for each of the parameters equal to None, reset the associated
        # instance attribute to the callable function that generates the
        # default distribution

        # pack into dict for easy stringification

        self.FSPS_args = {'t_form': None, 'g': None, 'A': None,
                          't_burstform': None, 't_burst': None,
                          'dt_cut': None, 'tau_V': None,
                          'mu': None, 'sigma': None}

        for k, v in self.FSPS_args.iteritems():
            # leave as-is if there's a single value given
            if v is not None:
                pass

    #=====
    # static methods
    #=====

    @staticmethod
    def age_form_gen(N, override=None):
        if override is not None:
            return override * np.ones(N)

        return np.random.uniform(low=1.5, high=13.5, size=N)

    @staticmethod
    def g_gen(N, override=None):
        if override is not None:
            return override * np.ones(N)

        return np.random(size=N)

    @staticmethod
    def age_burst_gen(N, age_form, pburst=0.5):

        # pburst is a tunable base burst probability, giving the probability
        # that a spectrum of maximum age will have a starburst sometime

        # since bursts are equally likely throughout time, normalize
        # so that bursts happen more often in spectra whose first stars
        # formed earlier

        # generate a possible starburst age, and randomly  decide whether a
        # maximal-age population would have it
        age_burst = FSPS_ParamBuilder.age_form_gen(N)
        has_burst = (np.random.rand(N) / ((age_form-1.5)/12.) < pburst)

        age_burst[~has_burst] = 14.

        return age_burst, has_burst

    @staticmethod
    def A_gen(N, override=None):
        if override is not None:
            return override * np.ones(N)


