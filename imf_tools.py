'''
Define several stellar initial mass functions,
    with some tools for working with them
'''

import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate

class IMF(object):
    '''
    stellar initial mass function
    '''

    __version__ = '0.1'

    def __init__(self, imftype='salpeter', ml=0.1, mh=150., mf=1., dm=.005):
        '''
        set up an IMF with some probability distribution, lower mass limit,
            and upper mass limit, that formed some mass

        all masses & luminosities are implicitly in solar units, and times
            are in Gyr

        I've provided several choices of IMF, all of which are normalized to
            produce one 1-solar-mass star
        '''

        assert imftype in ['salpeter', 'kroupa', 'chabrier', 'millerscalo'], \
            'invalid imftype'

        self.imftype = imftype
        self.ml = ml # low mass limit
        self.mh = mh # high mass limit
        self.dm = dm # standard mass differential for computations

        self.pdf = getattr(self, imftype)

    def mdf(self, m):
        '''
        mass distribution function
        '''
        return m * self.pdf(m, self.ml, self.mh)

    def plot_imf(self):
        m_a = np.arange(self.ml, self.mh, self.dm)
        p = np.array([self.pdf(m, self.ml, self.mh) for m in m_a])

        plt.figure(figsize=(5, 4), dpi=300)
        ax = plt.subplot(111)
        ax.plot(m_a, p, label=self.imftype)
        ax.set_xlabel(r'$M_*$', size=8)
        ax.set_ylabel(r'$\frac{\textrm{P}(M_*)}{\textrm{P}(M_{\odot})}$',
                      size=8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(loc='best', prop={'size':6})
        ax.set_title('Stellar IMF', size=8)
        plt.tight_layout()
        plt.show()

    def massfrac_remaining_at_age(self, t):
        '''
        Compute the fraction of the cluster initial mass present at time t
        '''
        # mass distribution function
        zams_mass = integrate.quad(self.mdf, self.ml, self.mh)[0]
        m_max = self.max_ms_mass_at_age(t)
        tms_mass = integrate.quad(self.mdf, self.ml, m_max)[0]
        return tms_mass / zams_mass

    def plot_mdf(self):
        m_a = np.arange(self.ml, self.mh, self.dm)
        m0 = np.array([self.mdf(m) for m in m_a])

        plt.figure(figsize=(5, 4), dpi=300)
        ax = plt.subplot(111)
        ax.plot(m_a, m0, label=self.imftype)
        ax.set_xlabel(r'$M_*$', size=8)
        ax.set_ylabel(r'$\frac{\textrm{M}(M_*)}{\textrm{M}(M_{\odot})}$',
                      size=8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(loc='best', prop={'size':6})
        ax.set_title('Stellar MDF', size=8)
        plt.tight_layout()
        plt.show()

    def plot_massfrac_remaining_at_age(self):
        t_a = np.logspace(-3., 1.5, 100)
        mf_a = np.array([self.massfrac_remaining_at_age(t) for t in t_a])

        plt.figure(figsize=(5, 4), dpi=300)
        ax = plt.subplot(111)
        ax.plot(t_a, mf_a, label=self.imftype)
        ax.set_xlabel(r'$t[\textrm{Gyr}]$', size=8)
        ax.set_ylabel(r'$\frac{\textrm{M}_{rem}}{\textrm{M}_0}$', size=8)
        ax.set_xscale('log')
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(loc='best', prop={'size':6})
        ax.set_title('Stellar MDF', size=8)
        plt.tight_layout()
        plt.show()

    # =====
    # @staticmethods
    # =====

    @staticmethod
    def salpeter(m, ml, mh):
        '''straight up power law'''
        if ml <= m <= mh:
            return m**-2.35
        else:
            return 0.

    @staticmethod
    def millerscalo_upperbranch(m):
        return m**-2.35

    @staticmethod
    def millerscalo_lowerbranch(m):
        return 1.

    @staticmethod
    def millerscalo(m, ml, mh):
        '''
        power law between 1 and mh, constant between ml and 1
        '''
        if m < ml:
            return 0.
        if ml <= m < 1.:
            return 1.
        elif 1. <= m < mh:
            return IMF.millerscalo_upperbranch(m)
        else:
            return 0.

    @staticmethod
    def kroupa_upperbranch(m):
        return m**-2.3

    @staticmethod
    def kroupa_middlebranch(m):
        # enforce continuity
        ub = IMF.kroupa_upperbranch(.5) # value of upper branch
        lb = .5**-1.3
        return (ub/lb) * m**-1.3

    @staticmethod
    def kroupa_lowerbranch(m):
        ub = IMF.kroupa_middlebranch(.08)
        lb = .08**-.3
        return (ub/lb) * m**-.3

    @staticmethod
    def kroupa(m, ml, mh):
        m1, m2 = .08, .5 # power-law transition masses
        if m < ml:
            return 0.
        elif ml <= m < m1:
            return IMF.kroupa_lowerbranch(m)
        elif m1 <= m < m2:
            return IMF.kroupa_middlebranch(m)
        elif m2 <= m < mh:
            return IMF.kroupa_upperbranch(m)
        else:
            return 0.

    @staticmethod
    def chabrier_upperbranch(m):
        return m**-2.3

    @staticmethod
    def chabrier_lowerbranch(m):
        # enforce continuity
        ub = IMF.chabrier_upperbranch(1.)
        c = lambda x: 10.**(
            -(np.log10(x) - np.log10(0.08))**2./(2*.69**2.))
        lb = c(1.)
        return (ub/lb) * c(m)

    @staticmethod
    def chabrier(m, ml, mh):
        if m < ml:
            return 0.
        elif ml <= m < 1.:
            return IMF.chabrier_lowerbranch(m)
        elif 1. <= m < mh:
            return IMF.chabrier_upperbranch(m)
        else:
            return 0.

    @staticmethod
    def ms_lum_at_m(m):
        return m**3.5

    @staticmethod
    def max_ms_mass_at_age(t):
        return (t/10.)**(-1/2.9)




