#!/usr/bin/env python3

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

from spectrophot import lumspec2lsun, Spec2Phot
import indices
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

req_param_keys = ['tf', 'tt', 'd1', 'gamma',  #  delayed tau model, ramp
                  'A', 'tb', 'dtb', 'nburst', #  burst properties
                  'tau_V', 'mu', 'logzsol']  #  other

class FSPS_SFHBuilder(object):
    '''
    programmatically generate lots of SSP characteristics

    conventions:
        - all "time"s are in reference to BB, in Gyr
        - all "age"s are in reference to now, in Gyr
    '''

    __version__ = '0.2'

    def __init__(self, max_bursts=5, override={}, RS=None, seed=None,
                 subsample_keys=['tau_V', 'mu', 'sigma'], Nsubsample=20, NBB=2.,
                 pct_notrans=0., tform_key=None, trans_mode=1.0):
        '''
        set up star formation history generation to use with FSPS

        General Form:

        SFR = {
            0                                 : t < tf
            A (t - tf) exp((t - tf)/d1)       : tf < t < tt
            A (tt - tf) exp((tt - tf)/d1) -
                (t - tt) / gamma                 : t > tt
        } * Burst

        arguments:

        **override:
            - tf: time (post-BB) that star formation began
            - tt: time of transition to ramp-up or ramp-down
            - d1: e-folding time of declining tau component
            - gamma: SFR-t [Msun/yr/Gyr] slope of SFR ramp-up/-down
            - A [array]: total stellar mass produced in each starburst,
                relative to total produced by continuous model
            - tb [array]: time (post-BB) that each burst began
            - dtb [array]: duration of each burst
            - tau_V: V-band optical depth affecting young stars
            - mu: fraction of V-band optical depth affecting old stars
            - logzsol: metallicity index
        '''

        self.req_param_keys = req_param_keys

        self.FSPS_args = {k: None for k in self.req_param_keys}

        # manage parameter subsampling
        self.subsample_keys = subsample_keys
        self.Nsubsample = Nsubsample

        # in a full SFH of length time0, avg # of bursts (Poisson)
        self.NBB = NBB
        self.pct_notrans = pct_notrans
        self.tform_key = tform_key

        # mode of transition time ditribution, in terms of peak time and present
        self.trans_mode = trans_mode

        if not RS:
            self.RS = np.random.RandomState()
        else:
            self.RS = RS

        if not seed:
            pass
        else:
            self.RS.seed(seed)

        self.max_bursts = max_bursts

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

        good = False

        # occasionally there are issues, so veryify MWA
        while not good:
            # underlying
            self.time_form_gen()
            self.delay_tau_gen()

            # incidentals
            self.sigma_gen()
            self.tau_V_mu_gen()
            self.mu_gen()

            self.logzsol_gen()

            # bursts
            self.burst_gen()

            self.fbhb_gen()
            self.sbss_gen()

            goodMWA = (0 <= self.mass_weighted_age <= self.time0)
            nonzeromstar = (self.mstar > 0.)
            finitemstar = (np.isfinite(self.mstar))
            good = goodMWA * nonzeromstar * finitemstar
            if not good:
                print('REDO -- goodMWA {} ({}); nonzeromstar {} ({}); finitemstar {}'.format(
                      goodMWA, self.mass_weighted_age, nonzeromstar, self.mstar, finitemstar))

    def _init_sp_(self):
        self.sp = fsps.StellarPopulation(
            zcontinuous=1, add_stellar_remnants=True,
            smooth_velocity=True, redshift_colors=False,
            vactoair_flag=False, tage=self.time0, masscut=150.,
            add_neb_continuum=False)

        self.sp.params['imf_type'] = 2
        self.sp.params['tage'] = self.time0
        self.sp.params['sfh'] = 3
        self.sp.params['dust_type'] = 0 # Charlot & Fall 2000
        self.sp.params['dust_tesc'] = 7.
        self.sp.params['dust1'] = 0. # optical depth for young stars
        self.sp.params['dust2'] = 0. # optical depth for old stars
        self.sp.params['dust_index'] = -0.7 # dust power law index for old stars
        self.sp.params['dust1_index'] = -1.3 # dust power law index for young stars

    def _cleanup_sp_(self):
        self.override = {}

    def run_fsps(self):
        '''
        run FSPS with given CSP parameters, using continuous SFH

        returns:
         - spec: flux-density (Lsol/AA)
        '''

        self.sp.params['logzsol'] = self.FSPS_args['logzsol']
        sfrs = self.sfrs

        self.sp.set_tabular_sfh(age=self.ts, sfr=sfrs)

        spec, specinds, ion_ph_rate, uv_slope = zip(
            *[self._run_fsps_newparams(
                  tage=self.time0,
                  d={'dust1': tau * (1. - mu), 'dust2': tau * mu, 'sigma_smooth': sigma})
              for tau, mu, sigma in zip(
                  self.FSPS_args['tau_V'], self.FSPS_args['mu'],
                  self.FSPS_args['sigma'])])

        spec = np.row_stack(spec)
        specinds = t.vstack(specinds)

        return spec, specinds, ion_ph_rate, uv_slope

    def _run_fsps_newparams(self, tage, d):
        for k in d:
            self.sp.params[k] = d[k]

        lam, spec = self.sp.get_spectrum(tage=tage, peraa=True)

        # calculate spectral indices using velocity dispersion of zero
        self.sp.params['sigma_smooth'] = 0.
        lam, spec_zeroveldisp = self.sp.get_spectrum(tage=tage, peraa=True)

        sis = indices.data['ixname']
        sis_tab = t.Table(
            data=[t.Column([indices.StellarIndex(si)(lam, spec_zeroveldisp, axis=0)],
                               name=si) for si in sis])

        self.sp.params['dust1'] = 0.
        self.sp.params['dust2'] = 0.
        lam, spec_zeroatten = self.sp.get_spectrum(tage=tage, peraa=True)
        H_ion_ph_rate = spec_to_photon_rate(
            x=lam, xunit=u.AA, spec=spec_zeroatten, specunit=u.Lsun/u.AA,
            ph_e_thresh=(c.h * c.c * c.Ryd), out_unit=u.ph/u.s)

        uv_slope = calc_uv_slope(
            x=lam, xunit=u.AA, spec=spec_zeroatten, specunit=u.Lsun/u.AA,
            ratio_unit=u.Lsun/u.AA, xrg=[505., 912.])

        self.sfh_changeflag = False

        return spec, sis_tab, H_ion_ph_rate, uv_slope

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

    def dump_tuners(self, loc):
        dump_fname = os.path.join(loc, 'sfh_tuners.par')
        with open(dump_fname, 'w') as f:
            for k, v in self.sfh_tuners_dict.items():
                print(k, v, sep=': ', file=f)


    def to_table(self):

        ks = self.subsample_keys
        tab = t.Table(rows=[self.FSPS_args])

        # make way for properly separated columns!
        for k in ks:
            del tab[k]

        tab.add_column(t.Column(data=[self.frac_mform_dt(age=.02)],
                                name='F_20M'))
        tab.add_column(t.Column(data=[self.frac_mform_dt(age=.05)],
                                name='F_50M'))
        tab.add_column(t.Column(data=[self.frac_mform_dt(age=.1)],
                                name='F_100M'))
        tab.add_column(t.Column(data=[self.frac_mform_dt(age=.2)],
                                name='F_200M'))
        tab.add_column(t.Column(data=[self.frac_mform_dt(age=.5)],
                                name='F_500M'))
        tab.add_column(t.Column(data=[self.frac_mform_dt(age=1)],
                                name='F_1G'))
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
        earlyt = .2
        latet = 13.
        midt1 = .4
        midt2 = 4.

        # if param already set on object instantiation, leave it alone
        if 'tf' in self.override:
            self.FSPS_args['tf'] = self.override['tf']
        elif self.tform_key == 'trapz':
            width = self.time0 - earlyt
            tformtrap = stats.trapz(
                loc=earlyt, scale=width, c=(midt1 - earlyt) / width,
                d=(midt2 - earlyt) / width)
            tformtrap.random_state = self.RS
            self.FSPS_args['tf'] = tformtrap.rvs()
        elif self.tform_key == 'log':
            self.FSPS_args['tf'] = 10.**self.RS.uniform(
                low=np.log10(earlyt), high=np.log10(latet))
        elif self.tform_key == 'loglinmix':
            # 25-75 mix between log-uniform and lin-uniform
            if np.random.rand() < .25:
                self.FSPS_args['tf'] = 10.**self.RS.uniform(
                    low=np.log10(earlyt), high=np.log10(latet))
            else:
                self.FSPS_args['tf'] = self.RS.uniform(low=earlyt, high=latet)
        elif self.tform_key == 'norm':
            # normal distribution
            tf_min, tf_max = earlyt, latet
            tf_mean = 5.
            tf_std = 4.
            tf_a, tf_b = (tf_min - tf_mean) / tf_std, (tf_max - tf_mean) / tf_std
            tf_dist = stats.truncnorm(a=tf_a, b=tf_b, loc=tf_mean, scale=tf_std)
            self.FSPS_args['tf'] = tf_dist.rvs(None, random_state=self.RS)
        else:
            self.FSPS_args['tf'] = self.RS.uniform(low=earlyt, high=latet)

    def delay_tau_gen(self):
        '''
        generate delayed tau-model
        '''

        d1 = self._d1_gen()  #  eft of tau component
        tt = self._tt_gen(d1)  #  transition time
        theta, gamma = self._theta_gamma_gen(d1=d1, tt=tt)

        params = {'d1': d1, 'tt': tt, 'gamma': gamma, 'theta': theta}
        self.FSPS_args.update(params)

        self.sfh_changeflag = True

    def _d1_gen(self, logmean=.4, logstd=.4, a_lin=.1, b_lin=15.):
        if 'd1' in self.override:
            return self.override['d1']
        a_log, b_log = np.log10(a_lin), np.log10(b_lin)
        a_dist, b_dist = ((a_log - logmean) / logstd, (b_log - logmean) / logstd)
        dist = stats.truncnorm(a=a_dist, b=b_dist, loc=logmean, scale=logstd)
        return 10.**dist.rvs(None, random_state=self.RS)

    def _tt_gen(self, d1, d1delay=0.25, ddt=0.):
        # if param already set on object instantiation, leave it alone
        if 'tt' in self.override:
            return self.override['tt']

        tf = self.FSPS_args['tf']

        # transition comes after tf + some # of EFT delay + const
        dt = d1 * d1delay + ddt

        if tf + dt >= self.time0:
            return self.time0 + .01
        elif np.random.rand() > 1. - self.pct_notrans:
            return self.time0 + .01
        else:
            return self.RS.uniform(tf + dt, self.time0 + .01, None)

    def _theta_gamma_gen(self, d1, tt):
        '''
        generate random SFR-t plane slope (SFRmax / Gyr) for late times
        '''

        tf = self.FSPS_args['tf']
        tpeak = tf + d1
        SFRpeak = self.delay_tau_model(t=tpeak, d={'tf': tf, 'd1': d1})

        if 'gamma' in self.override:
            gamma = self.override['gamma']
            theta = np.arctan(gamma)
            return theta, gamma
        elif 'theta' in self.override:
            theta = self.override['theta']
            gamma = np.tan(theta)
            return theta, gamma

        if tpeak > tt:
            theta = 0.
        else:
            # star formation may cut off immediately
            theta_min = -np.pi / 2.
            # max allowed angle corresponds to half the average slope of initial ramp-up
            theta_max = np.arctan(SFRpeak / d1)
            # mode corresponds to current SFR
            dt = .01
            dSFR = (self.delay_tau_model(t=tt + dt, d={'tf': tf, 'd1': d1}) - \
                    self.delay_tau_model(t=tt, d={'tf': tf, 'd1': d1}))
            theta_cur = np.arctan(dSFR / dt)
            theta = self.RS.triangular(left=theta_min, mode=theta_cur, right=theta_max)

        gamma = np.tan(theta)

        return np.atleast_1d(theta), np.atleast_1d(gamma)

    @property
    def dt_avail(self):
        '''
        time duration available for bursts
        '''

        # if ramp-up, then anytime after SFH peak is ok
        if np.sign(self.FSPS_args['gamma']) > 0:
            dt = self.time0 - (self.FSPS_args['tf'] + self.FSPS_args['d1'])
            if dt < 0.:
                return 0.
            else:
                return dt

        # otherwise, bursts must occur before ramp-down
        return self.FSPS_args['tt'] - self.FSPS_args['tf']

    def _time_burst_gen(self, nburst, dt):
        tf = self.FSPS_args['tf']
        d1 = self.FSPS_args['d1']
        t_ = self.RS.uniform(tf + d1, tf + d1 + dt, nburst)
        npad = self.max_bursts - nburst
        t_ = np.pad(t_, (0, npad), mode='constant', constant_values=0.)
        return t_

    def _dt_burst_gen(self, nburst):
        dt_ = self.RS.uniform(.05, 1., nburst)
        npad = self.max_bursts - nburst
        dt_ = np.pad(dt_, (0, npad), mode='constant', constant_values=0.)
        return dt_

    def _A_burst_gen(self, nburst):
        A_ = 10.**self.RS.lognormal(-1., 1., nburst)
        npad = self.max_bursts - nburst
        A_ = np.pad(A_, (0, npad), mode='constant', constant_values=0.)
        return A_

    def recent_variability(self, n, N):
        '''
        turn on recent variability in SFR, acc. to Faucher-Giguere '17
        '''

        mlin, slin2 = 1., 25.
        mlog = 2. * np.log(mlin) - 0.5 * np.log(slin2 + mlin**2.)
        slog2 = -2. * np.log(mlin) + np.log(slin2 + mlin**2.)
        norm = stats.norm(loc=mlog, scale=np.sqrt(slog2))
        factor = norm.rvs(n, random_state=self.RS)

        return np.pad(factor, pad_width=(N - n, 0), mode='constant',
                      constant_values=1.)

    def burst_gen(self):
        # statistically, `self.NBB` bursts occur in duration time0
        # but forbidden to happen after cutoff

        dt = self.dt_avail

        time_form = self.FSPS_args['tf']

        if 'nburst' in self.override:
            nburst = self.override['nburst']
        else:
            nburst = self.RS.poisson(self.NBB * dt / self.time0, None)

        if nburst > self.max_bursts:
            nburst = self.max_bursts

        if 'tb' in self.override:
            tb = self.override['tb']
        else:
            tb = self._time_burst_gen(nburst, dt)

        if 'dtb' in self.override:
            dtb = self.override['dtb']
        else:
            dtb = self._dt_burst_gen(nburst)

        if 'A' in self.override:
            A = self.override['A']
        else:
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
        # 60% chance of linearly-uniform metallicity range
        elif self.RS.rand() < .6:
            self.FSPS_args.update(
                {'logzsol': np.log10(ut.lin_transform(
                    r1=[0., 1.], r2=[zsol.max(), zsol.min()], x=d_.rvs()))})
        # 10% chance of logarithmically-uniform
        else:
            self.FSPS_args.update(
                {'logzsol': self.RS.uniform(
                    np.log10(zsol.min()), np.log10(zsol.max()))})

    def fbhb_gen(self):
        '''
        fraction of horiz branch stars that are blue
        '''
        if 'fbhb' in self.override:
            self.FSPS_args.update({'fbhb': self.override['fbhb']})
        else:
            self.FSPS_args.update(
                {'fbhb': self.RS.beta(2., 7., None)})

    def sbss_gen(self):
        '''
        specific frequency of blue stragglers (rel to all HB)
        '''
        # Santucci says that fBSS / fBHB is ~4 in thick disk,
        # ~1.5-2 in inner halo, ~1 in outer halo

        # to get specific num of BSS, mult fBSS / fBHB by fBHB

        if 'sbss' in self.override:
            self.FSPS_args.update({'sbss': self.override['sbss']})
        else:
            self.FSPS_args.update(
                {'sbss': 10. * self.RS.beta(1., 4., None)})

    def tau_V_mu_gen(self, loc=.4, scale=.2, a=-2., b=4.):
        tau_V_mu_dist = stats.truncnorm(loc=loc, scale=scale, a=a, b=b)
        tau_V_mu_dist.random_state = self.RS

        tau_V_mu = tau_V_mu_dist.rvs(self.Nsubsample)

        self.mu_gen()
        self.tau_V_gen(tau_V_mu, self.FSPS_args['mu'])


    def tau_V_gen(self, tau_V_mu, mu):
        if 'tau_V' in self.override:
            tau_V = self.override['tau_V']
        else:
            tau_V = tau_V_mu / mu

        self.FSPS_args.update({'tau_V': tau_V})

    def mu_gen(self, loc=.3, scale=.2, a=-1., b=3.):
        if 'mu' in self.override:
            mu = self.override['mu']
        else:
            mu_dist = stats.truncnorm(loc=loc, scale=scale, a=a, b=b)

            mu = mu_dist.rvs(size=self.Nsubsample, random_state=self.RS)

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

    def frac_mform_dt(self, age=1.):
        '''
        compute mass fraction formed sooner than `age`
        '''

        ages = self.time0 - self.ts
        i_ = np.argsort(ages)
        ages, sfrs = ages[i_], self.sfrs[i_]

        mf_all = np.trapz(x=ages, y=sfrs)
        mf_rec = np.trapz(x=ages[ages < age], y=sfrs[ages < age])

        return mf_rec / mf_all

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
        ages = (np.arange(1, 10, .1) * 10.**np.arange(-3, 2, 1)[:, None]).flatten()
        ts = self.time0 - ages
        dts = self.FSPS_args['d1'] * np.linspace(0., 2., 5)
        ts = np.unique(np.concatenate(
            [np.array([0, self.FSPS_args['tf'], self.FSPS_args['tf'] - .001]),
             ts, self.FSPS_args['tf'] + dts, discont]))
        ts = ts[(ts >= 0) * (ts <= self.time0)]
        ts.sort()
        return ts

    @property
    def allts(self):
        return np.linspace(0., self.time0, 5000)

    @property
    def sfrs(self):
        sfrs = self.all_sf_v(self.ts)
        if sfrs.max() < 1.:
            sfrs /= sfrs.max()
        return sfrs

    @property
    def allsfrs(self):
        sfrs = self.all_sf_v(self.allts)
        if sfrs.max() < 1.:
            sfrs /= sfrs.max()
        return sfrs

    @property
    def disconts(self):
        burst_starts = self.FSPS_args['tb']
        dtb = self.FSPS_args['dtb']
        burst_ends = burst_starts + dtb
        dt = .001

        burst_starts_m = burst_starts - dt
        burst_ends_p = burst_ends + dt
        sf_starts = self.FSPS_args['tf'] + np.array([-dt, dt])

        tt = self.FSPS_args['tt']
        tt_pm = np.array([tt - dt, tt, tt + dt]).flatten()

        points = np.concatenate([sf_starts, burst_starts, burst_starts_m,
                                 burst_ends, burst_ends_p, tt_pm])
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
    def sfh_tuners_dict(self):
        d = {'trans_mode': self.trans_mode, 'NBB': self.NBB,
             'pct_notrans': self.pct_notrans, 'tform_key': self.tform_key,
             'max_bursts': self.max_bursts}
        return d

    @property
    def sfr_at_max(self):
        tf = self.FSPS_args['tf']
        d1 = self.FSPS_args['d1']
        sfr = self.delay_tau_model(tf + d1, self.FSPS_args)
        return sfr

    # =====
    # utility methods
    # =====

    def delay_tau_model(self, t, d):
        '''
        eval the continuous portion of the SFR at some time

        Note: units are Msun/yr, while the time units are Gyr
        '''
        tf = d['tf']

        d1 = d['d1']

        return (t - tf) * np.exp(-(t - tf) / d1)

    def ramp(self, t, d):
        '''
        additive ramp
        '''

        # transition time
        tt = d['tt']
        gamma = d['gamma']

        # evaluate SFR at transition time
        sfrb = self.delay_tau_model(t=tt, d=d)

        ramp = ((t - tt) * gamma)
        if type(t) is np.ndarray:
            ramp[t < tt] = 0.
        else:
            if t < tt:
                ramp = 0.

        return ramp

    def burst_modifier(self, t, d):
        '''
        evaluate the SFR augmentation at some time
        '''

        return burst_modifier(t, d['nburst'], d['tb'], d['dtb'], d['A'])

    def all_sf(self, t):
        '''
        eval the full SF at some time
        '''

        d = self.FSPS_args

        dtm = self.delay_tau_model(t, d)
        ramp = self.ramp(t, d)
        cont = dtm + ramp
        if cont < 0.:
            cont = 0.

        burst = self.burst_modifier(t, d)

        return cont * (1. + burst)

    def all_sf_a(self, t):

        d = self.FSPS_args

        dtm = self.delay_tau_model(t, d)
        ramp = self.ramp(t, d)
        cont = dtm + ramp
        cont[cont < 0.] = 0.

        burst = np.array([self.burst_modifier(t_, d) for t_ in t])

        return cont + burst * self.sfr_at_max

    # =====
    # utility methods
    # =====

    def __repr__(self):
        return '\n'.join(
            ['{}: {}'.format(k, v) for k, v in self.FSPS_args.items()])

def burst_modifier(t, nburst, tb, dtb, A):
        '''
        evaluate the SFR augmentation at some time

        burst contribution is evaluated as a fraction of the peak of the
            continuous model's SFR
        '''
        if nburst == 0:
            return np.ones_like(t)

        tb = tb[:nburst]
        dtb = dtb[:nburst]
        A = A[:nburst]

        burst_ends = tb + dtb
        in_burst = ((t >= tb) * (t <= burst_ends))

        return A.dot(in_burst)

def get_sfh_sfrs(sfh, ts, override={}, norm='mformed'):
    '''
    given some override, get the sfrs from each of a list of times
    '''

    sfh._cleanup_sp_()
    sfh.override = override
    sfh.gen_FSPS_args()

    sfrs = sfh.all_sf_v(ts)

    if norm == 'mformed':
        mtot = np.trapz(ts, sfrs)
    elif norm == 'mstar':
        mtot = sfh.mstar / 1.0e9

    return sfrs / mtot

def make_csp(sfh, override=None):
    sfh._cleanup_sp_()

    if override is not None:
        sfh.override = override
    sfh.gen_FSPS_args()

    spec, spectral_indices, ion_ph_rate, uv_slope = sfh.run_fsps()
    specdata = t.Table(
        [np.log10(ion_ph_rate / sfh.mstar).clip(min=0.), uv_slope],
        names=['logQHpersolmass', 'uv_slope'])
    tab = t.hstack([sfh.to_table(), spectral_indices, specdata])

    allsfrs = sfh.allsfrs

    return spec, tab, sfh.FSPS_args, allsfrs

def spec_to_photon_rate(x, xunit, spec, specunit, ph_e_thresh, out_unit):
    x = x * u.Unit(xunit)
    spec = spec * u.Unit(specunit)

    ph_spec = spec.to(u.photon / u.s / u.Hz, equivalencies=u.spectral_density(x))
    ph_energy = x.to(u.eV, equivalencies=u.spectral())
    nu = x.to(u.Hz, equivalencies=u.spectral())

    ph_rate = np.abs(np.trapz(x=nu, y=ph_spec * (ph_energy >= (c.Ryd * c.c * c.h))))

    return ph_rate.to(out_unit).value

def calc_uv_slope(x, xunit, spec, specunit, ratio_unit, xrg=[1250., 2600.]):
    '''
    UV slope in range 1250, 2600 AA
    (Calzetti et al, http://adsabs.harvard.edu/abs/1994ApJ...429..582C)
    '''
    x = x * u.Unit(xunit)
    spec = spec * u.Unit(specunit)

    new_x = x.to('AA', equivalencies=u.spectral())
    new_spec = spec.to(ratio_unit, equivalencies=u.spectral_density(x))
    new_x_range = (xrg[0] * u.Unit(xunit), xrg[1] * u.Unit(xunit))
    in_new_x_range = ~np.logical_or(new_x < new_x_range[0], new_x > new_x_range[1])

    log_new_x = np.log10(new_x.value)
    log_new_spec = np.log10(new_spec.value)

    poly = np.polyfit(
        x=log_new_x[in_new_x_range], y=log_new_spec[in_new_x_range], deg=1)
    plslope, plint = poly

    return plslope


def write_spec_fits(metadata, lam, specs, loc, fname, nsubper, nsfhper):
    '''
    make FITS file of training spectra
    '''

    # initialize FITS HDUList
    hdulist = fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU(np.array(metadata)),
         fits.ImageHDU(lam), fits.ImageHDU(specs)])
    hdulist[1].header['EXTNAME'] = 'meta'
    hdulist[2].header['EXTNAME'] = 'lam'
    hdulist[3].header['EXTNAME'] = 'flam'

    hdulist[0].header['NSUBPER'] = nsubper
    hdulist[0].header['NSFHPER'] = nsfhper

    '''
    extension list:
     - [1], 'meta': FITS table equal to `metadata`
     - [2], 'lam': array with lambda, and dlogl header keyword
     - [3], 'flam': array with lum-densities for each CSP on same
        wavelength grid
    '''

    hdulist.writeto(os.path.join(loc, '{}.fits'.format(fname)), overwrite=True)

def find_sfh_ixs(i, nsfhperfile, nsubpersfh):
    # `i` is the pure number SFH;
    # find file number `fnum` by dividing by mper
    nspecperfile = nsfhperfile * nsubpersfh

    # file index in filelist
    fnum = i // nspecperfile

    # record index within individual spectral-data file
    fi = i % nspecperfile

    # record number in the corresponding param table
    fii = fi // nsubpersfh

    # row number
    fiii = fi % nsubpersfh

    return fnum, fi, fii, fiii

def retrieve_SFHs(filelist, i, massnorm='mformed', nsubpersfh=None, nsfhperfile=None):
    '''
    fetch a time array & SFH from a series of FITS archives,
        each with `nsfhperfile` SFHs & `nsubpersfh` Z/tau/mu realizations for
        each SFH
    '''

    if nsfhperfile is None:
        nsfhperfile = fits.getval(filelist[0], ext=0, keyword='NSFHPER')
    if nsubpersfh is None:
        nsubpersfh = fits.getval(filelist[0], ext=0, keyword='NSUBPER')

    fnum, fi, fii, fiii = find_sfh_ixs(i, nsfhperfile, nsubpersfh)

    '''
    print('trainer {0}: spectral:file-spec {1}-{2}; SFH:file-rec-subsample {1}-{3}-{4}'.format(
          i, fnum, fi, fii, fiii))
    '''
    fname = filelist[fnum]

    allts = fits.getdata(fname, extname='allts')
    allsfhs = np.repeat(fits.getdata(fname, extname='allsfhs'),
                        nsubpersfh, axis=0)

    # normalize mass either by total mass formed or current stellar mass
    mtot = fits.getdata(fname, massnorm)[:, None]

    return allts, allsfhs / mtot, fi

def retrieve_meta_table(filelist, i, nsubpersfh=None, nsfhperfile=None):
    '''
    '''

    if nsfhperfile is None:
        nsfhperfile = fits.getval(filelist[0], ext=0, keyword='NSFHPER')
    if nsubpersfh is None:
        nsubpersfh = fits.getval(filelist[0], ext=0, keyword='NSUBPER')

    fnum, fi, fii, fiii = find_sfh_ixs(i, nsfhperfile, nsubpersfh)

    fname = filelist[fnum]

    metadata = t.Table.read(fname)

    return metadata, fii

def write_SFH_fits(allts, allsfrs, mstar, loc, fname, nsubper, nsfhper):
    '''
    make FITS file of training data's SFHs
    '''
    mformed = np.repeat(np.trapz(y=allsfrs, x=allts, axis=1), nsubper)

    # initialize FITS HDUList
    hdulist = fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(allts), fits.ImageHDU(allsfrs),
         fits.ImageHDU(mstar / 1.0e9), fits.ImageHDU(mformed)])
    hdulist[1].header['EXTNAME'] = 'allts'
    hdulist[2].header['EXTNAME'] = 'allsfhs'
    hdulist[3].header['EXTNAME'] = 'mstar'
    hdulist[4].header['EXTNAME'] = 'mformed'

    hdulist[0].header['NSUBPER'] = nsubper
    hdulist[0].header['NSFHPER'] = nsfhper

    '''
    extension list:
     - [1], 'allts': time steps (identical for all)
     - [2], 'allsfhs': SFRs corresponding to time steps
     - [3], 'mstar': current stellar mass
     - [4], 'mformed': total stellar mass formed
    '''

    hdulist.writeto(os.path.join(loc, '{}.fits'.format(fname)), overwrite=True)

def make_spectral_library(spec_fname, sfh_fname, sfh,
                          loc='CSPs', lllim=3500., lulim=10000.,
                          override=None, nsubper=1, nsfhper=1):

    specs, metadata, dicts, allsfrs = zip(
        *[make_csp(sfh, override=override) for _ in range(nsfhper)])

    # write out dict to pickle
    with open(os.path.join(loc, '{}.pkl'.format(spec_fname)), 'wb') as f:
        pickle.dump(dicts, f)

    # assemble the full table & spectra
    metadata = t.vstack(metadata)
    specs = np.row_stack(specs)
    allsfrs = np.row_stack(allsfrs)
    lam = sfh.sp.wavelengths

    in_ONIR = (lam > 3000.) * (lam < 11000.)

    # find luminosity
    Lg = lumspec2lsun(lam=lam * u.AA,
                      Llam=specs * u.Unit('Lsun/AA'), band='g')
    Lr = lumspec2lsun(lam=lam * u.AA,
                      Llam=specs * u.Unit('Lsun/AA'), band='r')
    Li = lumspec2lsun(lam=lam * u.AA,
                      Llam=specs * u.Unit('Lsun/AA'), band='i')
    Lz = lumspec2lsun(lam=lam * u.AA,
                      Llam=specs * u.Unit('Lsun/AA'), band='z')
    LV = lumspec2lsun(lam=lam * u.AA,
                      Llam=specs * u.Unit('Lsun/AA'), band='V')

    s2p = Spec2Phot(lam=lam * u.AA, flam=specs * u.Unit('Lsun AA-1 pc-2'),
                    family='sdss2010-*', axis=1, redshift=None)
    Cgr = s2p.color('sdss2010-g', 'sdss2010-r')
    Cri = s2p.color('sdss2010-r', 'sdss2010-i')
    Cgi = s2p.color('sdss2010-g', 'sdss2010-i')

    s2p_z015 = Spec2Phot(lam=lam * u.AA, flam=specs * u.Unit('Lsun AA-1 pc-2'),
                         family='sdss2010-*', axis=1, redshift=.15)
    Cgr_z015 = s2p_z015.color('sdss2010-g-shift(0.15)', 'sdss2010-r-shift(0.15)')
    Cri_z015 = s2p_z015.color('sdss2010-r-shift(0.15)', 'sdss2010-i-shift(0.15)')
    Cgi_z015 = s2p_z015.color('sdss2010-g-shift(0.15)', 'sdss2010-i-shift(0.15)')

    MLr, MLi, MLz, MLV = (metadata['mstar'] / Lr,
                          metadata['mstar'] / Li,
                          metadata['mstar'] / Lz,
                          metadata['mstar'] / LV)
    ML = t.Table(data=[MLr, MLi, MLz, MLV, Cgr, Cri, Cgi,
                       Cgr_z015, Cri_z015, Cgi_z015],
                 names=['MLr', 'MLi', 'MLz', 'MLV',
                        'Cgr', 'Cri', 'Cgi', 'Cgr_z015', 'Cri_z015', 'Cgi_z015'])
    metadata = t.hstack([metadata, ML])

    write_spec_fits(metadata=metadata, lam=lam, specs=specs, loc=loc,
                    fname=spec_fname, nsubper=nsubper, nsfhper=nsfhper)
    write_SFH_fits(allts=sfh.allts, allsfrs=allsfrs,
                   mstar=metadata['mstar'], loc=loc, fname=sfh_fname,
                   nsubper=nsubper, nsfhper=nsfhper)

    return sfh

def random_SFH_plots(n=10, save=False, sfh=None):
    from time import time

    fig = plt.figure(figsize=(3, 2), dpi=300)
    ax = fig.add_subplot(111)

    # make color cycle
    c_ix = range(n)

    if sfh is None:
        # instantiate SFH builder
        sfh = FSPS_SFHBuilder()

    for c in c_ix:
        sfh.gen_FSPS_args()
        ts = sfh.ts
        sfrs = sfh.sfrs

        tf = sfh.FSPS_args['tf']
        ts, sfrs = ts[ts > tf], sfrs[ts > tf]

        mf = np.trapz(x=ts, y=sfrs)

        ax.plot(ts, sfrs / mf, color='C{}'.format(c),
                linewidth=0.5)

    ax.set_xlabel('time [Gyr]', size=8)
    ax.set_ylabel('Normed SFR', size=8)

    ax.tick_params(labelsize=8)

    # compute y-axis limits
    # cover a dynamic range of a few OOM, plus bursts
    ax.set_xlim([0., 13.7])

    fig.tight_layout()
    if save:
        fig.savefig('randomSFHs.png')
    else:
        plt.show()

    return sfh

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

    nfiles, nper, Nsubsample = 40, 100, 10
    name_ix0 = 0
    name_ixf = name_ix0 + nfiles

    CSPs_dir = '/usr/data/minhas2/zpace/CSPs/CSPs_CKC14_MaNGA_20190215-1/'
    if not os.path.isdir(CSPs_dir):
        os.makedirs(CSPs_dir)

    RS = np.random.RandomState()
    sfh = FSPS_SFHBuilder(RS=RS, Nsubsample=Nsubsample, max_bursts=3, NBB=0.5,
                          pct_notrans=.75, tform_key='norm', trans_mode=.75)
    sfh.dump_tuners(loc=CSPs_dir)

    print('Making spectral library...')

    #'''
    for i in range(name_ix0, name_ixf):
        sfh = make_spectral_library(
            sfh=sfh, spec_fname='CSPs_{}'.format(i),
            sfh_fname='SFHs_{}'.format(i), loc=CSPs_dir,
            lllim=3500., lulim=10000.,
            nsfhper=nper, nsubper=Nsubsample)
        print('Done with {} of {}'.format(i + 1, name_ixf))
    #'''

    #'''
    print('Making validation data...')
    sfh = make_spectral_library(
        sfh=sfh, spec_fname='CSPs_validation',
        sfh_fname='SFHs_validation', loc=CSPs_dir,
        nsfhper=nper, nsubper=Nsubsample,
        lllim=3500., lulim=10000.)
    #'''

    #'''
    print('Making test data...')
    sfh = make_spectral_library(
        sfh=sfh, spec_fname='CSPs_test',
        sfh_fname='SFHs_test', loc=CSPs_dir,
        nsfhper=2 * nper, nsubper=Nsubsample,
        lllim=3500., lulim=10000.)
    #'''