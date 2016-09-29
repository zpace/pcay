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

from spectrophot import lumdens2bbdlum

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

    __version__ = '0.1'

    def __init__(self, max_bursts=5, override={}):
        '''
        set up star formation history generation to use with FSPS

        arguments:

        **override:
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
        self.max_bursts = max_bursts

        self.cosmo = WMAP9
        # how long the universe has been around
        self.time0 = self.cosmo.age(0.).to('Gyr').value

        # call parameter generation method with override values
        self.override = override
        self.FSPS_args = self.gen_FSPS_args()

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
        params = self.override.copy()

        # underlying
        params.update(self.time_form_gen())
        params.update(self.eftu_gen())

        # incidentals
        params.update(self.sigma_gen())
        params.update(self.tau_V_gen())
        params.update(self.mu_gen())
        params.update(self.zmet_gen())

        # cuts
        params.update(self.cut_gen(**params))

        # bursts
        params.update(self.burst_gen(**params))

        return params

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
        sp.params['imf_type'] = 1
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

        plt.close('all')

        plt.figure(figsize=(3, 2), dpi=300)
        ax = plt.subplot(111)
        ax.plot(self.ts[self.ts < self.time0], self.sfrs[self.ts < self.time0],
                c='b', linewidth=0.5)
        ax.set_xlabel('time [Gyr]', size=8)
        ax.set_ylabel('SFR [sol mass/yr]', size=8)
        ax.set_yscale('log')
        ax.tick_params(labelsize=8)

        # compute y-axis limits
        # cover a dynamic range of a few OOM, plus bursts
        ylim_cont = [5.0e-3, 1.25]
        ylim = ylim_cont
        if self.FSPS_args['nburst'] > 0:
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

        return tab

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
        burst_starts = self.FSPS_args['time_burst'][:nburst]
        burst_ends = (self.FSPS_args['time_burst'] +
                      self.FSPS_args['dt_burst'])[:nburst]

        discont = self.disconts
        # ages starting at 1Myr, and going to start of SF
        ages = 10.**np.linspace(-4., np.log10(self.time0), 300)
        ts = self.time0 - ages
        ts = np.append(ts, discont)
        ts.sort()
        if ts[0] < 0:
            ts[0] = 0.
        return ts

    @property
    def sfrs(self):
        return self.all_sf_v(self.ts)

    @property
    def disconts(self):
        burst_ends = self.FSPS_args['time_burst'] + self.FSPS_args['dt_burst']
        points = np.append(self.FSPS_args['time_burst'], burst_ends)
        points = points[(0. < points) * (points < self.time0)]
        return points

    @property
    def mformed_integration(self):
        FSPS_args = self.FSPS_args
        mf, mfe = integrate.quad(
            self.all_sf, 0., self.time0,
            points=self.disconts, epsrel=5.0e-3)
        return mf * 1.0e9

    @property
    def mass_weighted_age(self):

        disconts = self.disconts
        mtot_cont = self.mtot_cont()

        def num_integrand(tau, mtot_cont):
            return tau * self.all_sf(self.time0 - tau)

        def denom_integrand(tau, mtot_cont):
            return self.all_sf(self.time0 - tau)

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
        disconts = self.disconts
        disconts = disconts[disconts > self.time0 - 1.]
        mf, mfe = integrate.quad(
            self.all_sf, self.time0 - 1., self.time0,
            points=disconts, epsrel=5.0e-3)
        F = mf / self.mformed_integration
        return F

    # =====
    # things to generate parameters
    # (allow master overrides)
    # =====

    def time_form_gen(self):
        if 'time_form' in self.override.keys():
            return {'time_form': self.override['time_form']}

        return {'time_form': np.random.uniform(low=1.5, high=13.5)}

    def eftu_gen(self):
        if 'eftu' in self.override.keys():
            return {'eftu': self.override['eftu']}

        return {'eftu': 1. / np.random.rand()}

    def time_cut_gen(self, time_form):
        if 'time_cut' in self.override.keys():
            return {'time_cut': self.override['time_cut']}

        return {'time_cut': np.random.uniform(time_form, self.time0)}

    def eftc_gen(self):
        if 'eftc' in self.override.keys():
            return {'eftc': self.override['eftc']}

        return {'eftc': 10.**np.random.uniform(-2, 0)}

    def cut_gen(self, time_form, **params):
        default = {'cut': False, 'time_cut': 0., 'eftc': 0.}
        cut_yn = np.random.rand() < 0.3

        # if either/both time_cut/eftc are provided, use them!
        if (('time_cut' in self.override.keys()) or
            ('eftc' in self.override.keys()) or
            ('cut' in self.override.keys())):

            return {'cut': self.override['cut'],
                    **self.time_cut_gen(time_form),
                    **self.eftc_gen()}

        # if nothing is provided, and there's no random cut, use default
        elif not cut_yn:
            return default
        # if nothing is provided, and there is a random cut, generate it!
        else:
            return {'cut': True, **self.time_cut_gen(time_form),
                    **self.eftc_gen()}

    def time_burst_gen(self, time_form, dt, nburst):
        t_ = np.random.uniform(time_form, time_form + dt, nburst)
        npad = self.max_bursts - nburst
        t_ = np.pad(t_, (0, npad), mode='constant', constant_values=0.)
        return {'time_burst': t_}

    def dt_burst_gen(self, nburst):
        dt_ = np.random.uniform(.01, .1, nburst)
        npad = self.max_bursts - nburst
        dt_ = np.pad(dt_, (0, npad), mode='constant', constant_values=0.)
        return {'dt_burst': dt_}

    def A_burst_gen(self, nburst):
        A_ = 10.**np.random.uniform(np.log10(1.), np.log10(20.), nburst)
        npad = self.max_bursts - nburst
        A_ = np.pad(A_, (0, npad), mode='constant', constant_values=0.)
        return {'A': A_}

    def burst_gen(self, cut, time_cut, time_form, **kwargs):
        # statistically, one burst occurs in duration time0
        # but forbidden to happen after cutoff
        if cut:
            dt = time_cut - time_form
        else:
            dt = self.time0 - time_form

        # =====

        # handle case that a burst is forbidden, i.e., override is 3 Nones
        if self.override.get('suppress_bursts', False):
            return {'A': np.zeros(self.max_bursts),
                    'dt_burst': np.zeros(self.max_bursts),
                    'time_burst': np.zeros(self.max_bursts),
                    'nburst': 0}

        # =====

        # handle case that single burst is specified or
        # partially specified, and move forward
        elif (sum(map(lambda k: k in self.override.keys(),
                      ['time_burst', 'dt_burst', 'A'])) > 0):

            nburst = self.override.get('nburst', 1)

            time_burst = self.override.get('time_burst', self.time_burst_gen(
                time_form, dt, nburst))
            dt_burst = self.override.get('dt_burst', self.dt_burst_gen(nburst))
            A = self.override.get('A', self.A_burst_gen(nburst))

            return {'time_burst': time_burst, 'dt_burst': dt_burst,
                    'A': A, 'nburst': nburst}

        # =====

        # handle the case that no burst information is given (3 NaNs)
        else:
            # number of bursts
            nburst = stats.poisson.rvs(dt / self.time0)
            if nburst > self.max_bursts:
                nburst = self.max_bursts

            return {'nburst': nburst,
                    **self.time_burst_gen(time_form, dt, nburst),
                    **self.dt_burst_gen(nburst), **self.A_burst_gen(nburst)}

    def zmet_gen(self, zsol=zsol_padova):
        if 'zmet' in self.override.keys():
            return {'zmet': self.override['zmet']}

        if np.random.rand() < .95:
            return {'zmet': np.random.uniform(0.2 * zsol, 2.5 * zsol)}

        return {'zmet': np.log10(np.random.uniform(.02 * zsol, .2 * zsol))}

    def tau_V_gen(self):
        if 'tau_V' in self.override.keys():
            return {'tau_V': self.override['tau_V']}

        mu_tau_V = 1.2
        std_tau_V = 1.272  # set to ensure 68% of prob mass lies < 2
        lclip_tau_V, uclip_tau_V = 0., 6.
        a_tau_V = (lclip_tau_V - mu_tau_V) / std_tau_V
        b_tau_V = (uclip_tau_V - mu_tau_V) / std_tau_V

        pdf_tau_V = stats.truncnorm(
            a=a_tau_V, b=b_tau_V, loc=mu_tau_V, scale=std_tau_V)

        tau_V = pdf_tau_V.rvs()

        return {'tau_V': tau_V}

    def mu_gen(self):
        if 'mu' in self.override.keys():
            return {'mu': self.override['mu']}

        mu_mu = 0.3
        std_mu = np.random.uniform(.1, 1)
        # 68th percentile range means that stdev is in range .1 - 1
        lclip_mu, uclip_mu = 0., 1.
        a_mu = (lclip_mu - mu_mu) / std_mu
        b_mu = (uclip_mu - mu_mu) / std_mu

        pdf_mu = stats.truncnorm(
            a=a_mu, b=b_mu, loc=mu_mu, scale=std_mu)

        mu = pdf_mu.rvs()

        return {'mu': mu}

    def sigma_gen(self):
        if 'sigma' in self.override.keys():
            return {'sigma': self.override['sigma']}

        return {'sigma': np.random.uniform(50., 400.)}

    def continuous_sf(self, t):
        '''
        eval the continuous portion of the SFR at some time

        Note: units are Msun/
        '''
        time_form = self.FSPS_args['time_form']
        eftu = self.FSPS_args['eftu']
        if t < time_form:
            return 0.

        return self.cut_modifier(t) * np.exp(-(t - time_form) / eftu)

    def cut_modifier(self, t):
        '''
        eval the SF cut's contribution to the overall SFR at some time
        '''

        cut = self.FSPS_args['cut']
        time_cut = self.FSPS_args['time_cut']
        eftc = self.FSPS_args['eftc']

        if not cut:
            return 1.
        elif t < time_cut:
            return 1.
        else:
            return np.exp(-(t - time_cut) / eftc)

    def burst_modifier(self, t):
        '''
        evaluate the SFR augmentation at some time
        '''
        nburst = self.FSPS_args['nburst']
        burst_starts = self.FSPS_args['time_burst'][:nburst]
        burst_ends = (self.FSPS_args['time_burst'] +
                      self.FSPS_args['dt_burst'])[:nburst]
        A = self.FSPS_args['A'][:nburst]

        in_burst = ((t >= burst_starts) * (t <= burst_ends))[:nburst]
        return A.dot(in_burst)

    def mtot_cont(self):
        # calculate total stellar mass formed in the continuous bit
        mtot_cont = integrate.quad(
            self.continuous_sf, 0, self.time0)
        return mtot_cont[0]

    def all_sf(self, t):
        '''
        eval the full SF at some time
        '''

        continuous = self.continuous_sf(t)
        burst = self.burst_modifier(t)

        return continuous * (1. + burst)

    # =====
    # utility methods
    # =====

    def __repr__(self):
        return '\n'.join(
            ['{}: {}'.format(k, v) for k, v in self.FSPS_args.items()])


def make_csp(params={}, return_l=False):
    print(os.getpid())
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


def make_spectral_library(n=1, pkl=True, lllim=3700., lulim=8900., dlogl=1.0e-4,
                          multiproc=False, nproc=8):

    if not pkl:
        if n is None:
            n = 1
        # generate CSPs and cache them
        CSPs = [FSPS_SFHBuilder(max_bursts=5).FSPS_args
                for _ in range(n)]
        with open('csps.pkl', 'wb') as f:
            pickle.dump(CSPs, f)
    else:
        with open('csps.pkl', 'rb') as f:
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
    Lr = lumdens2bbdlum(lam=l_full * u.AA,
                        Llam=specs * u.Unit('Lsun/AA'), band='r')
    Li = lumdens2bbdlum(lam=l_full * u.AA,
                        Llam=specs * u.Unit('Lsun/AA'), band='i')
    Lz = lumdens2bbdlum(lam=l_full * u.AA,
                        Llam=specs * u.Unit('Lsun/AA'), band='z')

    specs_interp = interp1d(x=l_full, y=specs, kind='linear', axis=-1)
    specs_reduced = specs_interp(l_final)

    MLr, MLi, MLz = (metadata['mstar'] / Lr.value,
                     metadata['mstar'] / Li.value,
                     metadata['mstar'] / Li.value)
    ML = t.Table(data=[MLr, MLi, MLz], names=['MLr', 'MLi', 'MLz'])
    metadata = t.hstack([metadata, ML])

    # initialize FITS HDUList
    hdulist = fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU(np.array(metadata)),
         fits.ImageHDU(l_final), fits.ImageHDU(specs_reduced)])
    hdulist[1].header['EXTNAME'] = 'meta'
    hdulist[2].header['EXTNAME'] = 'loglam'
    hdulist[3].header['EXTNAME'] = 'flam'
    '''
    extension list:
     - [1], 'meta': FITS table equal to `metadata`
     - [2], 'loglam': array with log-lambda, and dlogl header keyword
     - [3], 'flam': array with lum-densities for each CSP on same
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
