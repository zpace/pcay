import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

from astropy import units as u, table as t  # , constants as c
from astropy.io import fits

from scipy.signal import medfilt
from scipy.linalg import pinv2
import linalg
import sklearn.covariance

import os, sys, io
from itertools import takewhile, combinations as comb

from glob import glob

from importer import *

import manga_tools as m

import utils as ut
from partition import CovWindows

from functools import lru_cache

eps = np.finfo(float).eps

# =====

print('MaNGA data-product info:', mpl_v, '({})'.format(m.DRP_MPL_versions[mpl_v]))
print('MaNGA data location:', os.environ['SAS_BASE_DIR'])


class Cov_Obs(object):
    '''
    a class to precompute observational spectral covariance matrices
    '''

    def __init__(self, cov, lllim, dlogl, nobj):
        self.cov = cov # enforce_posdef(cov)
        self.nspec = len(cov)
        self.lllim = lllim
        self.loglllim = np.log10(self.lllim)
        self.dlogl = dlogl
        self.nobj = nobj

        self.precision, self.cov_rank = pinv2(self.cov, return_rank=True, rcond=1.0e-3)

    # =====
    # classmethods
    # =====

    @classmethod
    def from_fits(cls, fname):
        hdulist = fits.open(fname)
        cov = hdulist[1].data
        h = hdulist[1].header
        lllim = 10.**h['LOGL0']
        dlogl = h['DLOGL']
        nobj = h['NOBJ']
        return cls(cov=cov, lllim=lllim, dlogl=dlogl, nobj=nobj)

    @classmethod
    def from_tremonti(cls, fname, *args, **kwargs):
        '''
        Christy's covariance calculations
        '''
        cov_super = fits.getdata(fname, ext=1)
        wave = cov_super['WAVE'][0]
        cov = cov_super['COV_MATRIX'][0]
        nobj = 0
        dlogl = ut.determine_dlogl(np.log10(wave))
        lllim = wave[0]
        return cls(cov=cov, lllim=lllim, dlogl=dlogl, nobj=nobj, *args, **kwargs)

    @classmethod
    def from_YMC_BOSS(cls, fname, logl0=3.5524001):
        hdulist = fits.open(fname)
        cov = hdulist[1].data
        h = hdulist[1].header
        lllim = 10.**logl0
        dlogl = 1.0e-4
        nobj = 48000
        return cls(cov=cov, lllim=lllim, dlogl=dlogl, nobj=nobj)

    # =====
    # methods
    # =====

    def _init_windows(self, w):
        self.windows = diag_windows(self.cov, w)

    @lru_cache(maxsize=256)
    def take(self, i0):
        return self.windows[i0]

    def precompute_Kpcs(self, E):
        '''
        precompute PC covs, based on given eigenvectors (projection matrix)
        '''

        ETE = E.T @ E
        inv_ETE = linalg.spla_chol_invert(
            ETE + np.diag(np.diag(ETE)), np.eye(*ETE.shape))
        H = inv_ETE @ E.T
        self.covwindows = CovWindows(self.cov, H.T)

    def write_fits(self, fname='cov.fits'):
        hdu_ = fits.PrimaryHDU()
        hdu = fits.ImageHDU(data=self.cov)
        hdu.header['LOGL0'] = np.log10(self.lllim)
        hdu.header['DLOGL'] = self.dlogl
        hdu.header['NOBJ'] = self.nobj
        hdulist = fits.HDUList([hdu_, hdu])
        hdulist.writeto(fname, overwrite=True)

    def make_im(self, kind, max_disp=0.4, llims=None):
        l = self.l
        fig, ax = plt.subplots(1, 1, figsize=(4, 5), dpi=400)
        ax.tick_params(axis='both', bottom=True, top=True, left=True, right=True,
                       labelbottom=True, labeltop=True, labelleft=True,
                       labelright=False, labelsize=6)

        if llims is not None:
            ax.set_xlim(llims)
            ax.set_ylim(llims)

        vmax = np.abs(self.cov).max()**0.3
        extend = 'neither'
        if vmax > max_disp:
            vmax = max_disp
            extend = 'both'

        im = ax.imshow(
            np.sign(self.cov) * (np.abs(self.cov))**0.3,
            extent=[l.min(), l.max(), l.min(), l.max()], cmap='coolwarm',
            vmax=vmax, vmin=-vmax, interpolation='nearest', aspect='equal')
        ax.set_xlabel(r'$\lambda  ~ [{\rm \AA}]$', size=6)
        ax.set_ylabel(r'$\lambda ~ [{\rm \AA}]$', size=6)
        cb = plt.colorbar(im, ax=ax, extend=extend, orientation='horizontal')
        cb.set_label(r'$\textrm{sign}(K) ~ |K|^{0.3}$', size=6)
        cb.ax.tick_params(labelsize='xx-small')
        fig.tight_layout()

        fig.savefig('cov_obs_{}.png'.format(kind), dpi=200)

    # =====
    # properties
    # =====

    @property
    def logl(self):
        return self.loglllim + np.linspace(
            0., self.dlogl * self.nspec, self.nspec)

    @property
    def l(self):
        return 10.**self.logl


class ShrunkenCov(Cov_Obs):
    '''
    shrunken covariance matrix
    '''
    def __init__(self, cov, lllim, dlogl, nobj, shrinkage=0.):
        shrunken_cov = sklearn.covariance.shrunk_covariance(
            emp_cov=cov, shrinkage=shrinkage)
        super().__init__(shrunken_cov, lllim, dlogl, nobj)

def enforce_posdef(a, replace_val=1.0e-6):
    '''
    enforce positive-definiteness: calculate the nearest
        (in frobenius-norm sense) positive-definite matrix to
        supplied (symmetric) matrix `a`
    '''
    # eigen-decompose `a`
    evals, evecs = np.linalg.eig(a)

    # set all eigenvalues <= 0 to floating-point epsilon
    evals[evals <= 0] = replace_val

    # recompose approximation of original matrix
    a_new = evecs @ np.diag(evals) @ np.linalg.inv(evecs)

    return a_new

def diag_windows(x, n):
    from numpy.lib.stride_tricks import as_strided
    if x.ndim != 2 or x.shape[0] != x.shape[1] or x.shape[0] < n:
        raise ValueError("Invalid input")
    w = as_strided(x, shape=(x.shape[0] - n + 1, n, n),
                   strides=(x.strides[0]+x.strides[1], x.strides[0], x.strides[1]))
    return w

def display_cov(cov, dv):
    plt.imshow(np.sign(cov) * np.abs(cov)**.3, cmap='coolwarm', vmin=-dv, vmax=dv)
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.show()