import numpy as np
import matplotlib.pyplot as plt

import os, sys, glob

# add manga RC location to path, and import config
if os.environ['MANGA_CONFIG_LOC'] not in sys.path:
    sys.path.append(os.environ['MANGA_CONFIG_LOC'])

from astropy.cosmology import WMAP9, z_at_value
from astropy import units as u, constants as c
from astropy.io import fits

from figures_tools import savefig

def masked_z_at_value(fn, val, *args, **kwargs):
    try:
        z = z_at_value(fn, val, *args, **kwargs)
        return z, False
    except:
        return 0., True

def make_cosmic_sfr(CSP_dir):
    SFHs_fnames = glob.glob(os.path.join(CSP_dir, 'SFHs_*.fits'))
    nsubpersfh = fits.getval(SFHs_fnames[0], ext=0, keyword='NSUBPER')
    SFHs = np.row_stack(
        [fits.getdata(fn_, 'allsfhs') / fits.getdata(fn_, 'mformed')[::nsubpersfh, None]
         for fn_ in SFHs_fnames])
    ts = fits.getdata(SFHs_fnames[0], 'allts')
    zs, zmasks = zip(*[masked_z_at_value(WMAP9.age, t_ * u.Gyr) for t_ in ts])
    zs = np.ma.array(zs, mask=zmasks)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    SFRDmean = SFHs.mean(axis=0) / WMAP9.scale_factor(zs)**3.

    ax.plot(1. / WMAP9.scale_factor(zs), SFRDmean / SFRDmean.max())
    ax.set_xlim([1., 1. / WMAP9.scale_factor(10.)])
    ax.set_yscale('log')
    ax.set_ylim([.009, 1.05])
    ax.set_xlabel(r'$\frac{1}{a}$', size='x-small')
    ax.set_ylabel(r'$\log{\psi}$', size='x-small')
    ax.tick_params(labelsize='x-small', which='both')

    ax_ = ax.twiny()
    ax_.set_xlim(ax.get_xlim())
    zticks = np.linspace(0., 10., 11)
    inv_sf_ticks = 1. / WMAP9.scale_factor(zticks)
    ax_.set_xticks(inv_sf_ticks, minor=False)
    ax_.set_xticklabels(zticks)
    ax_.tick_params(labelsize='x-small')
    ax_.set_xlabel(r'$z$', size='x-small')

    fig.suptitle('``Cosmic" SFR', size='small')

    savefig(fig, 'CosmicSFR.png', CSP_dir, close=True)

if __name__ == '__main__':
    CSP_dir = '/usr/data/minhas2/zpace/CSPs/CSPs_CKC14_MaNGA_20171114-1/'
    make_cosmic_sfr(CSP_dir)
