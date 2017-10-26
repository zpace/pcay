import numpy as np
import matplotlib.pyplot as plt

import os, sys, glob

# add manga RC location to path, and import config
if os.environ['MANGA_CONFIG_LOC'] not in sys.path:
    sys.path.append(os.environ['MANGA_CONFIG_LOC'])

from astropy.cosmology import WMAP9, z_at_value
from astropy import units as u, constants as c, table as t
from astropy.io import fits

from figures_tools import savefig

import csp

def make_burst_dutycycle_fig(CSP_dir):
    '''
    tabulate the average burst duty cycle, compared to burst strength
    '''

    csps_fnames = glob.glob(os.path.join(CSP_dir, 'CSPs_*.fits'))
    nsubper = fits.getval(csps_fnames[0], ext=0, keyword='nsubper')
    sfhs = t.vstack(list(map(t.Table.read, csps_fnames)))[::nsubper]

    agez0 = csp.WMAP9.age(0.).to('Gyr').value
    agez1 = csp.WMAP9.age(1.).to('Gyr').value
    ts = np.linspace(agez1, agez0, 1000)

    burst_strengths = 1. + np.row_stack(
        [np.array(
             [csp.burst_modifier(t=t_, nburst=row['nburst'], tb=row['tb'],
                                 dtb=row['dtb'], A=row['A'])
              for t_ in ts])
         for row in sfhs]).flatten()
    log_burst_strengths = np.log10(burst_strengths)

    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(log_burst_strengths, bins=20, histtype='step', normed=True,
            label='Duty Cycle', color='k')

    ax_ = ax.twinx()
    ax_.hist(log_burst_strengths, bins=20, histtype='step', normed=True,
            label='CDF', cumulative=True, color='C0', linestyle='--')

    enhcs = np.array([0.3, 0.6, np.log10(5.)])
    enhcs_frq = np.array([.32, .05, .01])
    ax_.errorbar(enhcs, 1. - enhcs_frq, xlolims=True, lolims=True, linestyle='None',
                marker='o', markersize=0.5, xerr=.05, yerr=.03 * (1. - enhcs_frq),
                capsize=1., elinewidth=0.5, capthick=0.5, color='C0',
                label='Noeske+\'07')

    ax_.set_ylabel(r'CDF', size='x-small', color='C0')
    ax_.set_yscale('log')
    ax_.tick_params(labelsize='x-small', color='C0', labelcolor='C0', which='both')

    ax.set_xlabel(r'SFR Enhancement [dex]', size='x-small')
    ax.set_ylabel(r'PDF', size='x-small')
    ax.tick_params(labelsize='x-small')
    ax.set_yscale('log')
    ax.set_xlim([.1, 1.])

    fig.tight_layout()

    fig.suptitle(r'CSP Burst Strengths', size='x-small')

    basename = 'burst_strengths'
    savefig(fig, ''.join((basename, '.png')), CSP_dir, close=True)

if __name__ == '__main__':
    CSP_dir = '/usr/data/minhas2/zpace/CSPs/CSPs_CKC14_MaNGA_20171025-2/'
    make_burst_dutycycle_fig(CSP_dir)
