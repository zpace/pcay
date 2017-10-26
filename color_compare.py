import numpy as np
import matplotlib.pyplot as plt
import mpl_scatter_density
import astropy.table as t

import os, sys, glob

# add manga RC location to path, and import config
if os.environ['MANGA_CONFIG_LOC'] not in sys.path:
    sys.path.append(os.environ['MANGA_CONFIG_LOC'])

import mangarc

if mangarc.tools_loc not in sys.path:
    sys.path.append(mangarc.tools_loc)

# personal
import manga_tools as m

from figures_tools import savefig

def make_color_compare(CSP_dir, mpl_v, n_obs=10, model_zcolors=False,
                       randomseed=None, snr_min=5.):
    '''
    make (g-r) vs (r-i) plot
    '''

    if randomseed is not None:
        np.random.seed(randomseed)
    drpall = m.load_drpall(mpl_v, index='plateifu')
    drpall = drpall[(drpall['mngtarg2'] == 0) * (drpall['nsa_z'] != -9999)]

    plateifus = np.random.choice(drpall['plateifu'], replace=False, size=n_obs)
    # cols are g, r, and i fluxes in order
    obs_fluxes = [m.get_gal_bpfluxes(*plateifu.split('-'), mpl_v,
                                     ['g', 'r', 'i'], snr_min)
                  for plateifu in plateifus]
    obs_ext = np.concatenate(
        [np.repeat(drpall.loc[plateifu]['nsa_extinction'][None, 3:6],
                   f.shape[0], axis=0)
         for f, plateifu in zip(obs_fluxes, plateifus)])
    obs_fluxes = np.row_stack(obs_fluxes)

    integrated_g = drpall['nsa_elpetro_absmag'][:, 3] - \
                       drpall['nsa_extinction'][:, 3]
    integrated_r = drpall['nsa_elpetro_absmag'][:, 4] - \
                       drpall['nsa_extinction'][:, 4]
    integrated_i = drpall['nsa_elpetro_absmag'][:, 5] - \
                       drpall['nsa_extinction'][:, 5]
    integrated_gr = integrated_g - integrated_r
    integrated_ri = integrated_r - integrated_i

    obs_gr = -2.5 * np.log10(obs_fluxes[:, 0] / obs_fluxes[:, 1]) - \
        (obs_ext[:, 0] - obs_ext[:, 1])
    obs_ri = -2.5 * np.log10(obs_fluxes[:, 1] / obs_fluxes[:, 2]) - \
        (obs_ext[:, 1] - obs_ext[:, 2])

    models_fnames = glob.glob(os.path.join(CSP_dir, 'CSPs_*.fits'))

    if model_zcolors:
        color1 = 'Cgr_z015'
        color2 = 'Cri_z015'
        color1TeX = r'$^{.15}C_{gr}$'
        color2TeX = r'$^{.15}C_{ri}$'
    else:
        color1 = 'Cgr'
        color2 = 'Cri'
        color1TeX = r'$C_{gr}$'
        color2TeX = r'$C_{ri}$'

    models_colors = t.vstack([t.Table.read(fn_)[color1, color2]
                              for fn_ in models_fnames])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    ax.scatter([-100], [-100], c='g', s=1, marker='s', label='models')
    ax.scatter([-100], [-100], c='b', s=1, marker='s', label='spax')
    ax.scatter([-100], [-100], c='r', s=1, marker='s', label='integrated')
    ax.scatter_density(obs_gr, obs_ri, color='b')
    ax.scatter_density(models_colors[color1], models_colors[color2], color='g')
    #ax.scatter_density(integrated_gr, integrated_ri, color='r')
    ax.set_xlim([-.2, 1.4])
    ax.set_ylim([-.1, 0.6])
    ax.set_xlabel(color1TeX, size='small')
    ax.set_ylabel(color2TeX, size='small')
    ax.legend(loc='best')
    fig.suptitle(CSP_dir.replace('_', '-'), size='x-small')
    fig.tight_layout()
    ax.tick_params(labelsize='x-small')

    basename = 'Cgr_Cri'
    if model_zcolors:
        basename = '_'.join((basename, 'z015'))

    savefig(fig, ''.join((basename, '.png')), CSP_dir, close=True)

if __name__ == '__main__':
    CSP_dir = '/usr/data/minhas2/zpace/CSPs/CSPs_CKC14_MaNGA_20171025-2/'
    make_color_compare(CSP_dir, 'MPL-5', n_obs=100, model_zcolors=False ,
                       randomseed=123, snr_min=10.)
