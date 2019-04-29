import os, sys, matplotlib

# add manga RC location to path, and import config
if os.environ['MANGA_CONFIG_LOC'] not in sys.path:
    sys.path.append(os.environ['MANGA_CONFIG_LOC'])

import mangarc

if mangarc.tools_loc not in sys.path:
    sys.path.append(mangarc.tools_loc)

mpl_v = 'MPL-8'
daptype = 'SPX-MILESHC-MILESHC'
csp_basedir = '/usr/data/minhas2/zpace/CSPs/CSPs_CKC14_MaNGA_20190215-1'
manga_results_basedir = '/usr/data/minhas2/zpace/CSPs/CSPs_CKC14_MaNGA_20190215-1'
mocks_results_basedir = '/usr/data/minhas2/zpace/CSPs/CSPs_CKC14_MaNGA_20190215-1/fakedata'

from astropy.cosmology import WMAP9
cosmo = WMAP9

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.usetex'] = True
