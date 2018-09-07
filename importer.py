import os, sys

# add manga RC location to path, and import config
if os.environ['MANGA_CONFIG_LOC'] not in sys.path:
    sys.path.append(os.environ['MANGA_CONFIG_LOC'])

import mangarc

if mangarc.tools_loc not in sys.path:
    sys.path.append(mangarc.tools_loc)

mpl_v = 'MPL-6'
basedir = '/usr/data/minhas2/zpace/CSPs/CSPs_CKC14_MaNGA_20180713-1/'
from astropy.cosmology import WMAP9
