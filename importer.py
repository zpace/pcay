import os, sys, matplotlib

mpl_v = 'MPL-8'
daptype = 'SPX-MILESHC-MILESHC'
csp_basedir = os.environ['STELLARMASS_PCA_CSPBASE']
manga_results_basedir = os.environ['STELLARMASS_PCA_RESULTSDIR']
mocks_results_basedir = os.path.join(
    os.environ['STELLARMASS_PCA_RESULTSDIR'], 'mocks')

from astropy.cosmology import WMAP9
cosmo = WMAP9

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.usetex'] = True
