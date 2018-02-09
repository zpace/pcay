import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
from matplotlib import ticker as mticker
import mpl_scatter_density

from astropy import table as t
from astropy.cosmology import WMAP9 as cosmo
from astropy.visualization import hist as ahist
from astropy.stats import sigma_clip
from statsmodels.nonparametric.kernel_density import KDEMultivariate

import os
import sys

# local
import find_pcs
import figures_tools
from read_results import PCAOutput

from importer import *

# personal
import manga_tools as m
import spec_tools

eps = np.finfo(float).eps

class ParamNames(object):
    '''
    defines interfaces to parameter names in PCA results writeouts, model libraries,
        and (optionally) DAP results
    '''
    def __init__(self, name, model_lib_name, results_name, TeX_name=None,
                 in_dap=False, dap_extname=None, dap_channel_or_key=None):
        self.name = name
        self.model_lib_name = model_lib_name
        self.results_name = results_name
        self.TeX_name = TeX_name
        self.in_dap = in_dap
        self.dap_extname = dap_extname
        self.dap_channel_or_key = dap_channel_or_key

class ModelDataComparisonFigure(object):
    '''
    figure that holds model-data comparisons for many parameters
    '''
    def __init__(self, results_files, param_names_list, modeltab):
        '''
        - results_files: list of FITS files to look for the
        - param_names_list: list of ParamNames objects that interface with
            PCA results, model libraries, and DAP results
        '''
        self.gs, self.fig = gen_gridspec_fig(len(param_names_list))

class ModelDataComparisonClass(object):
    dap_color = 'C0'
    pca_color = 'C1'
    model_color = 'C2'

class ModelDataHist1DComparison(ModelDataComparisonClass):
    '''
    scatter plot that holds comparison between PCA results, models, and
        DAP fits using a single 1d histogram
    '''
    def __init__(self, pname, all_results, modeltab, ax):
        self.pname = pname
        self.all_results = all_results
        self.modeltab = modeltab

    def _accumulate_DAP(self):
        '''
        accumulate DAP results
        '''

    def _accumulate_PCA(self):
        '''
        accumulate PCA results
        '''

        mask = np.concatenate([res.flattenedmap('MASK')
                              for res in self.all_results])
        # 10 models should have weight 10% as large as best-fit
        badpdf = np.concatenate([res.flattenedcubechannel('FRAC') < 10
                                 for res in self.all_results])
        val = np.concatenate(
            [res.flattenedcubechannel(self.pname.results_name)
             for res in self.all_results])
        return np.ma.array(val, mask=mask)

    def plot_data(self, spec_snr_min):



if __name__ == '__main__':
    import glob
    results_fnames = glob.glob('results/*/*results.fits')

    model_dir = '/usr/data/minhas2/zpace/CSPs/CSPs_CKC14_MaNGA_20180130-1/'
    all_models = glob.glob(os.path.join(model_dir, 'CSPs_*.fits'))
    modeltab = t.vstack(list(map(t.Table.read, all_models)
