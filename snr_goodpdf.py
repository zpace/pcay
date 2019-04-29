import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.table as t
from astropy.io import fits

import os, sys, glob
from itertools import product as iterprod

from importer import *

# personal
import manga_tools as m

from figures_tools import savefig, gen_gridspec_fig

class PDFvsSNRFig(object):
    def __init__(self, fracs):
        self.fracs = fracs
        self.nfracs = len(fracs)

        # load figure and gridspec object
        self.gs, self.fig = gen_gridspec_fig(
            self.nfracs, spsize=(2.5, 1.))
        self.subplot_inds = list(iterprod(range(self.gs._nrows), range(self.gs._ncols)))

        self.fracs_plot_ix = dict(zip(sorted(fracs), self.subplot_inds))
        self.fig_axes = {f: self.fig.add_subplot(self.gs[ix[0], ix[1]])
                         for f, ix in self.fracs_plot_ix.items()}

        for f_, ax_ in self.fig_axes.items():
            ax_.tick_params(labelsize='x-small')
            ax_.set_xscale('log')
            ax_.set_xlim([.1, 200.])
            ax_.text(x=.2, y=.1, s='f = {:.2f}'.format(f_), size='x-small')

        self.fig.suptitle('SNR vs PDF population', size='x-small')

    @classmethod
    def from_PDFSNR_Result(cls, res):
        if not isinstance(res, PDFSNR_Result):
            raise TypeError('res must be an instance of PDFSNR_Result class')
        fracs = res.frac_dict.keys()
        return cls(fracs)

    def add_object(self, frac_dict, snr, **kwargs):
        for fk, fv in frac_dict.items():
            self.add_to_subplot(key=fk, frac=fv, snr=snr)

    def add_to_subplot(self, key, frac, snr, **kwargs):
        plotting_kwargs = dict(s=.5, edgecolor='None', alpha=.5)
        kw = {**plotting_kwargs, **kwargs}
        self.fig_axes[key].scatter(snr.flatten(), frac.flatten(), **kw)

    def savefig(self):
        savefig(self.fig, fname='snr_vs_goodpdf.png', fdir='.')

class PDFSNR_Result(object):
    def __init__(self, frac_dict, snr):
        self.snr = snr
        self.frac_dict = frac_dict

    @classmethod
    def from_fname(cls, fname):
        return cls(*retrieve_snr_fracs(fname))

    def __getitem__(self, k):
        return self.frac_dict[k]

def retrieve_snr_fracs(fn):
    hdulist = fits.open(fn)
    nfracs = hdulist['GOODFRAC'].data.shape[0]
    fracs = {hdulist['GOODFRAC'].header['FRAC{}'.format(fri)]: \
                 hdulist['GOODFRAC'].data[fri, :, :] for fri in range(nfracs)}
    snrs = hdulist['SNRMED'].data
    hdulist.close()
    return fracs, snrs

if __name__ == '__main__':
    res_fnames = glob.glob('results/*/*_res.fits')

    pdfsnr_results = list(map(PDFSNR_Result.from_fname, res_fnames))

    snrfig = PDFvsSNRFig.from_PDFSNR_Result(pdfsnr_results[0])

    for i, res in enumerate(pdfsnr_results):
        snrfig.add_object(frac_dict=res.frac_dict, snr=res.snr,
                          facecolor='C{}'.format(i % 10))

    snrfig.savefig()
