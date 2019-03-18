from importer import *

import os

from astropy.io import fits

def find_sfh_ixs(i, nsfhperfile, nsubpersfh):
    # `i` is the pure number SFH;
    # find file number `fnum` by dividing by mper
    nspecperfile = nsfhperfile * nsubpersfh

    # file index in filelist
    fnum = i // nspecperfile

    # record index within individual spectral-data file
    fi = i % nspecperfile

    # record number in the corresponding param table
    fii = fi // nsubpersfh

    # row number
    fiii = fi % nsubpersfh

    return fnum, fi, fii, fiii

def retrieve_SFHs(filelist, i, massnorm='mformed', nsubpersfh=None, nsfhperfile=None):
    '''
    fetch a time array & SFH from a series of FITS archives,
        each with `nsfhperfile` SFHs & `nsubpersfh` Z/tau/mu realizations for
        each SFH
    '''

    if nsfhperfile is None:
        nsfhperfile = fits.getval(filelist[0], ext=0, keyword='NSFHPER')
    if nsubpersfh is None:
        nsubpersfh = fits.getval(filelist[0], ext=0, keyword='NSUBPER')

    fnum, fi, fii, fiii = find_sfh_ixs(i, nsfhperfile, nsubpersfh)

    '''
    print('trainer {0}: spectral:file-spec {1}-{2}; SFH:file-rec-subsample {1}-{3}-{4}'.format(
          i, fnum, fi, fii, fiii))
    '''
    fname = filelist[fnum]

    allts = fits.getdata(fname, extname='allts')
    allsfhs = np.repeat(fits.getdata(fname, extname='allsfhs'),
                        nsubpersfh, axis=0)

    # normalize mass either by total mass formed or current stellar mass
    mtot = fits.getdata(fname, massnorm)[:, None]

    return allts, allsfhs / mtot, fi

def retrieve_meta_table(filelist, i, nsubpersfh=None, nsfhperfile=None):
    '''
    '''

    if nsfhperfile is None:
        nsfhperfile = fits.getval(filelist[0], ext=0, keyword='NSFHPER')
    if nsubpersfh is None:
        nsubpersfh = fits.getval(filelist[0], ext=0, keyword='NSUBPER')

    fnum, fi, fii, fiii = find_sfh_ixs(i, nsfhperfile, nsubpersfh)

    fname = filelist[fnum]

    metadata = t.Table.read(fname)

    return metadata, fii