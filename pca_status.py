from importer import *

import os
import numpy as np

def log_file_exists(plateifu):
    status_file_dir = os.path.join(
        os.environ['STELLARMASS_PCA_RESULTSDIR'], 'results',
        plateifu)

    return os.path.exists(os.path.join(status_file_dir, '{}.log'.format(plateifu)))

def write_log_file(plateifu, msg):
    '''write a log file
    '''
    status_file_dir = os.path.join(
        os.environ['STELLARMASS_PCA_RESULTSDIR'], 'results',
        plateifu)

    if not os.path.exists(status_file_dir):
        os.makedirs(status_file_dir)

    with open(os.path.join(status_file_dir, '{}.log'.format(plateifu)), 'w') as logf:
        logf.write(msg)

def summary_remaining(drpall, group_col='ifudesignsize'): 
    remaining = drpall[ 
        ~np.array(list(map(log_file_exists, drpall['plateifu'])))] 
    ifusize_grps = remaining.group_by(group_col) 
    print('remaining galaxies by {}'.format(group_col)) 
    for k, g in zip(ifusize_grps.groups.keys, ifusize_grps.groups): 
        print(k[group_col], ':', len(g))

if __name__ == '__main__':
    import manga_tools as m

    drpall = m.load_drpall(mpl_v)
    drpall = drpall[(drpall['ifudesignsize'] > 0) * (drpall['nsa_z'] != -9999.)]
    print(drpall)
    summary_remaining(drpall)