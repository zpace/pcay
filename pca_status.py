from importer import *

import os

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


