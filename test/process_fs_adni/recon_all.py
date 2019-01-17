#!/bin/env python
import os
import argparse
from multiprocessing import Pool

import pandas as pd

__description__ = """
One script to rule them all.
This script executes Freesurfer recon-all command over a whole dataset (ADNI)
USAGE:

    python recon_all.py -g [groupfile.csv] -f [dataset_folder] -o [output_folder]
"""

def parse_args():
    parser = argparse.ArgumentParser(description=__description__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-g', required=True, metavar='--groupfile', help='A CSV containing subject IDs to be processed.')
    parser.add_argument('-f', required=True, metavar='--folder', help='Folders where subjects are')
    parser.add_argument('-o', required=True, metavar='--output', help='Folders output.')
    parser.add_argument('-c', required=True, metavar='--cores', help='Number of cores used.', type=int)
    parser.add_argument('-ext', metavar='--extension', help='Extension of the images.', default='.nii')
    return parser.parse_args()


def get_list_of_files(dataset_folder, groupfile, ext):
    mr_files = []
    for root, dirs, files in os.walk(dataset_folder):
        for mr_file in files:
            if mr_file.endswith(ext):
                mr_files.append(os.path.join(root, mr_file))
    # Load groupfile an delete repeated files
    df = pd.read_csv(groupfile, index_col=0)
    mr_files_filtered = []
    for subject in df.index:
        subj_files = [mf for mf in mr_files if subject in mf]
        if len(subj_files):
            mr_files_filtered.append(subj_files[-1])
        else:
            print('[  ERROR  ] File(s) for subject {} not found.'.format(subject))
    return mr_files_filtered


def recon_all(f):
    # Extract subject ID from filename
    fn = os.path.basename(f).split('_')
    subject = '_'.join(fn[1:4])
    
    # Create command
    cmd = 'recon-all -i {} -s {} -sd {} -autorecon1'.format(f, subject, '/dev/shm')
    # cmd = 'mri_convert {} {}.mgz'.format(f, os.path.join('/dev/shm',subject))
    os.system(cmd)
    os.system('cd /dev/shm && zip -r {}.zip {}'.format(os.path.join(output_folder, subject), subject))
    os.system('rm -rf /dev/shm/{}'.format(subject))


if __name__ == "__main__":
    os.system('clear')
    # Parse arguments to variables
    args = parse_args()
    groupfile = args.g
    dataset_folder = args.f
    output_folder = args.o
    ext = args.ext
    n_cores = args.c

    # Look for files
    mri_files = get_list_of_files(dataset_folder, groupfile, ext)

    # Apply recon-all
    pool = Pool(n_cores)
    pool.map(recon_all, mri_files)
    pool.close()