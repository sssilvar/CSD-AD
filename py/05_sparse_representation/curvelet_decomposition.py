#!/bin/env python2
import os
import sys
import argparse
from os.path import basename, dirname, exists, join, splitext, realpath, isfile

import pyct as ct
import numpy as np
import pandas as pd
from scipy.stats import describe, gennorm

import matplotlib.pyplot as plt

root = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(join(root))

from lib.curvelets import get_sub_bands


def parse_args():
    parser = argparse.ArgumentParser(description="Curvelet decomposition of a RAW binary file.")
    parser.add_argument('-f', metavar='--file',
                        help='File to  decompose.',
                        type=str,
                        required=True)
    parser.add_argument('-sid', metavar='--subject-id',
                        help='Subject\'s ID',
                        type=str,
                        required=True)
    parser.add_argument('-out', metavar='--output-folder',
                        help='Output folder',
                        type=str,
                        required=True)
    parser.add_argument('-s', metavar='--scales',
                        help='Number of scales.',
                        type=int,
                        required=True)
    parser.add_argument('-a', metavar='--angles',
                        help='Number of angles.',
                        type=int,
                        required=True)

    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    data_file = str(args.f)
    nbs = args.s
    nba = args.a
    subject_id = args.sid
    out_folder = join(args.out, subject_id, 'curvelet')

    # Print info
    print('[  INFO  ] Decomposing info:')
    print('\t- File: %s' % data_file)
    print('\t- Subject: %s' % subject_id)
    print('\t- Output folder: %s' % out_folder)
    print('\t- Number of scales: %d' % nbs)
    print('\t- Number of angles: %d' % nba)

    # Set output filename
    filename, ext = splitext(data_file)
    filename = basename(filename)
    filename = join(out_folder, filename + '_curvelet_%d_%d_non_split.csv' % (nbs, nba))

    if not isfile(filename):
        # Set create (if not exist) and save individual results
        if not exists(dirname(out_folder)):
            os.mkdir(dirname(out_folder))
            os.mkdir(out_folder)
        elif not exists(out_folder):
            os.mkdir(out_folder)

        # Load img and split it in half
        if data_file.endswith('.raw'):
            img = np.fromfile(data_file).reshape([180, 90])
        elif data_file.endswith('.npz'):
            img = np.load(data_file)['img']

        # Define a curvelet object
        A = ct.fdct2(
            img.shape,
            nbs=nbs,
            nba=nba,
            ac=True,
            norm=False,
            vec=True,
            cpx=False)

        # Decompose
        feats = pd.Series()
        feats.name = subject_id

        # Start decomposition
        f = A.fwd(img)
        print('[  INFO  ] Curvelet decomposition shape: %s' % str(f.shape))

        # Use generalized Gaussian to fit features
        data_dict = get_sub_bands(A, f)
        for key, val in data_dict.items():
            beta_est, mean_est, var_est = gennorm.fit(val)
            feats[key + '_beta'] = beta_est
            feats[key + '_mean'] = mean_est
            feats[key + '_var'] = var_est

        # Save as CSV
        feats = pd.DataFrame(feats).T  # As a row observation (instead of column)
        feats.to_csv(filename)
    else:
        print('[  WARN  ] File {} already exists. Delete it if you want to replace it.'.format(filename))

