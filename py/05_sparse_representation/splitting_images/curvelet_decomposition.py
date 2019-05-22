#!/bin/env python2
import os
import sys
import argparse
from os.path import basename, dirname, exists, join, splitext, realpath

import pyct as ct
import numpy as np
import pandas as pd
from scipy.stats import describe, gennorm

import matplotlib.pyplot as plt

root = dirname(dirname(dirname(dirname(realpath(__file__)))))
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
    data_file = args.f
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

    # Set create (if not exist) and save individual results
    if not exists(dirname(out_folder)):
        os.mkdir(dirname(out_folder))
        os.mkdir(out_folder)
    elif not exists(out_folder):
        os.mkdir(out_folder)

    # Load img and split it in half
    img = np.fromfile(data_file).reshape([180, 90])
    img_list = np.split(img, 2, axis=0)

    # Define a curvelet object
    A = ct.fdct2(
        (90,90), 
        nbs=nbs, 
        nba=nba, 
        ac=True, 
        norm=False, 
        vec=True, 
        cpx=False)
   
    # # Visualize if necessary
    # plt.subplot(1,2,1)
    # plt.imshow(img_list[0].T, cmap='gray')
    # plt.axis('off')
    # plt.title('Left')
    # plt.subplot(1,2,2)
    # plt.imshow(img_list[1].T, cmap='gray')
    # plt.axis('off')
    # plt.title('Right')
    # plt.show()

    # Decompose
    feats = pd.Series()
    feats.name = subject_id
    for i, img in enumerate(img_list):
        side = 'left' if i is 0 else 'right'  # Define side
        print('\t- Part %d shape (%s): %s' % ((i + 1), side, img.shape))

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
        
        # Set features
        # for scale in range(nbs):
        #     for angle in range(nba):
        #         try:
        #             ix = A.index(scale, angle)
        #             data = f[ix[0]:ix[1]] # Extract magnitude

        #             # Extract several statistics
        #             n, (mi, ma), mea, var, skew, kurt = describe(data)

        #             # Assign to series
        #             feats['%s_%d_%d_n' % (side, scale, angle)] = n
        #             feats['%s_%d_%d_min' % (side, scale, angle)] = mi
        #             feats['%s_%d_%d_max' % (side, scale, angle)] = ma
        #             feats['%s_%d_%d_mean' % (side, scale, angle)] = mea
        #             feats['%s_%d_%d_var' % (side, scale, angle)] = var
        #             feats['%s_%d_%d_skew' % (side, scale, angle)] = skew
        #             feats['%s_%d_%d_kurtosis' % (side, scale, angle)] = kurt
        #         except IndexError:
        #             pass

    # Save as CSV
    filename, ext = splitext(data_file)
    filename = basename(filename)
    filename = join(out_folder, filename + '_curvelet_%d_%d.csv' % (nbs, nba))
    
    feats = pd.DataFrame(feats).T  # As a row observation (instead of column)
    feats.to_csv(filename)
    
