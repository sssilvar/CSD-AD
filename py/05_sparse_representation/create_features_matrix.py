#!/bin/env python2
from __future__ import print_function

import os
import sys
import glob
from os.path import join, realpath, dirname, basename, splitext, isdir
from configparser import ConfigParser

import pyct as ct
import numpy as np
import pandas as pd
from scipy.stats import describe, gennorm

# Set root folder
root = dirname(dirname(dirname(realpath(__file__))))

sys.path.append(join(root))
from lib.curvelets import get_sub_bands

if __name__ == '__main__':
    # Load configuration file
    cfg_file = join(root, 'config', 'config.cfg')
    cfg = ConfigParser()
    cfg.read(cfg_file)
    nbs = 4
    nba = 32

    # Get subjects folder
    mapped_subjects_dir = cfg.get('dirs', 'sphere_mapping')
    out_folder = join(mapped_subjects_dir, 'curvelet')
    output_file = join(out_folder, 'sobel_curvelet_features_non_split_{}_scales_{}_angles.csv'.format(nbs, nba))
    print('\t- Mapped subjects folder: {}'.format(mapped_subjects_dir))
    print('\t- Output folder: {}'.format(out_folder))
    print('\t- Output file: {}'.format(output_file))

    # Find NPZ files
    raw_files = glob.glob(join(mapped_subjects_dir, '**', '*.npz'))#, recursive=True)

    # Create features matrix
    features = pd.DataFrame()

    for f in raw_files:
        subject_id = f.split('/')[-2]
        file_basename = splitext(basename(f))[0].split('_')
        scale = '_to_'.join(file_basename[-2:])
        img_type = file_basename[0]
        print(f, scale, subject_id)

        # Load image
        img = np.load(f)['img']

        # Curvelet analysis
        A = ct.fdct2(
            img.shape,
            nbs=nbs,
            nba=nba,
            ac=True,
            norm=False,
            vec=True,
            cpx=False)

        feats = pd.Series()
        feats.name = subject_id
        feats['sphere'] = scale
        f = A.fwd(img)  # Forward transformation
        data_dict = get_sub_bands(A, f)  # Transform into a dictionary
        for key, val in data_dict.items():
            beta_est, mean_est, var_est = gennorm.fit(val)
            feats[key + '_beta'] = beta_est
            feats[key + '_mean'] = mean_est
            feats[key + '_var'] = var_est

        features = features.append(feats)

    # Create folder if does not exist
    if not isdir(out_folder):
        os.mkdir(out_folder)

    print(features.head())
    features.to_csv(output_file)
    print('Done!')

