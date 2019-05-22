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


def curvelet_decomposition(A, img, sphere, subject_id):
    """
    Performs a curvelet decomposition
    :param sphere: <str> Inner and outer radius of the sphere analyzed.
    :param img: <numpy.ndarray> Image to be decomposed.
    :param A: Curvelet object
    :param subject_id: <str> Subject ID
    :return: <pandas.Series> Subject fratures: generalized gaussian parameters (mean, std, beta)
    for each curvelet sub-band
    """
    feats = pd.Series()
    feats.name = subject_id
    feats['sphere'] = sphere
    f = A.fwd(img)  # Forward transformation
    data_dict = get_sub_bands(A, f)  # Transform into a dictionary
    for key, val in data_dict.items():
        beta_est, mean_est, var_est = gennorm.fit(val)
        feats[key + '_beta'] = beta_est
        feats[key + '_mean'] = mean_est
        feats[key + '_var'] = var_est
    return feats


if __name__ == '__main__':
    # Load configuration file
    cfg_file = join(root, 'config', 'config.cfg')
    cfg = ConfigParser()
    cfg.read(cfg_file)
    nbs = 4
    nba = 32

    # Mapping parameters
    # tk = 25
    # overlap = 0
    # ns = 2
    # Alternative
    tk = int(sys.argv[1])
    overlap = int(sys.argv[2])
    ns = int(sys.argv[3])

    # Get subjects folder
    mapped_subjects_dir = join(
        cfg.get('dirs', 'sphere_mapping'),
        'ADNI_FS_mapped_tk_{}_overlap_{}_ns_{}'.format(tk, overlap, ns))
    out_folder = join(mapped_subjects_dir, 'curvelet')
    print('\t- Mapped subjects folder: {}'.format(mapped_subjects_dir))
    print('\t- Output folder: {}'.format(out_folder))

    # Find NPZ files
    raw_files = glob.glob(join(mapped_subjects_dir, '**', '*.npz'))  # , recursive=True)

    # Create features matrix
    features_grad = pd.DataFrame()
    features_sobel = pd.DataFrame()

    for raw_file in raw_files:
        subject_id = raw_file.split('/')[-2]
        file_basename = splitext(basename(raw_file))[0].split('_')
        scale = '_to_'.join(file_basename[-2:])
        img_type = file_basename[0]
        print(raw_file, scale, subject_id)

        # Load image
        np_data = np.load(raw_file)
        img_sobel = np_data['img']
        img_grad = np_data['grad']

        # Curvelet analysis
        A = ct.fdct2(
            img_sobel.shape,
            nbs=nbs,
            nba=nba,
            ac=True,
            norm=False,
            vec=True,
            cpx=False)

        # Decompose gradients
        feats_grad = curvelet_decomposition(A, img_grad, scale, subject_id)
        features_grad = features_sobel.append(feats_grad)
        # Decompose sobel
        feats_sobel = curvelet_decomposition(A, img_sobel, scale, subject_id)
        features_sobel = features_sobel.append(feats_sobel)

    # Create folder if does not exist
    if not isdir(out_folder):
        os.mkdir(out_folder)

    # Print head for DataFrames
    print(features_grad.head())
    print(features_sobel.head())

    # Save results as CSV
    features_basename = '_curvelet_features_non_split_{}_scales_{}_angles.csv'.format(nbs, nba)
    features_grad.to_csv(join(out_folder, 'gradient' + features_basename))
    features_sobel.to_csv(join(out_folder, 'sobel' + features_basename))

    print('Done!')
