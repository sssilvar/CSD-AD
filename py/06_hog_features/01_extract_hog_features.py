import os
import logging
import argparse
from configparser import ConfigParser
from os.path import join, dirname, basename, realpath, isdir, isfile

import numpy as np
import pandas as pd
from skimage import exposure
from skimage import feature

import matplotlib.pyplot as plt

# Get root folder
root = dirname(dirname(dirname(realpath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='HOG feature extractor for mapped images in 2D.')
    parser.add_argument('-groupfile',
                        help='File containing the subjects to process',
                        default='/home/ssilvari/Documents/temp/ADNI_temp/ADNI_FS_sphere_mapped/groupfile_test.csv'
                        )
    return parser.parse_args()


if __name__ == '__main__':
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # Parse arguments
    args = parse_args()
    group_file = args.groupfile

    # Load config file
    cfg = ConfigParser()
    cfg.read(join(root, 'config', 'config.cfg'))

    # Load parameters
    n_cores = cfg.getint('resources', 'n_cores')
    data_folder = cfg.get('dirs', 'sphere_mapping')
    out_folder = join(data_folder, 'HOG')

    # Check if folder does not exist
    if not isdir(out_folder):
        os.mkdir(out_folder)

    # Read group file
    df = pd.read_csv(group_file, index_col=0)
    print('Number of subjects to process: {}'.format(df.shape[0]))

    # DataFrames where results will be saved
    df_int = pd.DataFrame()
    df_grad = pd.DataFrame()
    df_sobel = pd.DataFrame()

    for sid in df.index:
        # Find RAW files
        print('Processing subject {} ...'.format(sid))
        raw_folder = join(data_folder, sid)
        raw_files = [join(raw_folder, rf) for rf in os.listdir(raw_folder) if rf.endswith('.raw')]
        subject_out_folder = join(out_folder, sid)

        # Create output subject dir
        if not isdir(subject_out_folder):
            print('Creating subject folder...')
            os.mkdir(subject_out_folder)

        for raw_file in raw_files:
            subject_feats_file = join(subject_out_folder,
                                      basename(raw_file)[:-4] + '_hog.csv')

            if not isfile(subject_feats_file):
                img = np.fromfile(raw_file).reshape([180, 90])

                # Extract HOG features
                fd = feature.hog(img,
                                 orientations=16,
                                 pixels_per_cell=(16, 16),
                                 cells_per_block=(2, 2),
                                 visualize=False,
                                 block_norm='L2-Hys')

                subject_series = pd.Series(fd, name=sid)
                subject_series['sphere'] = '_'.join(basename(raw_file).split('_')[1:4])
                subject_series.to_csv(subject_feats_file, header=True)

                # Append to the corresponding dataframe
                if 'intensity' in raw_file:
                    df_int = pd.concat([df_int, subject_series], axis='columns', join='outer', sort=False)
                elif 'gradient' in raw_file:
                    df_grad = pd.concat([df_grad, subject_series], axis='columns', join='outer', sort=False)
                elif 'sobel' in raw_file:
                    df_sobel = pd.concat([df_sobel, subject_series], axis='columns', join='outer', sort=False)
            else:
                subject_series = pd.read_csv(subject_feats_file, index_col=0)

                # Append to the corresponding dataframe
                if 'intensity' in raw_file:
                    df_int = pd.concat([df_int, subject_series], axis='columns', join='outer', sort=False)
                elif 'gradient' in raw_file:
                    df_grad = pd.concat([df_grad, subject_series], axis='columns', join='outer', sort=False)
                elif 'sobel' in raw_file:
                    df_sobel = pd.concat([df_sobel, subject_series], axis='columns', join='outer', sort=False)

    # Transpose dataframes
    dataframes = {
        'intensity': df_int.transpose(),
        'gradient': df_grad.transpose(),
        'sobel': df_sobel.transpose()
    }

    # Save them as CSV
    for im_type, df in dataframes.items():
        csv_file = join(out_folder, '{}_hog_features.csv'.format(im_type))
        df.to_csv(csv_file)
