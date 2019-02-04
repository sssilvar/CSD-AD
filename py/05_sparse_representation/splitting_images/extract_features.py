#!/bin/env python2
import os
import argparse
from os.path import join, dirname, realpath, basename, isfile

import pandas as pd

current_dir = dirname(realpath(__file__))
curvelet_script = join(current_dir, 'curvelet_decomposition.py')


def read_subj_feats_in_sphere(csv_features, raw_file):
    df_subj = pd.read_csv(csv_features, index_col=0)
    df_subj['sphere'] = '_'.join(basename(raw_file).split('_')[1:4])
    return df_subj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curvelet decomposition for a dataset.")
    parser.add_argument('-g', metavar='--groupfile',
                        help='CSV containing all the subjects to process',
                        type=str,
                        required=True)
    parser.add_argument('-sd', metavar='--subjects-dir',
                        help='Folder containing subjects directory.',
                        type=str,
                        default=None)
    parser.add_argument('-s', metavar='--scales',
                        help='Number of scales.',
                        type=int,
                        required=True)
    parser.add_argument('-a', metavar='--angles',
                        help='Number of angles.',
                        type=int,
                        required=True)

    args = parser.parse_args()

    # Assign parsed
    groupfile = args.g
    subjects_dir = dirname(groupfile) if args.sd is None else args.sd
    nbs, nba = args.s, args.a

    # Print info
    print('[  INFO  ] Processing parameters:')
    print('\t- Groupfile CSV: %s' % groupfile)
    print('\t- Output folder: %s' % subjects_dir)
    print('\t- Number of scales: %d' % nbs)
    print('\t- Number of angles: %d' % nba)

    # Load subjects and start the magic
    df = pd.read_csv(groupfile, index_col=0)
    df_int = pd.DataFrame()
    df_grad = pd.DataFrame()
    df_sobel = pd.DataFrame()

    for subject in df.index:
        print('[  INFO  ] Processing %s' % subject)

        # Build command
        # raw_folder = join(subjects_dir, subject, 'raw')
        raw_folder = join(subjects_dir, subject)  # For some cases
        raw_files = [join(raw_folder, rf) for rf in os.listdir(raw_folder) if rf.endswith('.raw')]

        for raw_file in raw_files:
            subject_feats_file = join(subjects_dir, subject, 'curvelet',
                                      basename(raw_file)[:-4] + '_curvelet_%d_%d.csv' % (nbs, nba))

            if not isfile(subject_feats_file):
                cmd = 'python2 %s -f %s -sid %s -s %d -a %d -out %s' % (
                    curvelet_script, raw_file, subject, nbs, nba, subjects_dir)

                print('\t- Decomposing: %s' % raw_file)
                os.system(cmd)

                subj_feats_df = read_subj_feats_in_sphere(subject_feats_file, raw_file)
                print(subj_feats_df['sphere'])

                print('[  INFO  ] Appending to DataFrame: %s' % basename(subject_feats_file))
                if 'gradient' in raw_file:
                    df_grad = df_grad.append(subj_feats_df)
                elif 'intensity' in raw_file:
                    df_int = df_int.append(subj_feats_df)
                elif 'sobel' in raw_file:
                    df_sobel = df_sobel.append(subj_feats_df)
            else:
                print('[  WARN  ] File already exists: {}'.format(raw_file))
                subj_feats_df = read_subj_feats_in_sphere(subject_feats_file, raw_file)
                if 'gradient' in raw_file:
                    df_grad = df_grad.append(subj_feats_df)
                elif 'intensity' in raw_file:
                    df_int = df_int.append(subj_feats_df)
                elif 'sobel' in raw_file:
                    df_sobel = df_sobel.append(subj_feats_df)

    # Save results
    df_grad.to_csv(join(subjects_dir, 'gradient_curvelet_features_%d_scales_%d_angles.csv' % (nbs, nba)))
    df_int.to_csv(join(subjects_dir, 'intensity_curvelet_features_%d_scales_%d_angles.csv' % (nbs, nba)))
    df_sobel.to_csv(join(subjects_dir, 'sobel_curvelet_features_%d_scales_%d_angles.csv' % (nbs, nba)))
