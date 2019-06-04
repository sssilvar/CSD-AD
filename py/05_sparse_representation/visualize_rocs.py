#!/bin/env python3
import os
import argparse
from configparser import ConfigParser
from os.path import join, dirname, realpath, isdir, isfile

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

root = dirname(dirname(dirname(realpath(__file__))))
matplotlib.use('Agg')
plt.style.use('ggplot')


def parse_args():
    parser = argparse.ArgumentParser(description='Compile classification results in a single conversion-time ROC')
    parser.add_argument('-folder', default=None)
    parser.add_argument('-folds', default=10, type=int)
    parser.add_argument('-nbs', type=int, default=4)
    parser.add_argument('-nba', type=int, default=32)

    parser.add_argument('-tk', type=int, default=25)
    parser.add_argument('-overlap', type=int, default=0)
    parser.add_argument('-ns', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Load config
    cfg = ConfigParser()
    cfg.read(join(root, 'config', 'config.cfg'))
    fmt = 'png'  # Alt. 'eps'

    # Number of scales and angles
    nbs = args.nbs
    nba = args.nba

    # Set Sphere thickness, overlapping and sampling rate (deg)
    tk = args.tk
    overlap = args.overlap
    ns = args.ns

    # Load params
    if args.folder is None:
        data_folder = join(
            cfg.get('dirs', 'sphere_mapping'),
            f'ADNI_FS_mapped_tk_{tk}_overlap_{overlap}_ns_{ns}',
            'curvelet'
        )
    else:
        data_folder = args.folder

    roc_folder = join(data_folder, 'ROC')
    n_folds = args.folds

    # Classifiers, image types and months
    classifiers = ['svm', 'rf']
    img_types = ['gradient', 'sobel']
    times = [24, 36, 60]

    for img_type in img_types:
        for clf in classifiers:
            clf_name = 'Random Forest' if clf == 'rf' else 'SVM'
            plt.figure(figsize=(7, 7))
            for t in times:
                # gradient_curvelet_features_non_split_aio_5_fold_svm_60_months_final.csv
                file_pattern = f'{img_type}_curvelet_features_non_split_{nbs}_scales_{nba}_angles_aio_{n_folds}_' \
                    f'fold_{clf}_{t}_months_final.csv'
                data_file = join(roc_folder, file_pattern)

                if not isfile(data_file):
                    print(f'File {data_file} not found')
                    continue

                # Load data as DataFrame and
                # Plot ROC
                df = pd.read_csv(data_file)
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(
                    df['mean_fpr'], df['mean_tpr'],
                    label='{time} (AUC = {auc:.2f})'.format(
                        time='{} Months'.format(t),
                        auc=df['mean_auc'][0]),
                    linewidth=2.5,
                    alpha=0.8
                )
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.legend()
                plt.title('Mean ROC - {} - {} - {} folds'.format(clf_name, img_type.capitalize(), n_folds))
            # Save figures
            fig_folder = join(roc_folder, 'roc_by_month')
            if not isdir(fig_folder):
                os.mkdir(fig_folder)

            fig_file = join(
                fig_folder,
                '{img_type}_{clf}_{folds}_folds'.format(
                    img_type=img_type,
                    clf=clf_name.lower().replace(' ', '_'),
                    folds=n_folds
                )
            )
            print('Saving figure at: {}'.format(fig_file))
            plt.savefig(fig_file + '.{}'.format(fmt), bbox_inches='tight')
