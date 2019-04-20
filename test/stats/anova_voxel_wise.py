#!/bin/env puthon3
__author__ = 'Santiago Smith'
__description__ = 'An automated method for ROI extraction in groups of MRI'
__year__ = '2019'
__cite__ = """
Silva, et al. Characterizing brain patterns in conversion from mild cognitive impairment (MCI) to Alzheimer's disease,
13th International Symposium on Medical Information Processing and Analysis, 2017
"""


import argparse
from os.path import join, dirname, realpath

import numpy as np
import pandas as pd
import nibabel as nb

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

root = dirname(dirname(dirname(realpath(__file__))))


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(
        description='{desc}\n\nCitation:{cite}'.format(desc=__description__, cite=__cite__),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-groupfile', help='CSV containing subject IDs', required=True)
    parser.add_argument('-out', help='Output folder', required=True)
    parser.add_argument('-folder', help='Subjects folder', required=False)

    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Load groupfile
    csv_file = args.groupfile
    dataset_folder = args.folder
    out_folder = args.out

    # Load data
    subjects = pd.read_csv(csv_file, index_col=0)
    adnimerge = pd.read_csv(join(root, 'param', 'df_conversions_with_times.csv'), index_col='PTID')

    # Load images
    X = []
    for subject in subjects.index:
        try:
            label = adnimerge.loc[subject, 'target']
            if label == 'MCIc' or label == 'MCInc':
                nii_file = join(dataset_folder, subject, '001_reg.nii.gz')
                nii = nb.load(nii_file)
                nii_data = nii.get_data().ravel()
                print('Loading subject: {} | shape: {} | Dx: {}'.format(subject, nii_data.shape, label))
                X.append(nii_data)
            else:
                raise KeyError
        except KeyError:
            print('Subject {} not an MCIc/MCInc'.format(subject))

    X = np.vstack(X)
    print('Dataset shape: {}'.format(X.shape))

    # Pipeline for variance analysis
    scaler = StandardScaler()
    pca = PCA(n_components=5)

    pipeline = Pipeline([
        ('scaler', scaler),
        ('pca', pca)
    ])
    pipeline.fit(X)

    components = pca.components_
    for i, component in enumerate(components):
        print('Range PC{} ({},{})'.format(i + 1, component.min(), component.max()))

        nii_pca = nb.Nifti1Image(component.reshape(nii.shape).astype(np.float), nii.affine)
        nb.save(nii_pca, join(out_folder, 'PC{}_pca.nii.gz'.format(i + 1)))
