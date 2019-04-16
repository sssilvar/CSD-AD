#!/bin/env puthon3
from os.path import join

import numpy as np
import pandas as pd
import nibabel as nb

from  sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
    # Load groupfile
    csv_file = '/run/media/ssilvari/SMITH_DATA_1TB/Universidad/MSc/Thesis/Dataset/ADNI_FS_ZIP/groupfile.csv'
    dataset_folder = '/run/media/ssilvari/SMITH_DATA_1TB/Universidad/MSc/Thesis/Dataset/ADNI_FS_registered_flirt'
    subjects = pd.read_csv(csv_file, index_col=0)

    # Load images
    X = []
    for subject in subjects.index[:60]:
        nii_file = join(dataset_folder, subject, 'orig_reg.nii.gz')
        nii = nb.load(nii_file)
        nii_data = nii.get_data().ravel()
        print('Loading subject: {} | shape: {}'.format(subject, nii_data.shape))

        X.append(nii_data)

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
        nb.save(nii_pca, '/tmp/PC{}_pca.nii.gz'.format(i + 1))
