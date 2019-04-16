#!/bin/env puthon3
from os.path import join, dirname, realpath

import numpy as np
import pandas as pd
import nibabel as nb

from  sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

root = dirname(dirname(dirname(realpath(__file__))))

if __name__ == '__main__':
    # Load groupfile
    csv_file = '/run/media/ssilvari/SMITH_DATA_1TB/Universidad/MSc/Thesis/Dataset/ADNI_FS_ZIP/groupfile.csv'
    dataset_folder = '/run/media/ssilvari/SMITH_DATA_1TB/Universidad/MSc/Thesis/Dataset/ADNI_FS_registered_flirt'
    subjects = pd.read_csv(csv_file, index_col=0)
    adnimerge = pd.read_csv(join(root, 'param', 'df_conversions_with_times.csv'), index_col='PTID')

    # Load images
    X = []
    for subject in subjects.index[:20]:
        try:
            label = adnimerge.loc[subject, 'target']
            if label == 'MCIc' or label == 'MCInc':
                nii_file = join(dataset_folder, subject, 'orig_reg.nii.gz')
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
        nb.save(nii_pca, '/tmp/PC{}_pca.nii.gz'.format(i + 1))
