#!/bin/env python3
import os
from os.path import join, dirname, realpath

import numpy as np
import pandas as pd
import nibabel as nb
import scipy.ndimage as ndi
from skimage import filters

from nilearn import plotting
import matplotlib.pyplot as plt

# Set root folder
root = dirname(dirname(dirname(realpath(__file__))))

if __name__ == '__main__':
    # Load image
    mni_file = join(root, 'param', 'FSL_MNI152_FreeSurferConformed_1mm.nii')
    subject_file = '/home/ssilvari/Documents/temp/ADNI_temp/ADNI_FS_registered_flirt/002_S_0729/brainmask_reg.nii.gz'

    # Load indexes file
    indexes_file = join(os.getenv('HOME'), 'Downloads', 'indexes_tk_25_overlap_9_ns_2.h5')
    indexes_df = pd.read_hdf(indexes_file, key='indexes', index_col=0)

    print(indexes_df.head())

    # Get masks
    mni_nii = nb.load(mni_file)
    mni_data = mni_nii.get_data()
    rois = np.zeros(mni_nii.shape)

    # Load subject and calculate edges
    subj_nii = nb.load(subject_file)
    subj_data = subj_nii.get_data().astype(np.float32)

    sobel_mode = 'reflect'
    sobel_x = ndi.sobel(subj_data, axis=0, mode=sobel_mode)
    sobel_y = ndi.sobel(subj_data, axis=1, mode=sobel_mode)
    sobel_z = ndi.sobel(subj_data, axis=2, mode=sobel_mode)
    vol_sobel = np.sqrt(subj_data ** 2 + sobel_y ** 2 + sobel_z ** 2)

    # Get scales
    scales = indexes_df['scale'].value_counts().sort_index().index
    thetas = indexes_df['theta'].value_counts().sort_index().index
    phis = indexes_df['phi'].value_counts().sort_index().index
    dx, dy = len(thetas), len(phis)

    img_mapped = np.zeros([dx, dy])

    print('Starting radial analysis')
    print('\t- Scales: {}'.format(scales))
    print('\t- Image dimmensions: ({}, {})'.format(dx, dy))
    print(mni_data.shape)

    for i, scale in enumerate(scales):
        print('Mapping scale {} ...'.format(scale))
        for j, theta in enumerate(thetas):
            for k, phi in enumerate(phis):
                # print('Mapping scale: {} | angle ({}, {})'.format(scale, theta, phi))
                ix = indexes_df.loc[
                    (indexes_df['scale'] == scale) &
                    (indexes_df['theta'] == theta) &
                    (indexes_df['phi'] == phi)]['indexes'].values[0]
                img_mapped[j, k] = vol_sobel[ix].mean()

                if theta == 0 and phi == 0:
                    rois[ix] = i + 1
        plt.figure()
        plt.imshow(img_mapped.T, cmap='gray')
        plt.title('Scale: {}'.format(scale))

    # Create NIFTI for ROIs
    rois_nii = nb.Nifti1Image(rois, mni_nii.affine)

    # Plot results
    display = plotting.plot_anat(mni_nii)
    display.add_overlay(rois_nii, cmap='jet')
    plt.show()

    print('Done!')
