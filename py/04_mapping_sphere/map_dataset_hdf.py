#!/bin/env python3
import os
import sys
from os.path import join, dirname, realpath

import numpy as np
import pandas as pd
import nibabel as nb
from scipy.ndimage import filters

from nilearn import plotting
import matplotlib.pyplot as plt

# Set root folder and append it to path
root = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(root)

from lib.masks import solid_cone
from lib.transformations import rotate_vol
from lib.geometry import extract_sub_volume, get_centroid

if __name__ == '__main__':
    # Set dataset folder and csv with subjects
    data_folder = '/run/media/ssilvari/HDD/ADNI_FS_registered_flirt'
    group_csv = join(data_folder, 'groupfile.csv')
    mni_file = join(root, 'param', 'FSL_MNI152_FreeSurferConformed_1mm.nii')
    subj_file = 'brainmask_reg.nii.gz'

    # Load group file
    subjects = pd.read_csv(group_csv, index_col=0)

    # Calculate the inner and outer radius
    # for all the spheres: scales
    max_radius = 25
    tk = 25
    overlap = 0
    n_spheres = max_radius // (tk - overlap)
    scales = [(i * (tk - overlap), ((i + 1) * tk) - (i * overlap)) for i in range(n_spheres)]
    print('Number of scales: {} | Scales: {}'.format(len(scales), scales))

    # Start go over the whole sphere (x_angle: [0, pi] and z_angle [-pi, pi])
    ns = 2  # TODO: Check if it's necessary to change it

    # Get centroid of MNI152
    mni_aseg = nb.load(mni_file)
    centroid = tuple(get_centroid(mni_aseg.get_data() > 0))
    print('Centroid of MNI152: {}'.format(centroid))

    # Create pixels dataframe
    pix_df = pd.DataFrame()
    img = filters.sobel(nb.load(join(data_folder, '002_S_0729', subj_file)).get_data())

    # Iterate over scales
    for r_min, r_max in scales:
        # Initialize images
        img_2d = np.zeros([360 // ns, 180 // ns])
        img_grad_2d = np.zeros_like(img_2d)
        img_sobel_2d = np.zeros_like(img_2d)
        pix_count = 0

        # Compute Solid cone and extract sub-volume: sc, sub_mask
        sc = solid_cone(radius=(r_min, r_max), center=centroid)
        mask_sub, center = extract_sub_volume(sc, radius=(r_min, r_max), centroid=centroid)
        print('Radius range: ({} to {})'.format(r_min, r_max))

        # Iterate over angles (theta, phi)
        for i, z_angle in enumerate(range(-180, 180, ns)):
            for j, x_angle in enumerate(range(0, 180, ns)):
                solid_ang_mask = rotate_vol(mask_sub, angles=(x_angle, 0, z_angle))  # ROI
                pix_count += 1

                # Iterate over subjects
                for subject in [subjects.index[0]]:
                    # Load subject and process it
                    # img = filters.sobel(nb.load(join(data_folder, subject, subj_file)).get_data())
                    vol_sub, _ = extract_sub_volume(img, radius=(r_min, r_max), centroid=centroid)
                    roi_mean = np.nan_to_num(vol_sub[np.where(solid_ang_mask)].mean())

                    # Add result to dataframe
                    col_name = '{}_to_{}_{}'.format(r_min, r_max, pix_count)
                    pix_df.loc[subject, col_name] = roi_mean

                    # # Create images
                    # nii = nb.Nifti1Image(vol_sub, mni_aseg.affine)
                    # roi = nb.Nifti1Image(solid_ang_mask.astype(np.int8), mni_aseg.affine)
                    #
                    # display = plotting.plot_anat(nii)
                    # display.add_overlay(roi, alpha=0.8)
                    # plt.show()

                    print('Scale: {} | Angle: {} | Processing subject: {}'.format((r_min, r_max), (z_angle, x_angle),
                                                                                  subject), roi_mean)

                # print(pix_df.head())
                print('-' * 10)

    out_file = '/tmp/mapped.h5'
    pix_df.to_hdf(out_file, key='pixels')
    df_test = pd.read_hdf(out_file, key='pixels')

    print((df_test - pix_df).head())  # Differences (integrity)
    ix = ['{}_to_{}_{}'.format(0, 25, i) for i in range(1, np.prod(img_2d.shape).astype(int) + 1)]
    img = np.reshape(df_test.loc['002_S_0729', ix].values, img_2d.shape)

    plt.imshow(img, cmap='gray')
    plt.show()
