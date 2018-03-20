__author__ = 'xin yang'

import os
import nibabel as nib
import numpy as np
import math

src_us_folder = 'G:/Temp/Data/src/us'
src_seg_folder = 'G:/Temp/Data/src/seg'

aug_us_folder = 'G:/Temp/Data/aug/us'
aug_seg_folder = 'G:/Temp/Data/aug/seg'

img_n = 1
rotate_theta = np.array([0, math.pi / 2])

# augmentation
aug_cnt = 0
for k in range(img_n):
    src_us_file = os.path.join(src_us_folder, (str(k) + '.nii'))
    src_seg_file = os.path.join(src_seg_folder, (str(k) + '_seg.nii'))
    # load .nii files
    src_us_vol = nib.load(src_us_file)
    src_seg_vol = nib.load(src_seg_file)
    # volume data
    us_vol_data = src_us_vol.get_data()
    us_vol_data = (np.array(us_vol_data)).astype('uint8')
    seg_vol_data = src_seg_vol.get_data()
    seg_vol_data = (np.array(seg_vol_data)).astype('uint8')
    # get refer affine matrix
    ref_affine = src_us_vol.affine

    ############### flip volume ###############
    flip_us_vol = np.fliplr(us_vol_data)
    flip_seg_vol = np.fliplr(seg_vol_data)
    # construct new volumes
    new_us_vol = nib.Nifti1Image(flip_us_vol, ref_affine)
    new_seg_vol = nib.Nifti1Image(flip_seg_vol, ref_affine)
    # save
    aug_us_file = os.path.join(aug_us_folder, (str(aug_cnt) + '.nii'))
    aug_seg_file = os.path.join(aug_seg_folder, (str(aug_cnt) + '_seg.nii'))
    nib.save(new_us_vol, aug_us_file)
    nib.save(new_seg_vol, aug_seg_file)

    aug_cnt = aug_cnt + 1

    ############### rotate volume ###############
    for t in range(len(rotate_theta)):
        print 'rotating %d theta of %d volume...' % (t, k)
        cos_gamma = np.cos(t)
        sin_gamma = np.sin(t)
        rot_affine = np.array([[1, 0, 0, 0],
                               [0, cos_gamma, -sin_gamma, 0],
                               [0, sin_gamma, cos_gamma, 0],
                               [0, 0, 0, 1]])
        new_affine = rot_affine.dot(ref_affine)
        # construct new volumes
        new_us_vol = nib.Nifti1Image(us_vol_data, new_affine)
        new_seg_vol = nib.Nifti1Image(seg_vol_data, new_affine)
        # save
        aug_us_file = os.path.join(aug_us_folder, (str(aug_cnt) + '.nii'))
        aug_seg_file = os.path.join(aug_seg_folder, (str(aug_cnt) + '_seg.nii'))
        nib.save(new_us_vol, aug_us_file)
        nib.save(new_seg_vol, aug_seg_file)

        aug_cnt = aug_cnt + 1