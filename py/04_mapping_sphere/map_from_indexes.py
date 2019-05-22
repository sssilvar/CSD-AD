# /bin/env python3
import os
from multiprocessing import Pool
from os.path import join, dirname, realpath, isdir, isfile
from configparser import ConfigParser

import numpy as np
import pandas as pd
import nibabel as nb
from scipy import ndimage as ndi

import matplotlib.pyplot as plt

# Set root folder
root = dirname(dirname(dirname(realpath(__file__))))


def sobel_magnitude(nii_file):
    # Load file
    vol = nb.load(nii_file).get_data().astype(np.float)
    sobel_mode = 'reflect'
    sobel_x = ndi.sobel(vol, axis=0, mode=sobel_mode)
    sobel_y = ndi.sobel(vol, axis=1, mode=sobel_mode)
    sobel_z = ndi.sobel(vol, axis=2, mode=sobel_mode)
    return np.sqrt(sobel_x ** 2 + sobel_y ** 2 + sobel_z ** 2)


def grad_magnitude(nii_file):
    # Load file
    vol = nb.load(nii_file).get_data().astype(np.float)
    gx, gy, gz = np.gradient(vol)
    return np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)


def process_subject(subject):
    print('Starting mapping for subject {} ...'.format(subject))
    # Create subjects directory
    subj_brain_file = join(data_folder, subject, subj_file)

    if not isfile(subj_brain_file):
        "Check if file exists. Break otherwise"
        print('File {} does not exist.'.format(subj_brain_file))
        return 0

    out_subject_dir = join(out_folder, subject)
    if not isdir(out_subject_dir):
        os.mkdir(out_subject_dir)
    print('\t- Output: {}'.format(out_subject_dir))

    # Initialize mapped image
    img_mapped_sobel = np.zeros([dx, dy])
    img_mapped_grad = np.zeros_like(img_mapped_sobel)
    rois = np.zeros(vol_shape)

    vol_sobel = sobel_magnitude(subj_brain_file)
    vol_grad = grad_magnitude(subj_brain_file)

    for i, scale in enumerate(scales):
        print('Mapping scale {} ...'.format(scale))
        filename_base = join(out_subject_dir, 'sobel_{}'.format(scale))
        if not isfile(filename_base + '.npz'):
            for j, theta in enumerate(thetas):
                for k, phi in enumerate(phis):
                    # print('Mapping scale: {} | angle ({}, {})'.format(scale, theta, phi))
                    ix = ix_df.loc[
                        (ix_df['scale'] == scale) &
                        (ix_df['theta'] == theta) &
                        (ix_df['phi'] == phi)]['indexes'].values[0]
                    img_mapped_sobel[j, k] = vol_sobel[ix].mean()
                    img_mapped_grad[j, k] = vol_grad[ix].mean()

                    if theta == 0 and phi == 0:
                        rois[ix] = i + 1
            # Save results
            plt.imsave(filename_base + '.png', img_mapped_sobel.T, cmap='gray')
            np.savez_compressed(filename_base, img=img_mapped_sobel, grad=img_mapped_grad, indexes=index_file, brain_file=subj_file)
        else:
            print('File {} already exists.'.format(filename_base + '.png'))


if __name__ == '__main__':
    # Define params
    index_file = join(os.getenv('HOME'), 'Downloads', 'indexes_tk_25_overlap_9_ns_2.h5')
    cfg_file = join(root, 'config', 'config.cfg')
    subj_file = 'brainmask_reg.nii.gz'
    vol_shape = (256, 256, 256)

    cfg = ConfigParser()
    cfg.read(cfg_file)
    data_folder = cfg.get('dirs', 'dataset_folder_registered')
    out_folder = cfg.get('dirs', 'sphere_mapping')
    n_cores = cfg.getint('resources', 'n_cores')
    group_file = join(data_folder, 'groupfile.csv')

    # Print info
    sep = 15 * '='
    print(sep + ' SPHERE MAPPER ' + sep)
    print('\t- Indexes file: %s' % index_file)
    print('\t- Config. file: %s' % cfg_file)
    print('\t- Data folder : {}'.format(data_folder))
    print('\t- Output folder : {}'.format(out_folder))
    print('\t- Group file: {}'.format(group_file))
    print('\t- Volume shape: {}'.format(vol_shape))
    print('\t- Cores to be used: {}'.format(n_cores))

    # Load indexes file
    # and subjects file
    ix_df = pd.read_hdf(index_file, key='indexes', low_memory=False)
    subjects = pd.read_csv(group_file, index_col=0)

    # Get scales, angles and mapped image dimensions
    scales = ix_df['scale'].value_counts().sort_index().index
    thetas = ix_df['theta'].value_counts().sort_index().index
    phis = ix_df['phi'].value_counts().sort_index().index
    dx, dy = len(thetas), len(phis)

    # Process subjects in parallel
    pool = Pool(n_cores)
    pool.map(process_subject, subjects.index)
    pool.close()
