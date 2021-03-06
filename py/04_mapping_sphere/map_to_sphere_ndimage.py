import os
import sys
from time import time
from configparser import ConfigParser
from os.path import join, dirname, realpath, isfile

import numpy as np
import pandas as pd
import nibabel as nb
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.ndimage import affine_transform, rotate

# Set root folder
root = dirname(dirname(dirname(realpath(__file__))))

# Append path to add lib
sys.path.append(root)
from lib.param import load_params
from lib.masks import solid_cone
from lib.visualization import show_mri
from lib.transformations import rotate_vol
from lib.geometry import extract_sub_volume, get_centroid


def process_image(folder):
    subject_dir = join(dataset_registered_folder, folder)
    brainmask_file = join(subject_dir, 'brainmask_reg.nii.gz')
    subject_output_dir = join(results_folder, folder)

    # Execute if file exists
    if isfile(brainmask_file):
        # Print info message
        print('[  INFO  ] Processing subject %s located in %s' % (folder, subject_dir))

        # Try creating a folder for each subject
        try:
            os.mkdir(subject_output_dir)
            print('[  OK  ] Folder created at: ' + subject_output_dir)
        except OSError:
            print('[  WARNING  ] Folder already exists.')

        # Load MRI image and aseg file
        mgz = nb.load(brainmask_file)
        img = mgz.get_data().astype(np.float)

        # Calculate the gradient: img_grad
        gx, gy, gz = np.gradient(img)
        img_grad = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

        scales = [
            (0, 25),
            (25, 50),
            (50, 75),
            (75, 100),
        ]

        for n_scale, scale in enumerate(scales):
            # Crete a solid angle from a scale: sa
            r_min, r_max = scale
            sa = solid_cone(radius=(r_min, r_max), center=centroid)

            # Start go over the whole sphere (x_angle: [0, pi] and z_angle [-pi, pi])
            img_2d = np.zeros([360, 180])
            img_grad_2d = np.zeros_like(img_2d)

            mask_sub, center = extract_sub_volume(sa, radius=(r_min, r_max), centroid=centroid)
            vol_sub, _ = extract_sub_volume(img, radius=(r_min, r_max), centroid=centroid)
            grad_sub, _ = extract_sub_volume(img_grad, radius=(r_min, r_max), centroid=centroid)

            for i, z_angle in enumerate(range(0, 180, 8)):
                ti = time()
                rot_a = rotate(mask_sub, angle=z_angle, axes=(0,2), reshape=False)
                for j, x_angle in enumerate(range(-270, 90, 8)):
                    solid_ang_mask = rotate(rot_a, angle=x_angle, axes=(0,1), reshape=False)
                    img_masked = np.multiply(vol_sub, solid_ang_mask)
                    grad_masked = np.multiply(grad_sub, solid_ang_mask)
                    # img_masked = vol_sub * solid_ang_mask
                    # grad_masked = grad_sub * solid_ang_mask

                    # Number of voxels analyzed
                    n = solid_ang_mask.sum()

                    # Set pixel in plane as the mean of the voxels analyzed
                    img_2d[j, i] = np.nan_to_num(img_masked.sum() / n)
                    img_grad_2d[j, i] = np.nan_to_num(grad_masked.sum() / n)

                elapsed = time() - ti
                print('[ SA ] Scale: %d %s Ang: %s | Point (%d, %d) of (360/180) | Value: %f | Time: %.2f' %
                    (n_scale + 1, scale, (x_angle, z_angle), j, i, img_2d[j, i], elapsed))

            # Create results:
            # 2 png files / 2 raw files

            # PNG output for intensities
            img_filename = join(subject_output_dir, 'intensity_%d_to_%d_solid_angle_to_sphere' % (r_min, r_max))
            plt.imsave(img_filename + '.png', img_2d, cmap='gray')
            img_2d.tofile(img_filename + '.raw')

            # RAW output for gradients
            grad_filename = join(subject_output_dir, 'gradient_%d_to_%d_solid_angle_to_sphere' % (r_min, r_max))
            plt.imsave(grad_filename + '.png', img_grad_2d, cmap='gray')
            img_grad_2d.tofile(grad_filename + '.raw')
    else:
        print('[  ERROR  ] File {} was not found'.format(brainmask_file))
    
    exit()


if __name__ == '__main__':
    # Load database
    params = load_params()
    dataset_csv_file = os.path.normpath(root + params['data_file'])

    # Load the list of subjects and order by subject_id: df
    df = pd.read_csv(dataset_csv_file)
    df = df.sort_values('folder')

    # Parse configuration
    cfg = join(root, 'config', 'config.cfg')
    config = ConfigParser()
    config.read(cfg)

    dataset_registered_folder = config.get('dirs', 'dataset_folder_registered')
    results_folder = config.get('dirs', 'sphere_mapping')
    n_cores = config.getint('resources', 'n_cores')
    mni_file = join(root, 'param', 'FSL_MNI152_FreeSurferConformed_1mm.nii')

    # Get template centroid
    mni_aseg = nb.load(mni_file).get_data()
    centroid = tuple(get_centroid(mni_aseg > 0))
    print('[  OK  ] Centroid = {}'.format(centroid))

    # Pool the process
    pool = Pool(n_cores)
    pool.map(process_image, df['folder'])
    pool.close()
    pool.join()
