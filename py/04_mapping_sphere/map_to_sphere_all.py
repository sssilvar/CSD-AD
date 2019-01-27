import os
import sys
from time import time
from configparser import ConfigParser
from os.path import join, dirname, realpath, isfile

from multiprocessing import Pool
from functools import partial
from contextlib import contextmanager

import numpy as np
import pandas as pd
import nibabel as nb
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

# Set root folder
root = dirname(dirname(dirname(realpath(__file__))))

# Append path to add lib
sys.path.append(root)
from lib.param import load_params
from lib.masks import solid_cone
from lib.visualization import show_mri
from lib.transformations import rotate_vol
from lib.geometry import extract_sub_volume, get_centroid


@contextmanager
def poolcontext(*args, **kwargs):
    pool = Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def process_image(folder, n_scale, scale):
    # Set of folders important in the processing pipeline
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

        # Extract edges with sobel
        sobel_mode = 'reflect'
        sobel_x = ndi.sobel(img, axis=0, mode=sobel_mode)
        sobel_y = ndi.sobel(img, axis=1, mode=sobel_mode)
        sobel_z = ndi.sobel(img, axis=2, mode=sobel_mode)
        img_sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2 + sobel_z ** 2)

        # Crete a solid angle from a scale: sa
        r_min, r_max = scale
        sa = solid_cone(radius=(r_min, r_max), center=centroid)

        # Start go over the whole sphere (x_angle: [0, pi] and z_angle [-pi, pi])
        ns = 2  # TODO: Chack if it's neccesary to change it
        img_2d = np.zeros([360 // ns, 180 // ns])
        img_grad_2d = np.zeros_like(img_2d)
        img_sobel_2d = np.zeros_like(img_2d)

        # Extract only relevant volume
        mask_sub, center = extract_sub_volume(sa, radius=(r_min, r_max), centroid=centroid)
        vol_sub, _ = extract_sub_volume(img, radius=(r_min, r_max), centroid=centroid)
        grad_sub, _ = extract_sub_volume(img_grad, radius=(r_min, r_max), centroid=centroid)
        sobel_sub, _ = extract_sub_volume(img_sobel, radius=(r_min, r_max), centroid=centroid)

        for i, z_angle in enumerate(range(-180, 180, ns)):
            ti = time()
            for j, x_angle in enumerate(range(0, 180, ns)):
                solid_ang_mask = rotate_vol(mask_sub, angles=(x_angle, 0, z_angle))
                img_masked = np.multiply(vol_sub, solid_ang_mask)
                grad_masked = np.multiply(grad_sub, solid_ang_mask)
                sobel_masked = np.multiply(sobel_sub, solid_ang_mask)

                # Number of voxels analyzed
                n = solid_ang_mask.sum()

                # Set pixel in plane as the mean of the voxels analyzed
                img_2d[i, j] = img_masked.sum() / n
                img_grad_2d[i, j] = grad_masked.sum() / n
                img_sobel_2d[i, j] = sobel_masked.sum() / n

            elapsed = time() - ti
            print('[ SA ] Scale: %d %s Ang: %s | Point (%d, %d) of (360/180) | Value: %f | Time: %.2f' %
                (n_scale + 1, scale, (x_angle, z_angle), i, j, img_2d[i, j], elapsed))

        # Create results:
        # 2 png files / 2 raw files

        # Image output for intensities
        img_filename = join(subject_output_dir, 'intensity_%d_to_%d_solid_angle_to_sphere' % (r_min, r_max))
        plt.imsave(img_filename + '.png', img_2d, cmap='gray')
        img_2d.tofile(img_filename + '.raw')

        # Image output for gradients
        grad_filename = join(subject_output_dir, 'gradient_%d_to_%d_solid_angle_to_sphere' % (r_min, r_max))
        plt.imsave(grad_filename + '.png', img_grad_2d, cmap='gray')
        img_grad_2d.tofile(grad_filename + '.raw')

        # Image output for sobel
        sobel_filename = join(subject_output_dir, 'sobel_%d_to_%d_solid_angle_to_sphere' % (r_min, r_max))
        plt.imsave(sobel_filename + '.png', img_sobel_2d, cmap='gray')
        img_sobel_2d.tofile(grad_filename + '.raw')
    else:
        print('[  ERROR  ] File {} was not found'.format(brainmask_file))


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
    print('[  INFO  ] Number of cores used: {}'.format(n_cores))

    # Start processing the whole dataset
    scales = [
        (0, 25),
        (25, 50),
        (50, 75),
        (75, 100),
    ]
    for n_scale, scale in enumerate(scales):
        # Pool the process
        with poolcontext(processes=n_cores) as pool:
            pool.map(partial(process_image, n_scale=n_scale, scale=scale), df['folder'])
    # pool = Pool(n_cores)
    # pool.map(process_image, df['folder'])
    # pool.close()
    # pool.join()
