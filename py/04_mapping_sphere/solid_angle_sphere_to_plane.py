import os
import sys

import numpy as np
import pandas as pd
import nibabel as nb
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform

# Set root folder
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Append path to add lib
sys.path.append(root)
from lib.param import load_params
from lib.masks import solid_cone
from lib.visualization import show_mri
from lib.transformations import rotate_vol
from lib.geometry import extract_sub_volume, get_centroid


if __name__ == '__main__':
    # Load database
    params = load_params()
    dataset_csv_file = os.path.normpath(root + params['data_file'])
    dataset_registered_folder = '/home/jullygh/sssilvar/Documents/workdir/'
    results_folder = '/home/jullygh/sssilvar/Documents/results'

    # Load the list of subjects and order by subject_id: df
    df = pd.read_csv(dataset_csv_file)
    df = df.sort_values('folder')

    # Start processing the whole dataset
    for folder in df['folder']:
        # Set of folders important in the processing pipeline
        subject_dir = os.path.join(dataset_registered_folder, folder)
        brainmask_file = os.path.join(subject_dir, 'brainmask_reg.mgz')
        aseg_file = os.path.join(subject_dir, 'aseg.mgz')
        subject_output_dir = os.path.join(results_folder, folder)

        # Try creating a folder for each subject
        try:
            os.mkdir(subject_output_dir)
            print('[  OK  ] Folder created at: ' + subject_output_dir)
        except OSError:
            print('[  WARNING  ] Folder already exists.')

        # # Load MRI image and aseg file
        # mgz = nb.load(brainmask_file)
        # img = mgz.get_data().astype(np.float)
        #
        # # Calculate the gradient: img_grad
        # gx, gy, gz = np.gradient(img)
        # img_grad = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        #
        # # Load aseg.mgz file: aseg
        # mgz = nb.load(aseg_file)
        # aseg = mgz.get_data()
        #
        # # Get volume centroid
        # centroid = tuple(get_centroid(aseg > 0))
        # print('[  OK  ] Centroid = {}'.format(centroid))
        #
        # scales = [
        #     (0, 25),
        #     (25, 50),
        #     (50, 75),
        #     (75, 100),
        # ]
        #
        # for n_scale, scale in enumerate(scales):
        #     # Crete a solid angle from a scale: sa
        #     r_min, r_max = scale
        #     sa = solid_cone(radius=(r_min, r_max), center=centroid)
        #
        #     # Start go over the whole sphere (x_angle: [0, pi] and z_angle [-pi, pi])
        #     img_2d = np.zeros([360, 180])
        #
        #     mask_sub, center = extract_sub_volume(sa, radius=(r_min, r_max), centroid=centroid)
        #     vol_sub, _ = extract_sub_volume(img, radius=(r_min, r_max), centroid=centroid)
        #
        #     for i, z_angle in enumerate(range(-180, 180)):
        #         for j, x_angle in enumerate(range(0, 180)):
        #             solid_ang_mask = rotate_vol(mask_sub, angles=(x_angle, 0, z_angle))
        #             img_masked = vol_sub * solid_ang_mask
        #
        #             img_2d[i, j] = np.nan_to_num(img_masked.sum() / solid_ang_mask.sum())
        #         print('[ SA ] Scale: %d %s Ang: %s | Point (%d, %d) of (360/180) | Value: %f' %
        #               (n_scale + 1, scale, (x_angle, z_angle), i, j, img_2d[i, j]))
        #
        #     img_filename = os.path.join(root, 'output', 'gradient_%d_to_%d_solid_angle_to_sphere.png' % (r_min, r_max))
        #     plt.imsave(img_filename, img_2d, cmap='gray')
        #     img_2d.tofile(os.path.join(root, 'output', 'gradient_%d_to_%d_solid_angle_to_sphere.raw' % (r_min, r_max)))
