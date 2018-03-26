import os
import sys

import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform

# Set root folder
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Append path to add lib
sys.path.append(root)
from lib.masks import solid_cone
from lib.visualization import show_mri
from lib.transformations import rotate_vol
from lib.geometry import extract_sub_volume, get_centroid

if __name__ == '__main__':
    # Load MRI image and aseg file
    mgz = nb.load(os.path.join(root, 'test', 'test_data', '941_S_1363.mgz'))
    img = mgz.get_data()

    mgz = nb.load(os.path.join(root, 'test', 'test_data', 'mri', 'aseg.mgz'))
    aseg = mgz.get_data()

    # Get volume centroid
    centroid = tuple(get_centroid(aseg > 0))
    print('[  OK  ] Centroid = {}'.format(centroid))

    # Crete a solid angle from a scale: sa
    r_min, r_max = (33, 66)
    sa = solid_cone(radius=(r_min, r_max))

    # Start go over the whole sphere (x_angle: [0, pi] and z_angle [-pi, pi])
    img_2d = np.zeros([360, 180])

    mask_sub, center = extract_sub_volume(sa, radius=(r_min, r_max), centroid=centroid)
    vol_sub, _ = extract_sub_volume(img, radius=(r_min, r_max), centroid=centroid)

    for i, z_angle in enumerate(range(-180, 180)):
        for j, x_angle in enumerate(range(0, 180)):
            solid_ang_mask = rotate_vol(mask_sub, angles=(x_angle, 0, z_angle))
            img_masked = vol_sub * solid_ang_mask

            img_2d[i, j] = img_masked.sum() / solid_ang_mask.sum()
            print('[ SA ] Point (%d, %d) of (360/180)' % (i, j))

    img_filename = os.path.join(root, 'output', '%d_to_%d_solid_angle_to_sphere.png' % (r_min, r_max))
    plt.imsave(img_filename, img_2d, cmap='gray')
    img_2d.tofile(os.path.join(root, 'output', '%d_to_%d_solid_angle_to_sphere.raw' % (r_min, r_max)))