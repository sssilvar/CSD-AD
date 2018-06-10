from __future__ import print_function

import os

import numpy as np
import matplotlib.pyplot as plt

import pyct as ct

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

if __name__ == '__main__':
    print('===== Starting Curvelet decomposition test =====\n\n')
    print('[  INFO  ] Root folder in: ', root)

    # Set parameters
    number_of_scales = 4
    number_of_angles = 16

    # Load image
    img_filename = os.path.join(root, 'test', 'test_data', 'sphere_mapped', '30_to_60_solid_angle_to_sphere.raw')
    img = np.reshape(np.fromfile(img_filename), [360, 180]).T

    # Create a Curvelet object
    A = ct.fdct2(img.shape, nbs=number_of_scales, nba=number_of_angles, ac=True, norm=False, vec=True, cpx=False)

    # Apply curvelet to the image
    f = A.fwd(img)

    # Visualize it
    plt.imshow(img, cmap='gray')
    plt.show()
