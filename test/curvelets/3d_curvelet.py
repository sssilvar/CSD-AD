from __future__ import print_function
import os

import pyct as ct
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def print_cl_info(f, scale, angle):
    try:
        data = f(scale, angle)
        print('Scale %d / angle %d: ' % (scale, angle),
              '\t Shape: ', np.shape(data))
    except IndexError:
        print('[  ERROR  ] Index (%d, %d) out of range' % (scale, angle))


if __name__ == '__main__':
    print('Loading image...')
    filename = os.path.join(root, 'param', 'fsaverage', 'brainmask.mgz')
    img = nb.load(filename).get_data().astype(np.float)
    img = img[50:150, 50:150, 50:150]
    print('Image shape: ', img.shape)
    print('Number of elements: ', np.prod(img.shape))

    # Parameters
    number_of_scales = 7
    number_of_angles = 4

    # Setup curvelet params
    print('Applying Curvelet Transform...')
    A = ct.fdct3(img.shape, nbs=number_of_scales, nba=number_of_angles, ac=True, norm=False, vec=True, cpx=False)

    # Apply curvelet to the image
    f = A.fwd(img)

    print('Coefficients array:\n\t', f)
    print('Shape: ', f.shape)

    # Print Information
    for scale in range(0, number_of_scales):
        if scale == 0:
            angles = [0]
        elif scale == 1:
            angles = range(0, number_of_angles)
        elif scale % 2 == 0:
            angles = range(0, int(scale * number_of_angles))
        elif scale % 2 != 0:
            angles = range(0, int((scale - 1) * number_of_angles))
        else:
            angles = []
            raise ValueError('There is no angles inside the scale')

        # Go over all the angles in the scale
        for angle in angles:
            print_cl_info(f, scale, angle)

    # Reconstruct the image
    print('Applying Curvelet Transform...')
    y = A.inv(f)

    print('Plotting results')
    fig, ax = plt.subplots(1, 2, sharey='all')
    ax[0].imshow(img[:, :, 50], cmap='gray')
    ax[0].set_title('Original')

    ax[1].imshow(y[:, :, 50], cmap='gray')
    ax[1].set_title('Reconstructed')
    plt.show()