import os
import sys

import pyct as ct
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(root))

from lib.curvelets import curvelet_plot, clarray_to_mean_dict

if __name__ == '__main__':
    # Load test image
    filename = os.path.join(root, 'test', 'test_data', '941_S_1363.mgz')
    img = nb.load(filename).get_data().astype(np.float)[:, : , 128]

    # Define number of scales and angles
    n_scales = int(sys.argv[1])
    n_angles = int(sys.argv[2])

    # Get a Curvelet decomposition
    A = ct.fdct2(img.shape, nbs=n_scales, nba=n_angles, ac=True, norm=False, vec=True, cpx=False)
    f = A.fwd(img)
    f = np.abs(f) / np.max(np.abs(f))

    # Convert data to dict
    f_dict = clarray_to_mean_dict(A, f, n_scales, n_angles)

    # Print the dictionary
    for key, val in f_dict.items():
        print('Scale %s: ' % key)
        print('Values:\n\t {}'.format(val))

    curvelet_plot(n_scales, n_angles, f_dict)
    plt.show()
