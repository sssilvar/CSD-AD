import os
import sys

import pickle
import pyct as ct
import numpy as np
import nibabel as nb

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(root))

from lib.curvelets import clarray_to_mean_dict

if __name__ == '__main__':
    # Load test image
    # filename = os.path.join(root, 'test', 'test_data', '941_S_1363.mgz')
    # img = nb.load(filename).get_data().astype(np.float)[:, : , 128]
    filename = 'C:/Users/sssilvar/Downloads/Results/results_radial_vid_optimized/002_S_0729/raw/' \
               'gradient_000_to_005_solid_angle_to_sphere.raw'
    img = np.fromfile(filename)

    # Define number of scales and angles
    n_scales = int(sys.argv[1])
    n_angles = int(sys.argv[2])

    # Get a Curvelet decomposition
    A = ct.fdct2(img.shape, nbs=n_scales, nba=n_angles, ac=True, norm=False, vec=True, cpx=False)
    f = A.fwd(img)

    # Convert data to dict
    f_dict = clarray_to_mean_dict(A, f, n_scales, n_angles)

    # Print the dictionary
    for key, val in f_dict.items():
        print('Scale %s: ' % key)
        print('Values:\n\t {}'.format(val))

    file_results = os.path.join(root, 'output', 'curve_dec_test.npy')
    np.save(file_results, f_dict)
    # with open(file_results, 'wb') as fp:
    #     pickle.dump(f_dict, fp)

    script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plot_results_python3.py')
    os.system('python %s %s %d %d' % (script, file_results, n_scales, n_angles))
