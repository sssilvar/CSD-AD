import os
import sys

import pickle
import pyct as ct
import numpy as np
import nibabel as nb

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(root))

from lib.curvelets import clarray_to_mean_dict
from lib.path import get_file_list, mkdir

if __name__ == '__main__':
    # Set results folder and subjects to be processed
    results_folder = '/home/sssilvar/Documents/dataset/results_radial_vid_optimized/'

    subjects = [
        # '002_S_0729',
        '002_S_1155'
    ]

    # Get filenames
    for subject in subjects:
        filenames = get_file_list(os.path.join(results_folder, subject, 'raw'), ext='.raw')
        filenames = [filenames[0]]

        # Create folder for curvelet data
        output_folder = os.path.join(results_folder, subject, 'curvelet')
        print('[  OK  ] Saving results in: ' + output_folder)
        mkdir(output_folder)

        for filename in filenames:
            filename_path = os.path.join(results_folder, subject, 'raw', filename)
            print('Processing: {}'.format(filename))

            file_output = os.path.join(output_folder, filename[:-4] + '.npy')
            print(file_output)
            img = np.fromfile(filename, dtype=np.float).reshape([360, 180]).T

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

            np.save(file_output, f_dict)

        #
        #     script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plot_results_python3.py')
        #     os.system('python %s %s %d %d' % (script, file_results, n_scales, n_angles))
