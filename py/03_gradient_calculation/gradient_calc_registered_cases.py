import os
import sys

import pandas as pd
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.append(root)
from lib.param import load_params


def save_mgz(x, filename):
    # Create a folder
    try:
        os.mkdir(output_subject)
    except Exception as e:
        print('[  WARNING  ] Folder already exists')

    x_mgz = nb.Nifti1Image(x, affine=np.eye(4))
    nb.save(x_mgz, filename)


if __name__ == '__main__':
    # Load Params
    params = load_params()

    workspace = '/home/jullygh/sssilvar/Documents/workdir'
    output_folder = '/home/jullygh/sssilvar/Documents/workdir'

    # Load dataset file
    df = pd.read_csv(os.path.normpath(root + params['data_file']))

    for subject in df['folder']:
        # Set subject file path
        subject_dir = os.path.join(workspace, subject, 'brainmask_reg.mgz')
        print('Processing subject: {}'.format(subject_dir))

        try:
            mgz = nb.load(subject_dir)
            img = mgz.get_data().astype(np.float32)

            # Calculate gradient
            gx, gy, gz = np.gradient(img)
            r, theta, phi = (np.sqrt(gx ** 2 + gy ** 2 + gz ** 2),
                             np.arctan2(gy, gx),
                             np.arctan2(np.sqrt(gx ** 2 + gy ** 2), gz))

            output_subject = os.path.join(output_folder, subject)
            print(output_subject)

            save_mgz(r, os.path.join(output_subject, 'r.mgz'))
            save_mgz(theta, os.path.join(output_subject, 'theta.mgz'))
            save_mgz(phi, os.path.join(output_subject, 'phi.mgz'))

        except Exception as e:
            print(e)

