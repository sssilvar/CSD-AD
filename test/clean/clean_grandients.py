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
        print('Deleting subject: {}'.format(subject_dir))

        try:
            output_subject = os.path.join(output_folder, subject)
            os.remove(os.path.join(output_subject, 'r.mgz'))
            os.remove(os.path.join(output_subject, 'theta.mgz'))
            os.remove(os.path.join(output_subject, 'phi.mgz'))

        except Exception as e:
            print(e)

