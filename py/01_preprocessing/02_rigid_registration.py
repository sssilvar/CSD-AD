import os
import sys

import pandas as pd

root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

sys.path.append(root)
from lib.param import load_params

if __name__ == '__main__':
    params = load_params()
    df = pd.read_csv(os.path.normpath(root + params['data_file']))
    df = df.sort_values(by=['folder'])

    # Load dataset folder
    dataset_folder = params['dataset_folder']

    # Set template to register to
    dst = os.path.join(root, 'param', 'fsaverage.mgz')

    # Set an work directory
    workspace = '/home/jullygh/sssilvar/Documents/workdir'

    for folder in df['folder']:
        mov = os.path.join(dataset_folder, folder, 'mri', 'brainmask.mgz')
        lta = os.path.join(workspace, folder, 'transform.lta')
        mapmov = os.path.join(workspace, folder, 'brainmask_reg.mgz')

        # Create a folder per each subject
        # os.mkdir(os.path.join(workspace, folder))

        # Build command
        command = 'mri_robust_register --mov %s --dst %s --lta %s --mapmov %s --satit' \
                  % (mov, dst, lta, mapmov)
        print(command)
