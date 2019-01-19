import os
import sys

import pandas as pd

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print('[  ROOT  ] {}'.format(root))

sys.path.append(root)
from lib.param import load_params

if __name__ == '__main__':
    os.system('clear')
    params = load_params()
    df = pd.read_csv(os.path.normpath(root + params['data_file']))
    df = df.sort_values(by=['folder'])

    # Load dataset folder
    dataset_folder = '~/Documents/dataset/FreeSurferSD/'
    # dataset_folder = r"/run/media/ssilvari/HDD Data/Universidad/MSc/Thesis/Dataset/FreeSurferSD"

    # Set template to register to
    dst = os.path.join(root, 'param', 'FSL_MNI152_FreeSurferConformed_1mm.nii')

    # Set an work directory
    # workspace = r"/run/media/ssilvari/HDD Data/Universidad/MSc/Thesis/Dataset/registered"
    workspace = os.environ['HOME'] + '/Documents/dataset/FreeSurferSD_to_MNI'

    for folder in df['folder'][:5]:
        mov = os.path.join(dataset_folder, folder, 'mri', 'brainmask.mgz')
        mov_nii = '/dev/shm/brainmask.nii.gz'
        aff_mat = os.path.join(workspace, folder, 'transform.mat')
        mapmov = os.path.join(workspace, folder, 'brainmask_reg.nii.gz')

        # Create a folder per each subject
        try:
            os.mkdir(os.path.join(workspace, folder))
        except Exception as e:
            print('[  ERROR  ] Error creating folder: {}.'.format(e))

        # Convert to NIFTI
        command = 'mri_convert {} {}'.format(mov, mov_nii)
        os.system(command)

        # Register to MNI
        # flirt -ref /home/sssilvar/Documents/code/CSD-AD/param/FSL_MNI152_FreeSurferConformed_1mm.nii -in brain.nii.gz -omat my_affine_transf.mat -out registered.nii.gz
        command = 'flirt -ref {} -in {} -omat {} -out {}'.format(dst, mov_nii, aff_mat, mapmov)
        os.system(command)

        os.system('rm {}'.format(mov_nii))
