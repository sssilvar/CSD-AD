import os
import sys
from zipfile import ZipFile
from configparser import ConfigParser
from os.path import join, dirname, realpath, isfile, isdir

from multiprocessing import Pool
from contextlib import contextmanager

import pandas as pd

root = dirname(dirname(dirname(realpath(__file__))))
print('[  ROOT  ] {}'.format(root))

sys.path.append(root)


@contextmanager
def poolcontext(*args, **kwargs):
    pool = Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def register_subject_with_flirt(subject_id):
    """
        Registers a NII image to a MNI152 template
    """
    # Define template volume directory
    dst = os.path.join(root, 'param', 'FSL_MNI152_FreeSurferConformed_1mm.nii')

    # Get subjects directory from conf file
    subjects_dir = cfg.get('dirs', 'subjects_dir')

    # Define useful files for the registration process
    mov = os.path.join(subjects_dir, subject_id, 'mri', 'brainmask.mgz')
    mapmov = os.path.join(registered_folder, subject_id, 'brainmask_reg.nii.gz')
    mov_nii = '/dev/shm/{}.nii.gz'.format(subject_id)
    aff_mat = os.path.join(registered_folder, subject_id, 'transform.mat')

    # Check if file is zipped
    zipped_file = os.path.join(subjects_dir, subject_id + '.zip')

    if not isfile(mov) and isfile(zipped_file) and not isfile(mapmov):
        print('[  INFO  ] Extracting data from: {}'.format(zipped_file))
        bmask_path = join(subject_id, 'mri/brainmask.mgz')

        with ZipFile(zipped_file, 'r') as zf:
            try:
                zf.extract(bmask_path, '/dev/shm/')
            except KeyError as e:
                print('[  ERROR  ] Extraction failed: {}'.format(e))
        subjects_dir = '/dev/shm'

    # Re-define moving image path (if subjects zipped)
    mov = os.path.join(subjects_dir, subject_id, 'mri', 'brainmask.mgz')

    # Check if file exists
    if isfile(mov) and not isfile(mapmov):
        # Create a folder per each subject
        try:
            os.mkdir(os.path.join(registered_folder, subject_id))
        except Exception as e:
            print('[  ERROR  ] Error creating subject_id: {}.'.format(e))

        # Convert to NIFTI
        command = 'mri_convert {} {}'.format(mov, mov_nii)
        os.system(command)

        # Remove extracted if exist
        if isdir('/dev/shm/{}'.format(subject_id)):
            os.system('rm -rf /dev/shm/{}'.format(subject_id))

        # Register to MNI flirt
        # -ref FSL_MNI152_FreeSurferConformed_1mm.nii -in brain.nii.gz
        # -omat my_affine_transf.mat -out registered.nii.gz
        command = 'flirt -ref {} -in {} -omat {} -out {}'.format(dst, mov_nii, aff_mat, mapmov)
        os.system(command)

        # Remove folder from RAM
        os.system('rm {}'.format(mov_nii))
    elif isfile(mapmov):
        print('[  WARNING  ] Registered file was found for {} in: {}'.format(subject_id, mapmov))
    else:
        print('[  ERROR  ] File {} not found'.format(mov))


if __name__ == '__main__':
    # Clear screen
    os.system('clear')

    # Load variables of interest
    print('[  INFO  ] Loading configuration and dataset files...')
    cfg = ConfigParser()
    cfg.read(join(root, 'config/config.cfg'))

    dataset_folder = cfg.get('dirs', 'dataset_folder')
    registered_folder = cfg.get('dirs', 'dataset_folder_registered')
    n_cores = cfg.getint('resources', 'n_cores')

    # Read file containing subject's IDs
    df = pd.read_csv(join(dataset_folder, 'groupfile.csv'), index_col=0)

    with poolcontext(processes=n_cores) as pool:
        pool.map(register_subject_with_flirt, df.index)
