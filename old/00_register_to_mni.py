import os
import sys
from zipfile import ZipFile
from configparser import ConfigParser
from os.path import join, dirname, realpath, isfile, isdir

from multiprocessing import Pool
from contextlib import contextmanager

import numpy as np
import pandas as pd
import nibabel as nb

root = dirname(dirname(realpath(__file__)))
robex = join(os.getenv('HOME'), 'Apps', 'ROBEX', 'runROBEX.sh')
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
    mni = os.path.join(root, 'param', 'FSL_MNI152_FreeSurferConformed_1mm.nii')
    # dst = r'/home/ssilvari/Downloads/MNI152_T1_1mm_Brain_c.nii.gz'

    # Get subjects directory from conf file
    subjects_dir = cfg.get('dirs', 'subjects_dir')

    # Define useful files for the registration process
    mov = os.path.join(subjects_dir, subject_id, 'mri', 'orig', '001.mgz')
    mapmov = os.path.join(registered_folder, subject_id, '001_reg.nii.gz')
    aff_mat = os.path.join(registered_folder, subject_id, 'transform.mat')

    # Check if file is zipped
    zipped_file = os.path.join(subjects_dir, subject_id + '.zip')

    if not isfile(mov) and isfile(zipped_file) and not isfile(mapmov):
        print('[  INFO  ] Extracting data from: {}'.format(zipped_file))
        brain_path = join(subject_id, 'mri/orig/001.mgz')

        with ZipFile(zipped_file, 'r') as zf:
            try:
                zf.extract(brain_path, '/dev/shm/')
            except KeyError as e:
                print('[  ERROR  ] Extraction failed: {}'.format(e))
        subjects_dir = '/dev/shm'

    # Re-define moving image path (if subjects zipped)
    mov = os.path.join(subjects_dir, subject_id, 'mri', 'orig', '001.mgz')
    mov_nii = os.path.join(subjects_dir, subject_id, 'mri', 'orig', '001.nii.gz')
    mov_nii_stripped = os.path.join(subjects_dir, subject_id, 'mri', 'orig', '001_stripped.nii.gz')

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

        # Skull stripping
        if not isfile(mov_nii_stripped):
            command = 'bash {robex} {in_file} {out}'.format(robex=robex, in_file=mov_nii, out=mov_nii_stripped)
            os.system(command)

        # Register to MNI flirt
        if not isfile(mapmov):
            # command = 'fsl_rigid_register -r {ref} -i {i} -o {out}'.format(ref=mni, i=mov_nii_stripped, out=mapmov)
            command = 'flirt -ref {ref} -in {i} -out {out} ' \
                      '-omat {omat}'.format(ref=mni, i=mov_nii_stripped, out=mapmov, omat=aff_mat)
            print(command)
            os.system(command)

        # Remove extracted if exist
        if isdir('/dev/shm/{}'.format(subject_id)):
            os.system('rm -rf /dev/shm/{}'.format(subject_id))
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

    dataset_folder = cfg.get('dirs', 'subjects_dir')
    registered_folder = cfg.get('dirs', 'dataset_folder_registered')
    n_cores = cfg.getint('resources', 'n_cores')

    # Read file containing subject's IDs
    df = pd.read_csv(join(dataset_folder, 'groupfile.csv'), index_col=0)

    with poolcontext(processes=n_cores) as pool:
        pool.map(register_subject_with_flirt, df.index)
