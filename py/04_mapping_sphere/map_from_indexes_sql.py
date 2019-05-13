#!/bin/env python3
import os
from configparser import ConfigParser
from multiprocessing import Pool
from os.path import join, dirname, realpath, isdir

import numpy as np
import pandas as pd
import nibabel as nb
import scipy.ndimage as ndi
from sqlalchemy import create_engine

import matplotlib.pyplot as plt

root = dirname(dirname(dirname(realpath(__file__))))


def mkdir(path):
    if not isdir(path):
        os.mkdir(path)


def sobel_magnitude(nii_file):
    # Load file
    vol = nb.load(nii_file).get_data().astype(np.float)
    sobel_mode = 'reflect'
    sobel_x = ndi.sobel(vol, axis=0, mode=sobel_mode)
    sobel_y = ndi.sobel(vol, axis=1, mode=sobel_mode)
    sobel_z = ndi.sobel(vol, axis=2, mode=sobel_mode)
    return np.sqrt(sobel_x ** 2 + sobel_y ** 2 + sobel_z ** 2)


def subject_mapping(subject):
    print(f'Processing subject {subject}')
    subject_out_dir = join(out_folder, subject)
    subject_dir = join(data_folder, subject)
    mkdir(subject_out_dir)

    # Load volume
    vol_sobel = sobel_magnitude(join(subject_dir, subject_file))
    for scale in scales:
        scale_name = '%d_%d' % scale
        img = np.zeros([360 // ns, 180 // ns])

        # Create dataframe per scale
        res_sql = engine.execute(f'SELECT * FROM indexes WHERE scale=\'{scale_name}\'')
        df = pd.DataFrame(res_sql.fetchall())
        df.columns = res_sql.keys()

        for i, theta_i in enumerate(range(-180, 180, ns)):
            print('Processing scale: {} | X = {} ...'.format(scale_name, theta_i))
            ixs_theta = df.query(f'theta == {theta_i}')
            for j, phi_j in enumerate(range(0, 180, ns)):
                ixs = ixs_theta.query(f'phi == {phi_j}')[['ix', 'iy', 'iz']]
                ix, iy, iz = ixs['ix'].values[0], ixs['iy'].values[0], ixs['iz'].values[0]
                ix_where = np.frombuffer(ix, dtype=int), np.frombuffer(iy, dtype=int), np.frombuffer(iz, dtype=int)

                # Map result
                img[i, j] = vol_sobel[ix_where].mean()
        # Save mapped images
        file_basename = join(subject_out_dir, f'{scale_name}')
        plt.imsave(file_basename + '.png', img, cmap='gray')
        np.savez_compressed(file_basename, img=img, indexes=sql_file, brain_file=subject_file)


if __name__ == '__main__':
    # Setup params
    tk = 25
    overlap = 9
    max_radius = 100
    ns = 6  # Overlap
    n_spheres = max_radius // (tk - overlap)
    scales = [(i * (tk - overlap), ((i + 1) * tk) - (i * overlap)) for i in range(n_spheres)]

    sql_file = f'/dev/shm/indexes_tk_{tk}_overlap_{overlap}_ns_{ns}.sqlite'
    subject_file = 'brainmask_reg.nii.gz'
    cfg_file = join(root, 'config', 'config.cfg')
    print(f'SQL File: {sql_file}')
    print(f'Configuration File: {cfg_file}')

    # Load configuration
    cfg = ConfigParser()
    cfg.read(cfg_file)
    data_folder = cfg.get('dirs', 'dataset_folder_registered')
    group_file = join(data_folder, 'groupfile.csv')
    out_folder = join(cfg.get('dirs', 'sphere_mapping'), f'ADNI_FS_mapped_tk_{tk}_overlap_{overlap}_ns_{ns}')
    n_cores = cfg.getint('resources', 'n_cores')
    print(f'Registered subjects dir: {data_folder}')
    print(f'Mapped subjects dir: {out_folder}')
    print(f'Group file: {group_file}')

    # Create SQL engine
    engine = create_engine(f'sqlite:///{sql_file}')
    # Create folder if does not exist
    mkdir(out_folder)

    # Process subjects in parallel
    subjects = pd.read_csv(group_file, index_col=0)
    pool = Pool(n_cores)
    pool.map(subject_mapping, subjects.index)


