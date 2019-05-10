#!/bin/env python3
import os
import glob
from os.path import join, realpath, dirname, basename, splitext
from configparser import ConfigParser

import pandas as pd

# Set root folder
root = dirname(dirname(dirname(realpath(__file__))))

if __name__ == '__main__':
    # Load configuration file
    cfg_file = join(root, 'config', 'config.cfg')
    cfg = ConfigParser()
    cfg.read(cfg_file)

    # Get subjects folder
    mapped_subjects_dir = cfg.get('dirs', 'sphere_mapping')
    print('\t- Mapped subjects folder: {}'.format(mapped_subjects_dir))

    # Find NPZ files
    raw_files = glob.glob(join(mapped_subjects_dir, '**', '*.npz'), recursive=True)

    # Create features matrix
    features = pd.DataFrame(columns=['type', 'scale'])

    for f in raw_files:
        subject_id = f.split('/')[-2]
        file_basename = splitext(basename(f))[0].split('_')
        scale = '_'.join(file_basename[-2:])
        img_type = file_basename[0]

        features.loc[subject_id, ['type', 'scale']] = [img_type, scale]

    print(features.head())

