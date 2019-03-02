__author__ = "Santiago Silva"
__copyright__ = "Copyright 2018"

import os
from multiprocessing import Pool
from configparser import ConfigParser
from os.path import dirname, realpath, join, isfile, isdir, basename

import pandas as pd

# Set root folder
root = dirname(dirname(dirname(realpath(__file__))))
current_dir = dirname(realpath(__file__))
script = join(current_dir, 'curvelet_decomposition.py')


def process_subject(sid):
    raw_folder = join(data_folder, sid)  # For some cases
    raw_files = [join(raw_folder, rf) for rf in os.listdir(raw_folder) if rf.endswith('.raw')]

    for raw_file in raw_files:
        cmd = 'python2 {script} -f {raw_file} -sid {sid} -out {out} -s {n_scales} -a {n_angles}'.format(
            script=script,
            raw_file=raw_file,
            sid=sid,
            out=out_folder,
            n_scales=4,
            n_angles=32
        )
        os.system(cmd)


if __name__ == '__main__':
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # Load configuration
    cfg = ConfigParser()
    cfg.read(join(root, 'config', 'config.cfg'))

    # Load parameters
    data_folder = cfg.get('dirs', 'sphere_mapping')
    n_cores = cfg.getint('resources', 'n_cores')
    out_folder = join(data_folder, 'curvelets_non_split')
    group_file = join(data_folder, 'groupfile_last.csv')

    # Load group file
    subjects = pd.read_csv(group_file, index_col=0)

    # Create output folder if does not exist
    if not isdir(out_folder):
        os.mkdir(out_folder)

    # Process subjects
    pool = Pool(n_cores)
    pool.map(process_subject, subjects.index)
    pool.close()

    # Save them in a single dataframe
    img_types = ['intensity', 'gradient', 'sobel']
    for img_type in img_types:
        print('Loading {} features...'.format(img_type))
        df = pd.DataFrame()
        for sid in subjects.index:
            subj_folder = join(out_folder, sid, 'curvelet')
            csv_files = [join(subj_folder, rf) for rf in os.listdir(subj_folder) if rf.endswith('.csv')]

            for csv_file in csv_files:
                if img_type in csv_file:
                    subj_df = pd.read_csv(csv_file, index_col=0)
                    subj_df['sphere'] = '_'.join(basename(csv_file).split('_')[1:4])
                    df = df.append(subj_df)
        # Print some info
        print(df.head())
        print(df.shape)
        df.to_csv(join(out_folder, '{}_curvelet_features_non_split.csv'.format(img_type)))
