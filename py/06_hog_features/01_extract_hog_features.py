import os
import argparse
from configparser import ConfigParser
from os.path import join, dirname, basename, realpath, isdir

import numpy as np
from skimage.feature import hog

# Get root folder
root = dirname(dirname(dirname(realpath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='HOG feature extractor for mapped images in 2D.')
    parser.add_argument('-groupfile',
                        help='File containing the subjects to process',
                        required=True,
                        type=int)
    return parser.parse_args()


if __name__ == '__main__':
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # Parse arguments
    args = parse_args()
    group_file = args.groupfile

    # Load config file
    cfg = ConfigParser()
    cfg.read(join(root, 'config', 'config.cfg'))

    # Load parameters
    n_cores = cfg.getint('resources', 'n_cores')
    data_folder = cfg.get('dirs', 'sphere_mapping')
    out_folder = join(data_folder, 'HOG')

    # Check if folder does not exist
    if not isdir(out_folder):
        os.mkdir(out_folder)


