__author__ = "Santiago Silva"
__copyright__ = "Copyright 2018"
__description__ = """
Curvelet feature extraction. USAGE:
    python3 plsr_analysis.py -s [number_of_scales] -a [number_of_angles]
"""

import os
import sys
import argparse
from os.path import join, dirname, realpath

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import pandas as pd

current_dir = dirname(realpath(__file__))
root = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(join(root))

from lib.path import mkdir


def main():
    # Set results folder and csv with subjects
    dataset_csv = join(root, 'param', 'data_df.csv')

    # Load dataset and create output folder
    df = pd.read_csv(dataset_csv)
    mkdir(output_folder)

    # Execute command
    script = join(current_dir, 'feature_extraction_individual.py')

    up_to = 203
    # up_to = 2 # TEST
    for (subject, label) in zip(df['folder'][:up_to], df['dx_group'][:up_to]):
        cmd = 'python2.7 ' + script + \
            ' -f ' + results_folder + \
            ' -s ' + str(n_scales) + \
            ' -a ' + str(n_angles) + \
            ' -subject ' + subject + \
            ' -label ' + label
        os.system(cmd)

if __name__ == '__main__':
    # --- ARG PARSING ---
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('-f', metavar='--folder',
                        help='Subjects\' folder.',
                        default='/user/ssilvari/home/Documents/temp/sphere_mapped')
    parser.add_argument('-s', metavar='--scales',
                        help='Number of scales.',
                        type=int,
                        default=6)
    parser.add_argument('-a', metavar='--angles',
                        help='Number of angles (subbands) per scale.',
                        type=int,
                        default=8)
    parser.add_argument('-t', metavar='--type',
                        help='type of image (intensity, gradient)',
                        default='gradient')
    args = parser.parse_args()
    
    # --- MAIN ---
    # Set parameters
    n_scales = args.s
    n_angles = args.a
    img_type = args.t
    results_folder = args.f
    output_folder = join(root, 'output')
    output_subfolder = join(output_folder, 'curv_feats_%s_nscales_%d_nangles_%d' % (img_type, n_scales, n_angles))
    
    try:
        os.mkdir(output_subfolder)
    except OSError:
        pass
    
    filename_features =join(output_folder, 'spherical_curvelet_features_nscales_%d_nangles_%d.h5' % (n_scales, n_angles))
    # results_folder = '/home/sssilvar/Documents/dataset/results_radial_vid_optimized/'

    os.system('clear')
    print('======= CURVELET FEATURE EXTRACTION =======')
    print('\n\t- N. Scales: %d' % n_scales)
    print('\n\t- N. Angles: %d' % n_angles)
    print('\n\t- Feats. Folder: %s' % output_subfolder)

    step = 1 # Propagation step
    delta = 5 # Sphere thickness
    sphere_radius = [i for i in range(0, 95, step)]

    # Start main
    main()