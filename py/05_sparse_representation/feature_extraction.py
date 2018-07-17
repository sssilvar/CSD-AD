__author__ = "Santiago Smith"
__copyright__ = "Copyright 2018"
__description__ = """
Curvelet feature extraction. USAGE:
    python3 plsr_analysis.py -s [number_of_scales] -a [number_of_angles]
"""

import os
import sys
import argparse

try:
    import pyct as ct
except ImportError as e:
    print('[  ERROR  ] {}'.format(e))

import numpy as np
import pandas as pd

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(root))

from lib.curvelets import clarray_to_mean_dict
from lib.path import get_file_list, mkdir

def main():
    # Set results folder and csv with subjects
    # results_folder = '/home/sssilvar/Documents/dataset/results_radial_vid_optimized/'
    results_folder = r'C:\Users\sssilvar\Documents\code\data_test'
    output_folder = os.path.join(root, 'output')
    dataset_csv = os.path.join(root, 'param', 'data_df.csv')
    
    # Print some shit (info)
    print('[  INFO  ] ===== CURVELET FEATURE EXTRACTION ====')
    print('\t- Data to be processed at: %s' % results_folder)
    print('\t- Scales: %d: | Angles: %d' % (n_scales, n_angles))
    print('\t- Output folder: %s' % output_folder)

    # Load dataset and create output folder
    df = pd.read_csv(dataset_csv)
    mkdir(output_folder)

    # Start feature extraction
    columns = []

    for subject in df['folder']:
        # Set filename(s)
        raw_folder = os.path.join(results_folder, subject, 'raw')
        
        for r in sphere_radius:
            # Get type of image and sphere params
            raw_file = os.path.join(
                raw_folder,
                '%s_%03d_to_%03d_solid_angle_to_sphere.raw' % (
                    img_type, r, (r + step)
                ))
            
            # Load and do the magic!
            img = np.fromfile(raw_file, dtype=np.float).reshape([360, 180]).T
            
            # Get a Curvelet decomposition
            A = ct.fdct2(
                img.shape, 
                nbs=n_scales, 
                nba=n_angles, 
                ac=True, 
                norm=False, 
                vec=True, 
                cpx=False)
            f = A.fwd(img)
    
    # Save results
    df.to_hdf(os.path.join(output_folder, 'test.h5'), key='features', mode='w')

if __name__ == '__main__':
    # --- ARG PARSING ---
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('-s', metavar='--scales',
                        help='Number of scales.',
                        default=6)
    parser.add_argument('-a', metavar='--angles',
                        help='Number of angles (subbands) per scale.',
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

    step = 5
    sphere_radius = [i for i in range(0, 21, step)]

    # Start main
    main()