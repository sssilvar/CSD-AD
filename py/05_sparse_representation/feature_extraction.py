__author__ = "Santiago Silva"
__copyright__ = "Copyright 2018"
__description__ = """
Curvelet feature extraction. USAGE:
    python3 plsr_analysis.py -s [number_of_scales] -a [number_of_angles]
"""

import os
import sys
import argparse
from tqdm import tqdm

try:
    import pyct as ct
except ImportError as e:
    print('[  ERROR  ] {}'.format(e))

import numpy as np
import pandas as pd

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(root))

from lib.curvelets import clarray_to_gen_gaussian_dict
from lib.path import get_file_list, mkdir

def main():
    # Set results folder and csv with subjects
    dataset_csv = os.path.join(root, 'param', 'data_df.csv')
    
    # Print some shit (info)
    print('[  INFO  ] ===== CURVELET FEATURE EXTRACTION ====')
    print('\t- Data to be processed at: %s' % results_folder)
    print('\t- Scales: %d: | Angles: %d' % (n_scales, n_angles))
    print('\t- Output folder: %s' % output_folder)

    # Load dataset and create output folder
    df = pd.read_csv(dataset_csv)
    mkdir(output_folder)

    up_to = 203
    for i, (subject, label) in enumerate(zip(df['folder'][:up_to], df['target'][:up_to])):
        print('Processing subject ' + subject)
        
        # Set filename(s)
        raw_folder = os.path.join(results_folder, subject, 'raw')
        
        # Initialize a feature dictionary per subject
        f_dict = {}
        f_dict['subject'] = subject
        f_dict['target'] = label
        
        for r in tqdm(sphere_radius, desc='Sphere scale'):
            # Get type of image and sphere params
            raw_file = os.path.join(
                raw_folder,
                '%s_%03d_to_%03d_solid_angle_to_sphere.raw' % (
                    img_type, r, (r + delta)
                ))
            
            # Load and do the magic!
            try:
                img = np.fromfile(raw_file, dtype=np.float).reshape([360, 180]).T
            except:
                print('No file found: ' + raw_file)
            
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

            # Convert data to dict
            buff = clarray_to_gen_gaussian_dict(A, f, n_scales, n_angles, r)
            f_dict.update(buff)

        if i is 0:
            df_features = pd.DataFrame(f_dict, index=[0])
        else:
            df_subject = pd.DataFrame(f_dict, index=[0])
            df_features = df_features.append(df_subject)
    
    # Save results
    df_features.to_hdf(filename_features, key='features', mode='w')
    os.system('chmod 766 ' + filename_features)


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
    output_folder = os.path.join(root, 'output')
    
    filename_features =os.path.join(output_folder, 'spherical_curvelet_features_nscales_%d_nangles_%d.h5' % (n_scales, n_angles))
    # results_folder = '/home/sssilvar/Documents/dataset/results_radial_vid_optimized/'

    os.system('clear')
    print('======= CURVELET FEATURE EXTRACTION =======')
    print('\n\t- N. Scales: %d' % n_scales)
    print('\n\t- N. Angles: %d' % n_angles)
    print('\n\t- Feats. File: %s' % filename_features)

    step = 1 # Propagation step
    delta = 5 # Sphere thickness
    sphere_radius = [i for i in range(0, 95, step)]

    # Start main
    main()