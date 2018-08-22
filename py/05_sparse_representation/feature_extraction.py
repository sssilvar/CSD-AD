__author__ = "Santiago Silva"
__copyright__ = "Copyright 2018"
__description__ = """
Curvelet feature extraction. USAGE:
    python3 plsr_analysis.py -s [number_of_scales] -a [number_of_angles]
"""

import os
import gc
import sys
import argparse
from tqdm import tqdm
from os.path import join, dirname, realpath

try:
    import pyct as ct
except ImportError as e:
    print('[  ERROR  ] {}'.format(e))

import numpy as np
import pandas as pd

root = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(join(root))

from lib.curvelets import clarray_to_gen_gaussian_dict
from lib.path import get_file_list, mkdir


def main():
    # Set results folder and csv with subjects
    dataset_csv = join(root, 'param', 'data_df.csv')
    
    # Print some shit (info)
    print('[  INFO  ] ===== CURVELET FEATURE EXTRACTION ====')
    print('\t- Data to be processed at: %s' % results_folder)
    print('\t- Scales: %d: | Angles: %d' % (n_scales, n_angles))
    print('\t- Output folder: %s' % output_folder)

    # Load dataset and create output folder
    df = pd.read_csv(dataset_csv)
    mkdir(output_folder)

    # Create Curvelet object for 360x180 px
    A = ct.fdct2(
        (360,180), 
        nbs=n_scales, 
        nba=n_angles, 
        ac=True, 
        norm=False, 
        vec=True, 
        cpx=False)


    up_to = 203
    for i, (subject, label) in enumerate(zip(df['folder'][:up_to], df['target'][:up_to])):
        print('Processing subject ' + subject)
        
        # Set filename(s)
        raw_folder = join(results_folder, subject, 'raw')
        
        # Initialize a feature dictionary per subject
        f_dict = {}
        f_dict['subject'] = subject
        f_dict['target'] = label
        f_dict['n_scales'] = n_scales
        f_dict['n_angles'] = n_angles
        
        for r in tqdm(sphere_radius, desc='Sphere scale'):
        # for r in sphere_radius:
            # Get type of image and sphere params
            raw_file = join(
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
            f = A.fwd(img)

            # Convert data to dict
            # buff = clarray_to_gen_gaussian_dict(A, f, n_scales, n_angles, r)
            # f_dict.update(buff)
            del buff, f, img
            
            # Set RAM Free
            gc.collect()

        # Save subject results
        subject_feats_file = join(output_subfolder, '%s.npz' % subject)
        # np.savez_compressed(subject_feats_file, **f_dict)
        del f_dict
    
        # Give permissions
        os.system('chmod 777 ' + subject_feats_file)
    os.system('chmod -R 777 ' + output_subfolder)


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
    sphere_radius = [i for i in range(0, 3, step)]

    # Start main
    main()