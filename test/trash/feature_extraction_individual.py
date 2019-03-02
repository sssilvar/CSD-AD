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

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

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
    # Create Curvelet object for 360x180 px
    A = ct.fdct2(
        (360,180), 
        nbs=n_scales, 
        nba=n_angles, 
        ac=True, 
        norm=False, 
        vec=True, 
        cpx=False)


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
        print(f.shape)

        # Convert data to dict
        buff = clarray_to_gen_gaussian_dict(A, f, n_scales, n_angles, r)
        f_dict.update(buff)
        del buff, f, img

    # Save subject results
    subject_feats_file = join(output_subfolder, '%s.npz' % subject)
    np.savez_compressed(subject_feats_file, **f_dict)

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
    parser.add_argument('-subject', metavar='--type',
                        help='Subject ID',
                        default='002_S_0729')
    parser.add_argument('-label', metavar='--type',
                        help='Label of patient (diagnosis)',
                        default='MCIc')
    args = parser.parse_args()
    
    # --- MAIN ---
    # Set parameters
    n_scales = args.s
    n_angles = args.a
    img_type = args.t
    subject = args.subject
    label = args.label
    target = label == 'MCIc'
    results_folder = args.f

    output_folder = join(root, 'output')
    output_subfolder = join(output_folder, 'curv_feats_%s_nscales_%d_nangles_%d' % (img_type, n_scales, n_angles))
    
    try:
        os.mkdir(output_subfolder)
    except OSError:
        pass

    step = 25 # Propagation step
    delta = 25 # Sphere thickness
    sphere_radius = [i for i in range(0, 100, step)]

    # Start main
    main()