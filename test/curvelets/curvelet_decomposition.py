#!/bin/env python2
import os
import argparse
from os.path import join, dirname, basename, realpath

import pyct as ct
import numpy as np
import pandas as pd
import nibabel as nb
from scipy import stats

from nilearn import plotting

root = dirname(dirname(dirname(realpath(__file__))))

def parse_args():
    # Define default ROIs
    rois = [77, 78, 79, 10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58] + [4, 14, 15, 28, 43, 60, 72] + [i for i in range(1000,1036)] + [i for i in range(2000,2036)] # + [i for i in range(3000,3036)] + [i for i in range(4000,4036)]

    # Parse arguments
    parser = argparse.ArgumentParser(description="Curvelet extraction for neuroimages ROI")
    parser.add_argument('-sid',
                        help='Subject ID',
                        required=True)
    parser.add_argument('-out',
                        help='Output Directory.',
                        required=True)
    parser.add_argument('-vol',
                        help='Path to Brain volume',
                        required=True)
    parser.add_argument('-aseg',
                        help='Path to segmentation file',
                        required=True)
    parser.add_argument('-scales',
                        help='(Curvelet) Number of scales',
                        type=int,
                        default=4)
    parser.add_argument('-angles',
                        help='(Curvelet) Number of angles',
                        type=int,
                        default=4)
    parser.add_argument('-rois',
                        type=int,
                        help='List of ROIS. E.g. "17 18 ... 57"',
                        nargs='+',
                        default=rois)
    return parser.parse_args()


def extract_cubic_roi(vol, aseg, visualize=False):
    # Extract ROI
    roi_mask = (aseg.get_data() == roi).astype(np.int)
    if np.any(roi_mask):
        roi_vol = vol.get_data() * roi_mask

        # Reduce dimensionality to ROI
        ix, iy, iz = np.where(roi_mask)
        sx, sy, sz = max(ix) - min(ix), max(iy) - min(iy), max(iz) - min(iz)
        d = max([sx, sy, sz])
        
        roi_sq = np.zeros([d, d, d])

        # Bould an square volume from ROI and calculate volume in voxels
        roi_sq[:sx, :sy, :sz] = roi_vol[min(ix): max(ix), min(iy): max(iy), min(iz): max(iz)]
        voxels = np.sum(roi_mask)
        
        # PLot ROI if necessary
        if visualize:
            roi_nii = nb.Nifti1Image(roi_mask, affine=aseg.get_affine())
            plotting.plot_roi(roi_nii, bg_img=vol, black_bg=False)

            roi_sq_nii = nb.Nifti1Image(roi_sq, affine=aseg.get_affine())
            plotting.plot_anat(roi_sq_nii)
            plotting.show()
    else:
        print('[  WARNING  ] ROI not found, returning an empty volume')
        roi_sq = np.zeros([32, 32, 32])
        voxels = 0
    
    return roi_sq, voxels


def curvelet_decomposition(vol, n_scales, n_angles):
    # Create a Curvelet object
    A = ct.fdct3(
        vol.shape,
        nbs=n_scales,
        nba=n_angles,
        ac=True,
        norm=False,
        vec=True,
        cpx=False)

    # Apply curvelet to volume
    f = A.fwd(vol)
    
    # Assembly feature vector
    feats = pd.Series()
    for scale in range(n_scales):
        for angle in range(n_angles):
            # Get values in respective scale/angle
            try:
                ix = A.index(scale, angle)
                data = f[ix[0]:ix[1]] # Extract magnitude

                # Extract several statistics
                n, (mi, ma), mea, var, skew, kurt = stats.describe(data)

                # Assign to series
                feats['%d_%d_n' % (scale, angle)] = n
                feats['%d_%d_min' % (scale, angle)] = mi
                feats['%d_%d_max' % (scale, angle)] = ma
                feats['%d_%d_mean' % (scale, angle)] = mea
                feats['%d_%d_var' % (scale, angle)] = var
                feats['%d_%d_skew' % (scale, angle)] = skew
                feats['%d_%d_kurtosis' % (scale, angle)] = kurt
            except IndexError:
                pass

    return feats


if __name__ == "__main__":
    # Load parsed args
    args = parse_args()

    # Load ROI Names
    lut = pd.read_csv(join(root, 'param/FreeSurferColorLUT.csv'), index_col='region_id')
    roi_names = lut.loc[args.rois, 'label_name']

    # Print some info
    os.system('clear')
    print('[  INFO  ] Process information:')
    print('\t- Subject ID: %s' % args.sid)
    print('\t- MRI Volume: %s' % args.vol)
    print('\t- Segmentation: %s' % args.aseg)
    print('\t- Output folder: %s' % args.vol)
    print('\t- Number of scales: %d' % args.scales)
    print('\t- Number of angles: %d' % args.angles)

    # Load images
    vol = nb.load(args.vol)
    aseg = nb.load(args.aseg)

    # Define a DataFrame
    df = pd.DataFrame()

    # Extract ROIs
    for roi in args.rois:
        print('[  INFO  ] Extracting ROI: %s' % roi_names[roi])

        # Extract a cubic ROIs
        roi_img, roi_vol = extract_cubic_roi(vol, aseg, visualize=False)
        print('\t- ROI shape: %s' % str(roi_img.shape))
        print('\t- ROI volume: %d Vox.' % roi_vol)

        # Extract curvelet decomposition
        feats = curvelet_decomposition(roi_img, n_scales=args.scales, n_angles=args.angles)
        feats.name = args.sid
        feats['ROI'] = roi
        df = df.append(feats)

    df['ROI'] = df['ROI'].astype('int')  # Cast ROI to int

    # Save results
    out_subj_dir = join(args.out, args.sid)
    print('[  INFO  ] Saving subject\'s results in: %s' % out_subj_dir)
    os.system('mkdir %s' % out_subj_dir)
    df.to_csv(join(out_subj_dir, 'curvelet_roi_features_%d_scales_%d_angles.csv' % (args.scales, args.angles)))
        
        
        
    