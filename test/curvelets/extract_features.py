import os
import sys
from os.path import join, dirname, realpath

import pandas as pd

current_dir = dirname(realpath(__file__))

if __name__ == "__main__":
    # Define some parameters
    # group_file = '/disk/FreeSurferSD/groupfile.csv'
    group_file = sys.argv[1]
    subjects_dir = dirname(group_file)
    out_folder = sys.argv[2]

    print(group_file)
    print(out_folder)

    # Load dataset
    df = pd.read_csv(group_file, index_col=0)
    
    n_scales = 4
    n_angles = 4
    rois = [77, 78, 79, 10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58] + [4, 14, 15, 28, 43, 60, 72] + [i for i in range(1000,1036)] + [i for i in range(2000,2036)] # + [i for i in range(3000,3036)] + [i for i in range(4000,4036)]
    rois_str = " ".join(map(str, rois))

    for i, subject in enumerate(df.index):
        print('[  INFO  ] Processing subject: %s' % subject)
        print('\t- ROIs: "%s"' % rois_str)

        # Create command
        script = join(current_dir, 'curvelet_decomposition.py')
        
        # # Load Brainmask + segmentation
        # vol = join(subjects_dir, subject, 'mri/brainmask.mgz')
        # aseg = join(subjects_dir, subject, 'mri/aparc+aseg.mgz')

        # Load just ROI (Fully shape analysis)
        mgz = join(subjects_dir, subject)
        mgz = join(mgz, 'FreeSurfer_Cross-Sectional_Processing_aparc+aseg')
        mgz = join(mgz, next(os.walk(mgz))[1][0])
        mgz = join(mgz, next(os.walk(mgz))[1][0])
        mgz = join(mgz, 'mri/aparc+aseg.mgz')
        vol = mgz
        aseg = mgz
        
        cmd = '%s -sid %s -vol %s -aseg %s -out %s -scales %d -angles %d -rois %s' % (script, subject, vol, aseg, out_folder, n_scales, n_angles, rois_str)
        print(cmd)
        os.system(cmd)

        # Create a whole DataFrame
        csv_file = join(out_folder, subject, 'curvelet_roi_features_%d_scales_%d_angles.csv' % (n_scales, n_angles))
        if i is 0:
            df_feats = pd.read_csv(csv_file, index_col=0)
        else:
            dfb = pd.read_csv(csv_file, index_col=0)
            df_feats = pd.concat([df_feats, dfb], axis=0, ignore_index=False, sort=False)

    # Save results
    df_feats.to_csv(join(out_folder, 'curvelet_features_rois_%d_scales_%d_angles.csv' % (n_scales, n_angles)))

