import os
from os.path import join, dirname, realpath

import numpy as np
import nibabel as nb
import pandas as pd

# Set root folder
root = dirname(dirname(dirname(realpath(__file__))))
print(root)

if __name__ == '__main__':
    os.system('clear')
    # Set groupfile (subjects to process)
    # group_file = '/disk/Datasets/ADNI/screening_aseg/ADNI/groupfile.csv'
    # group_file = '/disk/fs_conversion/groupfile.csv'
    group_file = '/disk/Datasets/OASIS/oasis_extracted/groupfile.csv'
    dataset_folder = dirname(group_file)

    df_group = pd.read_csv(group_file, index_col=0)
    lut = pd.read_csv(join(root, 'param/FreeSurferColorLUT.csv'), index_col='region_id')

    # Create a dataframe
    rois = [77, 78, 79, 10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]
    cols = lut.loc[rois, 'label_name'].tolist()
    df = pd.DataFrame(columns=cols)
    
    # Extract volumnes
    for subject in df_group.index:
        # Load image
        mgz = join(dataset_folder, subject)
        # mgz = join(mgz, 'FreeSurfer_Cross-Sectional_Processing_aparc+aseg')
        # mgz = join(mgz, next(os.walk(mgz))[1][0])
        # mgz = join(mgz, next(os.walk(mgz))[1][0])
        # mgz = join(mgz, 'mri/aparc+aseg.mgz')
        mgz = join(mgz, 'mri/aseg.mgz')
        print('[  INFO  ] Processing: %s' % mgz)

        img = nb.load(mgz).get_data()

        # Calculate volume
        data = {key: np.sum(np.where(img == val)) for key,val in zip(cols, rois)}

        # Create series
        series = pd.Series(data=data, name=subject)
        print(series)
        df = df.append(series)

    # Save output
    csv_out = join(dataset_folder, 'wmh_voxels.csv')
    print('[  INFO  ] Saving results as: %s' % csv_out)
    df.to_csv(csv_out)

