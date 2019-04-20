#!/bin/env python3
__author__ = 'Santiago Smith'
__description__ = 'An automated method for ROI extraction in groups of MRI'
__year__ = '2019'
__cite__ = """
Silva, et al. Characterizing brain patterns in conversion from mild cognitive impairment (MCI) to Alzheimer's disease,
13th International Symposium on Medical Information Processing and Analysis, 2017
"""

import argparse
from os.path import join, dirname, realpath

import numpy as np
import pandas as pd
import nibabel as nb
from skimage import exposure

import matplotlib.pyplot as plt

root = dirname(dirname(realpath(__file__)))
plt.style.use('ggplot')


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(
        description='{desc}\n\nCitation:{cite}'.format(desc=__description__, cite=__cite__),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-groupfile', help='CSV containing subject IDs', required=True)
    parser.add_argument('-out', help='Output folder', required=True)
    parser.add_argument('-folder', help='Subjects folder', required=False)

    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    csv_file = args.groupfile
    data_folder = args.folder if args.folder is not None else dirname(csv_file)
    out_folder = args.out
    print(data_folder, out_folder)

    # Load subjects
    subjects_df = pd.read_csv(csv_file, index_col=0)

    # Load labels file
    csv_mci = join(root, 'param', 'df_conversions_with_times.csv')
    mci_df = pd.read_csv(csv_mci, index_col='PTID')

    # Load MNI template
    mni_nii = nb.load(join(root, 'param', 'FSL_MNI152_FreeSurferConformed_1mm.nii'))

    # Initialize average images
    converter_sum_vol = np.zeros([256, 256, 256])
    stable_sum_vol = np.zeros_like(converter_sum_vol)

    # Number of cases found
    n_converters = n_stables = 0

    # Do the magic
    for sid in subjects_df.index:
        try:
            dx = mci_df.loc[sid, 'target']
            print('Processing {} - {}'.format(sid, dx))
            mri_vol = nb.load(join(data_folder, sid, '001_reg.nii.gz')).get_data()  # .astype(np.float)
            # plt.hist(mri_vol[np.where(mri_vol != 0)].ravel(), bins=100)
            # plt.show()

            mri_vol = exposure.equalize_hist(mri_vol, mask=(mri_vol != 0))

            if dx == 'MCIc':
                converter_sum_vol -= mri_vol
                n_converters += 1
            elif dx == 'MCInc':
                stable_sum_vol += mri_vol
                n_stables += 1
        except KeyError as e:
            print('Subject {} not found in MCI list'.format(e))
        except FileNotFoundError as e:
            print(e)

    print('\nFinal report:\n\t- Number of MCIc:{mcic}\n\t- Number of MCInc: {mcinc}'
          .format(mcic=n_converters, mcinc=n_stables))

    # Calculate the average
    stable_avg_vol = stable_sum_vol / n_stables
    converter_avg_vol = converter_sum_vol / n_converters

    # Map of absolute differences
    map_of_differences_vol = np.abs(stable_sum_vol - converter_sum_vol)
    map_of_differences_vol_norm = map_of_differences_vol / map_of_differences_vol.max() * 255

    # Create NIFTI images
    stable_nii = nb.Nifti1Image(stable_avg_vol, affine=mni_nii.affine)
    converter_nii = nb.Nifti1Image(converter_avg_vol, affine=mni_nii.affine)
    map_of_differences_nii = nb.Nifti1Image(map_of_differences_vol, mni_nii.affine)
    map_of_differences_norm_nii = nb.Nifti1Image(map_of_differences_vol_norm, mni_nii.affine)

    # Save them all
    nb.save(stable_nii, join(out_folder, 'MCInc.nii.gz'))
    nb.save(converter_nii, join(out_folder, 'MCIc.nii.gz'))
    nb.save(map_of_differences_nii, join(out_folder, 'differences.nii.gz'))
    nb.save(map_of_differences_norm_nii, join(out_folder, 'differences_norm.nii.gz'))
