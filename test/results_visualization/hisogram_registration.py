#!/bin/env python3
import numpy as np
import nibabel as nb
from skimage import exposure

import matplotlib.pyplot as plt

plt.style.use('ggplot')

if __name__ == '__main__':
    # Load images
    orig_file = '/home/ssilvari/Downloads/001_stripped.nii.gz'
    bmask_fs = '/home/ssilvari/Documents/temp/ADNI_temp/ADNI_FS/002_S_0729/mri/brainmask.mgz'
    registered_fs_file = '/home/ssilvari/Downloads/001_reg.nii.gz'
    registered_fsl_file = '/home/ssilvari/Downloads/001_registered.nii.gz'

    # Load files
    nii_orig = nb.load(orig_file).get_data()
    nii_bmask = nb.load(bmask_fs).get_data()
    nii_fs = nb.load(registered_fs_file).get_data()
    nii_fsl = nb.load(registered_fsl_file).get_data()

    nii_orig_eq = exposure.equalize_hist(nii_orig, mask=(nii_orig != 0))
    nii_bmask_eq = exposure.equalize_hist(nii_bmask, mask=(nii_bmask != 0))

    # Plot histograms
    plt.hist(nii_bmask[np.where(nii_bmask != 0)], label='Brain mask FS', bins=100, alpha=0.6)
    plt.hist(nii_orig[np.where(nii_orig != 0)], label='Stripped', bins=100, alpha=0.6)
    plt.hist(nii_fs[np.where(nii_fs != 0)], label='Registered FS', bins=100, alpha=0.6)
    plt.hist(nii_fsl[np.where(nii_fsl != 0)], label='Registered FSL', bins=100, alpha=0.6)
    plt.legend()

    # Plot equalized histograms
    plt.figure()
    plt.hist(nii_orig_eq[np.where(nii_orig != 0)], label='Brain mask ROBEX', bins=100, alpha=0.6)
    plt.hist(nii_bmask_eq[np.where(nii_bmask != 0)], label='Brain mask FS', bins=100, alpha=0.6)
    # plt.yscale('log')
    plt.legend()
    plt.show()
