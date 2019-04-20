#!/bin/env python3
from os.path import join, dirname, realpath

import nibabel as nb
from skimage import exposure

from nilearn import plotting
import matplotlib.pyplot as plt

root = dirname(dirname(dirname(realpath(__file__))))

if __name__ == '__main__':
    # Load MNI and target
    mni_file = join(root, 'param', 'FSL_MNI152_FreeSurferConformed_1mm.nii')
    target_file = '/run/media/ssilvari/SMITH_DATA_1TB/Universidad/MSc/Thesis/Dataset/ADNI_FS_registered_flirt' \
                  '/002_S_0729/orig_reg.nii.gz'
    mni_nii = nb.load(mni_file)
    target_nii = nb.load(target_file)

    # Equalize
    # img_eq = exposure.equalize_hist(target_nii.get_data())
    img = target_nii.get_data()
    img_eq = exposure.equalize_hist(img, mask=(img != 0))
    nii_eq = nb.Nifti1Image(img_eq, target_nii.affine)

    # Plot results
    fig, ax = plt.subplots(nrows=2)
    plotting.plot_anat(target_nii, axes=ax[0], title='Original')
    plotting.plot_anat(nii_eq, axes=ax[1], title='Intensity normalized')

    plt.show()



