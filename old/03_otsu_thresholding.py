#!/bin/env python3
from os.path import join, dirname

import numpy as np
import nibabel as nb

from skimage import filters
from skimage import exposure

import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_diff = '/home/ssilvari/Downloads/brain_avg/differences_norm.nii.gz'

    # Load NIFTI
    print('Loading file...')
    nii = nb.load(file_diff)
    nii_data = nii.get_data()

    # Threshold it
    print('Applying OTSU...')
    th = filters.threshold_isodata(nii_data[np.where(nii_data > 0)])
    nii_otsu_data = (nii_data > th).astype(np.float)
    print('\t- Threshold: {:.2f}'.format(th))

    # Create NIFTI output
    print('Saving result....')
    nii_otsu = nb.Nifti1Image(nii_otsu_data, nii.affine)
    nb.save(nii_otsu, join(dirname(file_diff), 'otsu.nii.gz'))

    # Plot results
    print('Plotting results...')
    hist, bins_center = exposure.histogram(nii_data)
    plt.plot(bins_center, hist, lw=2)
    plt.yscale('log')
    plt.axvline(th, color='k', ls='--')
    plt.show()
