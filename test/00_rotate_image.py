import os
from time import time
from os.path import dirname, join, realpath

import nibabel as nb
from nilearn import plotting
from scipy import ndimage as ndi

import matplotlib.pyplot as plt

root = dirname(dirname(realpath(__file__)))

if __name__ == "__main__":
    # Define image to load
    nii_file = join(root, 'param', 'FSL_MNI152_FreeSurferConformed_1mm.nii')
    nii = nb.load(nii_file)
    img = nii.get_data()

    # Rotate it!
    ti = time()
    centroid = ndi.center_of_mass(img > 0)
    shift = tuple([int(128 - x) for x in centroid])
    print('[  INFO  ] Center of mass calculation: {} s'.format((time() - ti)))
    
    ti = time()
    img_shift = ndi.shift(img, shift=shift)
    print('[  INFO  ] Shifting operation: {} s'.format((time() - ti)))

    img_shift = img_shift[28:228, 28:228, 28:228]
    print(img_shift.shape)

    ti = time()
    img_rot = ndi.rotate(img_shift, angle=45, axes=(2,0), reshape=False)
    print('[  INFO  ] Rotation operation: {} s'.format((time() - ti)))
    
    # Build NIFTI Images
    nii_shift = nb.Nifti1Image(img_shift, nii.affine)
    nii_rot = nb.Nifti1Image(img_rot, nii.affine)
    
    # Plot images
    display = plotting.plot_anat(nii_shift)
    display.add_edges(nii_rot)

    # Show it!
    plt.show()