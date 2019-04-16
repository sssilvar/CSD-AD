#!/bin/env python3
from os.path import dirname, join, realpath

import numpy as np
import nibabel as nb
from nilearn import plotting

from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

import matplotlib.pyplot as plt

root = dirname(dirname(dirname(realpath(__file__))))

if __name__ == '__main__':
    mni_nii = join(root, 'param', 'FSL_MNI152_FreeSurferConformed_1mm.nii')
    test_nii = r'/run/media/ssilvari/HDD Data/Universidad/MSc/Thesis/Dataset/FreeSurferSD/002_S_0729/mri/brainmask.mgz'

    # Load images
    mni_nii = nb.load(mni_nii)
    mni_data = mni_nii.get_data()

    test_nii = nb.load(test_nii)
    test_data = test_nii.get_data()

    # Metric for registration
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    # Transform center of mass
    c_of_mass = transform_centers_of_mass(mni_data, mni_nii.affine,
                                          test_data, test_nii.affine)

    # Params
    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(mni_data, test_data, transform, params0,
                                  mni_nii.affine, test_nii.affine,
                                  starting_affine=starting_affine)

    # Create nii
    transformed = translation.transform(test_data)
    reg_nii = nb.Nifti1Image(transformed, translation.affine)

    # Plot both
    display = plotting.plot_anat(mni_nii)
    display.add_overlay(reg_nii, alpha=0.6)
    plt.show()




