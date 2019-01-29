#!/bin/env python
from os.path import join, dirname, realpath
from nilearn import plotting

import matplotlib.pyplot as plt

# Get root folder
root = dirname(dirname(dirname(realpath(__file__))))

# Set main folder
main_folder = '/home/ssilvari/Documents/temp/ADNI_test/'
failed_folder = join(main_folder, 'failed_registration')
success_folder = join(main_folder, 'registered')
out_folder = '/home/ssilvari/Documents/latex/cimalab-slides/advisory-advances/results/registration'

subjects = ['002_S_0729', '002_S_0782', '002_S_0954', '002_S_1070', '002_S_1155']

# MNI template path
mni_file = join(root, 'param/FSL_MNI152_FreeSurferConformed_1mm.nii')

for i, subject in enumerate(subjects):
    failed_file = join(failed_folder, subject, 'brainmask_reg.mgz')
    succcess_file = join(success_folder, subject, 'brainmask_reg.nii.gz')

    display = plotting.plot_anat(mni_file, title='Failed registration')
    display.add_overlay(failed_file, alpha=0.5)
    plt.savefig(join(out_folder, '{}_failed.png'.format(i)))

    display = plotting.plot_anat(mni_file, title='Successful registration')
    display.add_overlay(succcess_file, alpha=0.5)
    plt.savefig(join(out_folder, '{}_success.png'.format(i)))
    
    # plt.show()