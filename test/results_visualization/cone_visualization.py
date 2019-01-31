import sys
from configparser import ConfigParser
from os.path import join, dirname, realpath

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
from nilearn import plotting

root = dirname(dirname(dirname(realpath(__file__))))

sys.path.append(root)
from lib.geometry import sphere
from lib.geometry import get_centroid


if __name__ == '__main__':
    # Load from configuration
    cfg = ConfigParser()
    cfg.read(join(root, 'config', 'config.cfg'))
    registered_folder = cfg.get('dirs', 'dataset_folder_registered')

    # Define file names
    vol_filename = join(registered_folder, '002_S_0729/brainmask_reg.nii.gz')
    mni_filename = join(root, 'param/FSL_MNI152_FreeSurferConformed_1mm.nii')

    # Load images
    nii = nb.load(vol_filename)
    mni_vol = nb.load(mni_filename).get_data()
    vol = nii.get_data()

    # Define radius and center
    # radius = (0, 25)
    scales = [
        (0, 25),
        (25, 50),
        (50, 75),
        (75, 100)
    ]
    centroid = tuple(get_centroid(mni_vol > 0))
    print('[  OK  ] Centroid = {}'.format(centroid))

    for i, radius in enumerate(scales):
        # Create a binary mask (cone between scales)
        mask = sphere(radius=radius, center=centroid)

        # Mask sub-sampled volume
        vol_masked = np.multiply(vol, mask)

        # Create NIFTI Images
        nii_mask = nb.Nifti1Image(mask.astype(np.int32), nii.affine)
        nii_masked = nb.Nifti1Image(vol_masked, nii.affine)

        # Plot result
        display = plotting.plot_anat(nii, title='%s to %s vox.' % radius)
        display.add_overlay(nii_mask, alpha=0.3)
        display.add_overlay(nii_masked, alpha=0.6)

        # Save figures
        plt.savefig('/tmp/%s_to_%s_visualization.png' % radius)

        plt.show()
