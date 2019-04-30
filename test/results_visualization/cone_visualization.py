import sys
from configparser import ConfigParser
from os.path import join, dirname, realpath

import numpy as np
import nibabel as nb
import scipy.ndimage as ndi
from skimage import feature

import matplotlib.pyplot as plt
from nilearn import plotting

root = dirname(dirname(dirname(realpath(__file__))))

sys.path.append(root)
from lib.geometry import sphere, solid_cone
from lib.geometry import get_centroid, extract_sub_volume

if __name__ == '__main__':
    # Load from configuration
    cfg = ConfigParser()
    cfg.read(join(root, 'config', 'config.cfg'))
    registered_folder = cfg.get('dirs', 'dataset_folder_registered')
    extract_subs = False
    gradients = False

    # Define file names
    vol_filename = join(registered_folder, '002_S_0729/brainmask_reg.nii.gz')
    mni_filename = join(root, 'param/FSL_MNI152_FreeSurferConformed_1mm.nii')

    # Load images
    nii = nb.load(vol_filename)
    mni_vol = nb.load(mni_filename).get_data()
    vol = nii.get_data()

    # Calculate the inner and outer radius
    # for all the spheres: scales
    max_radius = 100
    tk = 20
    overlap = 5
    n_spheres = max_radius // (tk - overlap)
    scales = [(i * (tk - overlap), ((i + 1) * tk) - (i * overlap)) for i in range(n_spheres)]

    # Compute the centroid
    centroid = tuple(get_centroid(mni_vol > 0))
    print('[  OK  ] Centroid = {}'.format(centroid))

    # Use gradients
    if gradients:
        vol = mni_vol.astype(np.float)
        dx, dy, dz = np.gradient(vol)
        mag = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        nii_grad = nb.Nifti1Image(mag, nii.affine)
        display = plotting.plot_anat(nii_grad, annotate=True, draw_cross=False, dim=-1.25)
        # display.add_overlay(nii_grad)
    else:
        for i, radius in enumerate(scales):
            # Create a binary mask (cone between scales)
            mask = sphere(radius=radius, center=centroid)
            cone = solid_cone(radius=radius, center=centroid)

            # Mask sub-sampled volume
            vol_masked = np.multiply(vol, cone)

            # Extract sub-volumes
            if extract_subs:
                vol_sub, _ = extract_sub_volume(vol, radius=radius, centroid=centroid)
                mask, _ = extract_sub_volume(mask, radius=radius, centroid=centroid)
                vol_masked, _ = extract_sub_volume(vol_masked, radius=radius, centroid=np.array(centroid))
            else:
                vol_sub = vol

            # Create NIFTI Images
            nii_sub = nb.Nifti1Image(vol_sub, nii.affine)
            nii_mask = nb.Nifti1Image(mask.astype(np.int32), nii.affine)
            nii_masked = nb.Nifti1Image(cone.astype(np.int32), nii.affine)

            # Plot result
            display = plotting.plot_anat(nii_sub,
                                         # title='%s to %s vox.' % radius,
                                         black_bg=True,
                                         alpha=0.8)
            display.add_overlay(nii_mask, alpha=0.7, cmap='hot')
            display.add_overlay(nii_masked, alpha=0.8)

            # Save figures
            plt.savefig('/tmp/%s_to_%s_visualization.png' % radius)

            # save NIFTIs
            if i == 1:
                nb.save(nii_sub, '/tmp/brain.mgz')
                nb.save(nii_mask, '/tmp/sphere.mgz')
                nb.save(nii_masked, '/tmp/cone.mgz')

    # plt.show()
