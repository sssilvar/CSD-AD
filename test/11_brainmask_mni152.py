import os

import numpy as np
import nibabel as nb

root = os.path.dirname(os.path.dirname(__file__))


if __name__ == '__main__':
    """
    MNI template extracted from
    http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009
    """

    filename = 'C:/Users/sssilvar/Desktop/converted/mni_152.nii'
    mask_filename = 'C:/Users/sssilvar/Desktop/converted/mni152_mask.nii'

    # load volume
    nii = nb.load(filename)
    img = nii.get_data()

    # Load brainmask
    nii_mask = nb.load(mask_filename)
    mask = nii_mask.get_data().astype(np.bool)

    # Do the mathemagics
    img_masked = img * mask

    # Save image
    output_filename = os.path.join(root, 'param', 'mni_152_brainmask.mgz')
    out = nb.Nifti1Image(img_masked, affine=np.eye(4))
    nb.save(out, output_filename)
