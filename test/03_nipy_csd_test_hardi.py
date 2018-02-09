import os
import logging as log
import nibabel as nb
from dipy.viz import fvtk
import dipy.core.gradients as gr
from dipy.data import get_sphere
from dipy.data import read_stanford_hardi
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response

log.warning('Setting parameters...')
img_folder = os.path.join('/home/sssilvar/.dipy/stanford_hardi')

img_filename = os.path.join(img_folder, 'HARDI150.nii.gz')
bval_filename = os.path.join(img_folder, 'HARDI150.bval')
bvec_filename = os.path.join(img_folder, 'HARDI150.bvec')

log.warning('Loading files...')
img = nb.load(img_filename)
gtab = gr.gradient_table(bval_filename, bvec_filename)
data = img.get_data()

n = 100
data = data[:, :, :, 1:n]
gtab.bvals = gtab.bvals[1:n]
gtab.bvecs = gtab.bvecs[1:n, :]
gtab.gradients = gtab.gradients[1:n, :]

log.warning('Extracting response and ratio')
response, ratio = auto_response(gtab, data, roi_radius=5, fa_thr=0.7)

log.warning('Calculating CSD model...')
csd_model = ConstrainedSphericalDeconvModel(gtab, response)

log.warning('Initialising a sphere...')
sphere = get_sphere('symmetric642')
ren = fvtk.ren()

log.warning('Fitting ODF')
data_small = data[30:60, 60:90, 38:39]
csd_fit = csd_model.fit(data_small)
csd_odf = csd_fit.odf(sphere)

fodf_spheres = fvtk.sphere_funcs(csd_odf, sphere, scale=1.2, norm=False)
fvtk.add(ren, fodf_spheres)

log.warning('Saving illustration as csd_odfs.png')
fvtk.record(ren, out_path='csd_odfs.png', size=(600, 600))
