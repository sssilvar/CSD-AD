from __future__ import print_function

import os
import numpy as np
from dipy.data import get_sphere
from dipy.sims.voxel import single_tensor_odf
from dipy.viz import fvtk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Filename for output
filename = 'csd_response.png'
size = (400, 400)

# Add path to filename and set ggplot for plt: filename
filename = os.path.join(os.getcwd(), 'test_data', filename)
plt.style.use('ggplot')

# Create a render object
ren = fvtk.ren()

# Define values for fODF reconstruction
evals = np.array([2, 2, 5])

evecs = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]])

# Set a shpere
sphere = get_sphere('symmetric642')

# Assign responses
response_odf = single_tensor_odf(sphere.vertices,
                                 evals,
                                 evecs)

response_actor = fvtk.sphere_funcs(response_odf, sphere)

# Render the spherical function
fvtk.add(ren, response_actor)
print('[  OK  ]\t Saving illustration: ' + filename)
fvtk.record(ren, out_path=filename, size=size)
print('[  OK  ] DONE!')

# Visualise the output
img = plt.imread(filename)
plt.imshow(img)
plt.axis('off')
plt.show(block=False)
