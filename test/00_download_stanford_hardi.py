from __future__ import print_function

import numpy as np
from dipy.reconst.csdeconv import auto_response
from dipy.data import fetch_stanford_hardi, read_stanford_hardi

# Download HARDI images from stanford
fetch_stanford_hardi()

# Load images and get data
img, gtab = read_stanford_hardi()
data = img.get_data()

# get response
response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)

# Print data
print(data)
