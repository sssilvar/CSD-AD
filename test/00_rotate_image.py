import os

import numpy as np
from numpy import pi
import cv2

from scipy.ndimage import affine_transform

img = cv2.imread(os.path.join(os.getcwd(), 'test_data', 'marker.png'))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cos_gamma = np.cos(-pi / 40)
sin_gamma = np.sin(-pi / 40)
rot_affine = np.array([[cos_gamma, -sin_gamma, 10],
                       [sin_gamma, cos_gamma, 10],
                       [0, 0, 1]])

out = affine_transform(gray, rot_affine)
cv2.imshow('Frame', out)

k = cv2.waitKey(5) & 0xFF

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


