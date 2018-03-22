import os

import numpy as np
from numpy import pi
import cv2
from scipy.ndimage import rotate

from scipy.ndimage import affine_transform

img = cv2.imread(os.path.join(os.getcwd(), 'test_data', 'marker.png'))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_rot = rotate(gray, angle=30, reshape=False)
print(gray.ndim)

cos_gamma = np.cos(-pi / 6)
sin_gamma = np.sin(-pi / 6)
rot_affine = np.array([[cos_gamma, sin_gamma],
                       [-sin_gamma, cos_gamma]])
offset = np.array([112, 112])
offset = np.dot(rot_affine, offset)
tmp = np.zeros((2,), dtype=np.float64)
tmp[0] = float(225) / 2.0 - 0.5
tmp[1] = float(225) / 2.0 - 0.5
offset = tmp - offset

out = np.zeros_like(gray)
affine_transform(gray, rot_affine, offset=offset, output=out)
cv2.imshow('Orig', gray)
cv2.imshow('Rotated', gray_rot)
cv2.imshow('Affine', out)

k = cv2.waitKey(5) & 0xFF

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


