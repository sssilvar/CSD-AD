import os
import nibabel as nb

# Choose the file to be played with
filename = 'test_data/941_S_1363.mgz'

# Correct filename
filename = os.path.join(os.getcwd(), filename)
print('[  OK  ] File to be processed is located in: %s' % filename)

# Load MRI file
mri = nb.load(filename)
img = mri.get_data()

# Define a center and a radius
x, y, z = (128, 128, 128)
r = 50

# Start mapping!


