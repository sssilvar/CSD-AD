import os
import numpy as np
import pandas as pd
import nibabel as nb
import matplotlib.pyplot as plt

# Define dataset location
dataset_folder = os.path.join('/run/media/sssilvar/DATA/FreeSurferSD')  # Desktop PC
dataset_data_file = os.path.join(os.getcwd(), 'test_data', 'csv_xls', 'data_df.csv')

root = os.path.join(os.getcwd(), '..')
region_labels_path = os.path.join(root, 'param', 'FreeSurferColorLUT.csv')

# Load DataFrame: df
df = pd.read_csv(dataset_data_file)

# Define subject - i
i = 10
sb_dir = os.path.join(dataset_folder, df['folder'][i])

brainmask_path = os.path.join(sb_dir, 'mri', 'brainmask.mgz')
aseg_path = os.path.join(sb_dir, 'mri', 'aseg.mgz')

# Load image, aseg, and labels
brainmask = nb.load(brainmask_path)
brainmask_data = brainmask.get_data()
aseg = nb.load(aseg_path)
aseg_data = aseg.get_data()

region_labels_df = pd.read_csv(region_labels_path, index_col=False)

# Plot a region
# region_name = 'Left-Cerebral-Cortex'
region_name = 'Left-Hippocampus'

region_id = region_labels_df['region_id'][region_labels_df['label_name'] == region_name].values
region_mask = aseg_data == region_id
region_data = brainmask_data * region_mask

m = 0
for i, slide in enumerate(region_data):
    if slide.mean() > m:
        m = slide.mean()
        sel_slide = slide
        slide_num = i


plt.figure()
plt.imshow(sel_slide, cmap='gray')
plt.axis('off')
plt.title('%s at slide: %d' % (region_name, slide_num))
plt.show(block=False)
