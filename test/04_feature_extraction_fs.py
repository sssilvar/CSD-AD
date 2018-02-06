from __future__ import print_function

import os
import pandas as pd
import nibabel as nb
import matplotlib.pyplot as plt

# Set parameters
dataset_folder = 'C:\Users\Smith\Documents\FreeSurferSD'
data_fle = os.path.join(os.getcwd(), 'test_data', 'csv_xls', 'dataset.xlsx')
plt.style.use('ggplot')

# Create a DataFrame for the dataset
df = pd.read_excel(data_fle)

# Create extra columns
df['target'] = df['dx_group'].astype('category')
df['folder'] = df['center_id'].apply(lambda x: '%03d' % int(x)) +\
               '_S_' + df['subject_id'].apply(lambda x: '%04d' % int(x))

# # Some descriptive statistics
# df['target'].value_counts().plot(kind='bar', rot=60)
# plt.title('Number of subjects per category')
# plt.ylabel('Counts')

# Load annotations


plt.show()
