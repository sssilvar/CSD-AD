from __future__ import print_function

import os
import json
import numpy as np
import pandas as pd
import nibabel as nb
import nibabel.freesurfer as nbfs
import matplotlib.pyplot as plt


# Load params
with open('param/params.json') as json_file:
    jf = json.load(json_file)
    dataset_folder = jf['dataset_folder']
    xls_orig = jf['xls_orig']
    csv_out = jf['data_file']


# Set plot style: ggplot
plt.style.use('ggplot')

# Create a DataFrame for the dataset
df = pd.read_excel(xls_orig)

# Create extra columns
df['target'] = df['dx_group'].apply(lambda x: int(x == 'MCIc'))
df['target_categories'] = df['dx_group'].astype('category')
df['folder'] = df['center_id'].apply(lambda x: '%03d' % int(x)) +\
               '_S_' + df['subject_id'].apply(lambda x: '%04d' % int(x))

# # Some descriptive statistics
# df['target'].value_counts().plot(kind='bar', rot=60)
# plt.title('Number of subjects per category')
# plt.ylabel('Counts')

print(df.info())

# Save cleaned dataframe: data_df.csv
df.to_csv(csv_out)

# Show plots
plt.show()
