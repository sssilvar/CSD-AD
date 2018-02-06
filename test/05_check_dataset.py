from __future__ import print_function

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Load params
with open('param/params.json') as json_file:
    jf = json.load(json_file)
    dataset_folder = jf['dataset_folder']
    data_file = jf['data_file']

# Define plot style
plt.style.use('ggplot')

# Set up logger
log_filename = os.path.join(dataset_folder, 'check_dataset.log')

# create logger with 'spam_application'
log = logging.getLogger('applog')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler(log_filename)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
ch.setFormatter(formatter)
log.addHandler(ch)

# Load DataFrame: df
df = pd.read_csv(data_file)

# Define types
df['target_categories'] = df['target_categories'].astype('category')
categories = df['target_categories'].cat.categories.tolist()

# Plot data histogram
# df['dx_group'].value_counts().plot(kind='bar', rot=40)

# Print data description
log.info('\n[  OK  ] Getting info:\n{}'.format(df['target_categories'].value_counts()))

# Check directories
tree = os.walk(dataset_folder)
folder_list = next(tree)[1]

error_count = 0
for folder in folder_list:
    """Check for folder existence"""
    flag = os.path.exists(os.path.join(dataset_folder, folder))

    if flag:
        log.info('[  OK  ] Folder %s exists', folder)
    else:
        log.info('[  ERROR  ] Folder %s does not exist')
        error_count += 1

# Check for errors
if error_count == 0:
    log.info('[  SUCCESS  ] Process finished without errors')
elif error_count > 0:
    log.info('[  WARNING  ] Process finished with ERRORS')
else:
    log.info('[  ERROR  ] UNEXPECTED ERROR')

log.info('Done!')


# Show plots
plt.show()
