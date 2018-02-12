import os
import json
import pandas as pd
import lib.Freesurfer as Fs
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# Define params
root = os.path.join(os.getcwd(), '..')
params_file = os.path.join(root, 'param', 'params.json')

# Load params
with open(params_file) as json_file:
    jf = json.load(json_file)
    dataset_folder = jf['dataset_folder']
    data_file = jf['data_file']


# Load dataset data into a DataFrame: df
df = pd.read_csv(os.path.join(root, data_file))

# Define a subject
subjects_zipped = zip(df['folder'], df['dx_group'], df['target'])

# Create a FS object
Fs = Fs.Freesurfer(dataset_folder, df)

# Define a DataFrame for features extracted
cols = ['region_id', 'region_name', 'mean', 'std', 'hemisphere', 'subject_id', 'target_name', 'target']
features_df = pd.DataFrame(columns=cols)

# Define an error dict
error = {
    'counter': 0,
    'subjects': [],
    'error_type': []
}

# Check for features file 'features.csv'
if not os.path.exists(os.path.join(os.getcwd(), 'features.csv')):

    # Start extracting features
    for i, (subject, dx_group, label) in enumerate(subjects_zipped):
        """Goes over every subject extracting features."""
        try:
            subject_folder = os.path.join(dataset_folder, subject)
            print(subject_folder)

            # Load labels
            tk_stats_lh, _ = Fs.read_thickness(subject, 'lh')
            tk_stats_rh, _ = Fs.read_thickness(subject, 'rh')

            # Merge Hemispheres info
            frames = [tk_stats_lh, tk_stats_rh]
            tk_stats_df = pd.concat(frames)

            # Add extra info
            tk_stats_df['hemi_and_reg'] = tk_stats_df['hemisphere'] + '_' + tk_stats_df['region_name']
            tk_stats_df['subject_id'] = subject
            tk_stats_df['target_name'] = dx_group
            tk_stats_df['target'] = label

            features_df = features_df.append(tk_stats_df)
        except IOError as e:
            print('[  ERROR  ] it seems like subject %s does not exist' % subject)
            error['counter'] += 1
            error['subjects'].append(subject)
            error['error_type'].append(e)

    # Print Final report
    print('[  OK  ] Process finished with %d errors' % error['counter'])
    print('List of subjects with errors: \n', error['subjects'])

    # Save results to a csv file
    features_df.to_csv('features.csv')


# Start Classification
print('\n\n[  OK  ] STARTING CLASSIFICATION')
features_df = pd.read_csv('features.csv')

# Define X and y
X = features_df[['mean', 'std']].values
y = features_df['target'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=21, stratify=y)

# Select relevant features: coef
# lasso = Lasso(alpha=0.4, normalize=True)
lasso = Lasso(alpha=0.4)
lasso.fit(X, y)

lasso_coef = lasso.coef_
print(lasso_coef)

# Plot features
plt.figure()
plt.plot(range(X.shape[1]), lasso_coef)
plt.xticks(range(X.shape[1]), features_df['region_name'], rotation=60)
plt.margins(0.02)
plt.show(block=False)
