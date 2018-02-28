from __future__ import print_function

import sys

import os
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

# Define params
root = os.path.join(os.getcwd(), '..')
params_file = os.path.join(root, 'param', 'params.json')
lut_csv = os.path.join(root, 'lib', 'FreeSurferColorLUT.csv')

features_file = 'sph_features.csv'
features_file = os.path.join(root, 'features', features_file)
sys.path.append(root)
import lib.Freesurfer as Fs

# Load params
with open(params_file) as json_file:
    jf = json.load(json_file)
    dataset_folder = jf['dataset_folder']
    data_file = jf['data_file']

# Load dataset data into a DataFrame: df
df = pd.read_csv(os.path.join(root, data_file))
df = df.sort_values('folder')

# Define a subject
subjects_zipped = zip(df['folder'], df['dx_group'], df['target'])

# Create a FS object
fs_obj = Fs.Freesurfer(dataset_folder, df)

# Define a DataFrame for features extracted
features_df = pd.DataFrame()

# Define an error dict
error = {
    'counter': 0,
    'subjects': [],
    'error_type': []
}

cols_test = []
# Check for features file 'features.csv'
if not os.path.exists(features_file):

    # Start extracting features
    for i, (subject, dx_group, label) in enumerate(subjects_zipped):
        """Goes over every subject extracting features."""
        try:
            subject_folder = os.path.join(dataset_folder, subject)
            print(subject_folder)
            print(dx_group)

            # Execute test:
            sph_feat, aseg, sh_complex, brainmask = fs_obj.extract_sph_features(subject, lut_csv)
            cols_test.append(features_df.columns)  # Test feature names

            # Add extra info
            print('[  OK  ] Processing data')
            sph_feat['subject_id'] = subject
            sph_feat['target_name'] = dx_group
            sph_feat['target'] = label

            # Append to the main DataFrame
            features_df = features_df.append(sph_feat)

            # Extract the hippo and save it
            region = sh_complex * (aseg == 17)
            for j, slide in enumerate(region.__abs__()):
                if slide.mean() > 0:
                    plt.imsave(
                        os.path.join(root, 'features', 'left_hippo', dx_group, '%d.png' % j),
                        slide[90:140, 90:140]
                    )

        except IOError as e:
            print('[  ERROR  ] it seems like subject %s does not exist' % subject)
            error['counter'] += 1
            error['subjects'].append(subject)
            error['error_type'].append(e)



        if i >= 1:
            break
    # Print Final report
    print('[  OK  ] Process finished with %d errors' % error['counter'])
    print('List of subjects with errors: \n', error['subjects'])

    # Clean and save results to a csv file
    features_df = features_df.reset_index()
    features_df.to_csv(features_file)

# # Start Classification
# print('\n\n[  OK  ] STARTING CLASSIFICATION')
# features_df = pd.read_csv(features_file)
#
# # Define X and y
# features_df = features_df.drop(['Unnamed: 0', 'index'], axis=1)  # Clean Dataframe
# X_df = features_df.drop(['subject_id', 'target_name', 'target'], axis=1)
# X = X_df.values
# y = features_df['target'].values
#
# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=21, stratify=y)
#
# # # Start Classifying: SVM-RBF
# lasso = Lasso(alpha=0.020408163265306121)
#
# lasso.fit(X_train, y_train)
#
# lasso_coef = lasso.coef_
# print(lasso_coef)
#
# # Plot features
# plt.style.use('ggplot')
# plt.figure()
# plt.plot(range(X.shape[1]), lasso_coef)
# plt.xticks(range(X.shape[1]), X_df.columns, rotation=30)
# plt.margins(0.02)
#
# selected_features_cols = X_df.columns[lasso_coef != 0]
# print(selected_features_cols)
#
# # Define a new X (feature selection)
# X = X_df[selected_features_cols]
#
# # Split data (again)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=21, stratify=y)
#
# # Start classifying
# print('[  OK  ]: Starting Pipeline...')
# preprocessing = Pipeline([
#     ('scaler', StandardScaler()),
#     ('svm', SVC(kernel='rbf', probability=True))
# ])
#
# param_grid = {
#     'svm__C': np.logspace(-3, 2, 20),
#     'svm__gamma': np.logspace(-3, 2, 20)
# }
#
# preprocessing = GridSearchCV(preprocessing, param_grid)
#
# print('[  OK  ]: Fitting model...')
# preprocessing.fit(X_train, y_train)
# y_pred = preprocessing.predict(X_test)
#
# print('Classification report: \n {}'.format(classification_report(y_test, y_pred)))
# print('Score: {}'.format(preprocessing.score(X_test, y_test)))
# print('Best Params: {}'.format(preprocessing.best_params_))
#
# y_pred_proba = preprocessing.predict_proba(X_test)[:, 1]
#
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
# auc = roc_auc_score(y_test, y_pred_proba)
#
# plt.figure()
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr)
# plt.legend(['AUC = ' + str(auc)])
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
#
# plt.show()
