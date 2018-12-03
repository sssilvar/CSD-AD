#!/bin/env python3
import os
import sys
from os.path import join, basename, dirname, realpath

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Define root folder
root = dirname(dirname(dirname(dirname(realpath(__file__)))))

def get_features_and_labels(df):
    """
    Get labels from dataframe IDs (ADNI only)
    """
    # Clean DataFrame
    df = df.fillna(0)

    # Load progressions from MCI to Dementia (MCIc)
    print('[  INFO  ] Loading confounders...')
    adni_prog = pd.read_csv(join(root, 'param/common/adnimerge_conversions_v2.csv'), index_col='PTID')
    adni_no_prog = pd.read_csv(join(root, 'param/common/adnimerge_MCInc_v2.csv'), index_col='PTID')
    adnimerge = pd.read_csv(join(root, 'param/common/adnimerge.csv'), index_col='PTID', low_memory=False)
    adnimerge = adnimerge[adnimerge['VISCODE'] == 'bl']
    df_prog = pd.concat([adni_no_prog, adni_prog], axis=0)

    Y = df_prog.loc[df.index, ['PTGENDER', 'AGE']] #, 'PTRACCAT', 'SITE']#, 'APOE4', 'FDG']
    Y['PTGENDER'] = Y['PTGENDER'].astype('category').cat.codes
    Y['AGE2'] = Y['AGE'] ** 2
    Y = Y.fillna(Y.mean())
    
    # Load Intra Cranial Volume (ICV)
    icv = df_prog.loc[df.index, 'ICV']
    icv = icv.fillna(icv.mean())
    
    # Normalize volumes by ICV (if file contains volumes)
    if np.any(['vol' in col for col in df.columns]):
        print('[  INFO  ] Normalizing by ICV...')
        df = df.divide(icv, axis=0)
        print(df.head())
    
    # Create a numpy array
    X = df.values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Substract effect of confounders
    print('[  INFO  ] Substracting confounders effect...')
    W = np.linalg.inv(Y.T.dot(Y)).dot(Y.T.dot(X))    
    X = np.nan_to_num(X - Y.dot(W))

    # Assign labels
    labels = list(np.empty(len(df.index)))
    for i, sid in enumerate(df.index):
        if sid in adni_prog.index:
            labels[i] = 'MCIc'
        elif sid in adni_no_prog.index:
            labels[i] = 'MCInc'
        elif sid in adnimerge.index:
            labels[i] = adnimerge.loc[sid, 'DX.bl']
        else:
            labels[i] = 'NA'
    # df['label'] = labels
        df['label'] = pd.Categorical(
            labels, 
            categories=['MCInc', 'MCIc'], 
            ordered=False)

    return df


if __name__ == "__main__":
    os.system('clear')
    # Load dataset
    # data_file = '/home/ssilvari/Documents/temp/spherical_mapping/sphere_mapped_4_spheres/gradient_curvelet_features_5_scales_16_angles.csv'
    data_file = sys.argv[1]

    # Load dataset
    print('[  INFO  ] Loading dataset...')
    df = pd.read_csv(data_file, index_col=0)
    df['sphere'] = df['sphere'].astype('category')
    print('\t- DataFrame Shape: %s' % str(df.shape))


    for sphere in df['sphere'].cat.categories:
        print('[  INFO  ] Classifying %s' % sphere)
        df_sphere = df[df['sphere'] == sphere].drop('sphere', axis=1)
        
        # Standardize and correct 
        df_std = get_features_and_labels(df_sphere)
        
        # Get X and y
        X = df_std.drop('label', axis=1).values
        y = df_std['label'].cat.codes

        # Set a pipeline and classify
        # Split data into training and testing set
        print('[  INFO  ] Splitting dataset...')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=21, stratify=y)
        print('\t- Number of features (%d)' % X.shape[1])
        print('\t- Total observations (%d)' % len(y))
        print('\t- Training observations (%d)' % len(y_train))
        print('\t- Test observations (%d)' % len(y_test))

        # Start Classifying: SVM-RBF
        print('[  INFO  ] Setting Classifier up...')
        clf = SVC(probability=True)
        # clf = RandomForestClassifier(random_state=42)

        # Set up the parameters to evaluate
        param_grid = {
            'kernel': ['rbf'],
            'C': [0.001, 0.01, 0.1, 1, 10],
            'gamma': [0.0001, 0.001, 0.01, 0.1, 1]
        }
        # param_grid = {
        #     'n_estimators': [200, 500],
        #     'max_features': ['auto', 'sqrt', 'log2'],
        #     'max_depth': [4, 5, 6, 7, 8],
        #     'criterion': ['gini', 'entropy']
        #     }

        # Set a grit to hypertune the classifier
        print('\t- Tunning classifier...')
        clf_grid = GridSearchCV(clf, param_grid, cv=StratifiedKFold(n_splits=10, random_state=42), iid=True)
        clf_grid.fit(X_train, y_train)

        y_pred = clf_grid.predict(X_test)
        y_pred_proba = clf_grid.predict_proba(X_test)[:,1]

        # Compute rates and plot AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)

        print('[  INFO  ] Classification report: \n%s' % classification_report(y_test, y_pred))
        print('[  INFO  ] Best Params: %s' % clf_grid.best_params_)

        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.legend(['AUC = %.2f' % auc])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC for %s' % sphere)

        fig_file = join(dirname(data_file), 'ROC', basename(data_file)[:-4] + '_roc_%s.png' % sphere)
        plt.savefig(fig_file, bbox_inches='tight')
