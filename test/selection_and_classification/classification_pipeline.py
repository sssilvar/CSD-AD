#!/bin/env python3
import os
import sys
from os.path import join, isfile, dirname, realpath

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

root = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(root)
from lib.feature_selection import MRMR

if __name__ == '__main__':
    # Features file
    try:
        feats_file = sys.argv[1]
    except IndexError:
        feats_file = '/home/ssilvari/Documents/temp/ADNI_temp/mapped/ADNI_FS_mapped_tk_25_overlap_0_ns_1/curvelet' \
                     '/sobel_curvelet_features_non_split_4_scales_32_angles.csv'
    assert isfile(feats_file), f'File {feats_file} not found.'

    # ADNIMERGE file
    adnimerge = pd.read_csv(join(root, 'param', 'df_conversions_with_times.csv'), index_col='PTID')

    # Load into a DataFrame
    df = pd.read_csv(feats_file, index_col=0, low_memory=False)
    df['label'] = adnimerge.reindex(df.index)['target']

    # Sort dataframe
    sorted_columns = ['label'] + [c for c in df.columns if 'label' not in c]
    df = df[sorted_columns]
    # Create a list of spheres
    spheres = sorted(df['sphere'].value_counts().index.tolist())

    # Drop subjects (single visit, regressions)
    subjects_to_drop = df[['label']].isnull().query('label == True').index
    df.drop(subjects_to_drop, inplace=True)
    print(df.head())

    classifiers = {
        'svm': SVC(gamma='auto', kernel='rbf'),
        'rf': RandomForestClassifier(),
        'lda': LinearDiscriminantAnalysis()
    }

    # Create a classification pipeline
    pipeline = Pipeline([
        # ('scaler', RobustScaler()),
        ('selector', MRMR(method='MIQ', k_features=10)),
        ('clf', classifiers['lda'])
    ])

    # Define conversion times in months
    times = [24, 36, 60]

    print(adnimerge['Month.STABLE'].head())

    for t in times:
        subjects_in_time = adnimerge.loc[(adnimerge['Month.STABLE'] >= t) | (adnimerge['Month.CONVERSION'] <= t)].index
        df_time = df.loc[subjects_in_time]
        # Classify per sphere
        for sphere in spheres:
            # Extract sphere features and features
            sph_df = df_time.query(f'sphere == "{sphere}"')
            X_df = sph_df.drop(['label', 'sphere'], axis='columns').fillna(0)
            y_df = sph_df['label'].astype('category')

            print(X_df.head())
            print(y_df.value_counts())
            print(y_df.head())
            print(y_df.cat.codes.head())

            X, y = X_df.values, y_df.cat.codes.values

            # Perform a k-fold
            kf = StratifiedKFold(n_splits=5)
            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Train model
                pipeline.fit(X_train, y_train)

                # Test model
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f'Accuracy: {accuracy}')
                print(classification_report(y_test, y_pred, labels=y_df.cat.categories.tolist()))

        # Print info
        print(f'-- Conversion time: {t} months --')
        print(y.value_counts())
