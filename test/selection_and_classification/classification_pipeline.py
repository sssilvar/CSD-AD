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
        feats_file = '/home/ssilvari/Documents/temp/ADNI_temp/mapped/ADNI_FS_mapped_tk_15_overlap_4_ns_1/curvelet' \
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

    # # Drop Scale 0
    # drop_cols = [c for c in df.columns if '0_0_' in c]
    # df.drop(drop_cols, axis=1, inplace=True)

    classifiers = {
        'svm': SVC(gamma='auto', kernel='rbf'),
        'rf': RandomForestClassifier(),
        'lda': LinearDiscriminantAnalysis()
    }

    # Create a classification pipeline
    pipeline = Pipeline([
        # ('scaler', RobustScaler()),
        ('selector', MRMR(method='MIQ', k_features=100)),
        ('clf', classifiers['rf'])
    ])

    # Define conversion times in months
    times = [24, 36, 60]

    for t in times:
        print(f'-- Conversion time: {t} months --')
        subjects_in_time = adnimerge.loc[(adnimerge['Month.STABLE'] >= t) | (adnimerge['Month.CONVERSION'] <= t)].index
        df_time = df.loc[subjects_in_time]

        feature_cols = [c for c in df_time.columns if '_' in c] + ['label']
        X_df = df_time.pivot(columns='sphere', values=feature_cols)
        X_df = X_df.dropna(axis='columns', how='all').fillna(X_df.mean())

        y_df = X_df['label'].iloc[:, 0]
        categories = ['MCInc', 'MCIc']

        # Print info
        print(y_df.value_counts())

        print(X_df.head())
        print(y_df.value_counts())
        print(y_df.head())
        print(y_df.head())

        X = X_df.drop('label', axis='columns').values
        y = np.array([1 if label == 'MCIc' else -1 for label in y_df])
        print(X.shape, y.shape)

        # Perform a k-fold
        kf = StratifiedKFold(n_splits=5, random_state=42)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train model
            pipeline.fit(X_train, y_train)

            # Test model
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy}')
            print(classification_report(y_test, y_pred))
