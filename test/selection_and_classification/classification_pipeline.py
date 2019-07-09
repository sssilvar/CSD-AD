#!/bin/env python3
import os
import sys
from multiprocessing import cpu_count
from os.path import join, isfile, dirname, realpath, basename

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.feature_selection import SelectFromModel

root = dirname(dirname(dirname(realpath(__file__))))
sns.set()
# plt.style.use('ggplot')
sys.path.append(root)
from mifs import MutualInformationFeatureSelector
from lib.feature_selection import MRMR

if __name__ == '__main__':
    # Features file
    try:
        feats_file = sys.argv[1]
    except IndexError:
        feats_file = '/home/ssilvari/Documents/temp/ADNI_temp/mapped/ADNI_FS_mapped_tk_15_overlap_4_ns_1/curvelet' \
                     '/sobel_curvelet_features_non_split_4_scales_32_angles_norm.csv'
    assert isfile(feats_file), f'File {feats_file} not found.'

    n_cores = cpu_count() // 2

    tk = feats_file.split('/')[-3].split('_')[4]
    overlap = feats_file.split('/')[-3].split('_')[6]

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

    selectors = {
        'mrmr': MRMR(method='MID', k_features=10),
        'mrmr2': MutualInformationFeatureSelector(method='MRMR', n_features=100, n_jobs=n_cores),
        'svc': SelectFromModel(LinearSVC(penalty='l2')),
        'lasso': SelectFromModel(LassoCV(cv=5))
    }

    # Create a classification pipeline
    pipeline = Pipeline([
        # ('scaler', RobustScaler()),
        ('selector', selectors['mrmr2']),
        ('clf', classifiers['rf'])
    ])

    # Define conversion times in months
    times = [24, 36, 60]

    # Results dataframe
    results = pd.DataFrame()

    for t in times:
        print(f'-- Conversion time: {t} months --')
        subjects_in_time = adnimerge.loc[(adnimerge['Month.STABLE'] >= t) | (adnimerge['Month.CONVERSION'] <= t)].index
        df_time = df.loc[subjects_in_time]

        print('--- Reshaping dataframe ---')
        feature_cols = [c for c in df_time.columns if '_' in c] + ['label']
        X_df = df_time.pivot(columns='sphere', values=feature_cols)
        X_df = X_df.dropna(axis='columns', how='all').fillna(0)

        print('--- Extracting labels --')
        y_df = X_df['label'].iloc[:, 0]
        categories = ['MCInc', 'MCIc']

        # Print info
        print(y_df.value_counts())

        print(X_df.head())
        print(y_df.value_counts())
        print(y_df.head())

        feature_names = np.array([i[0] for i in X_df.columns if 'label' not in i])
        X = X_df.drop('label', axis='columns').values
        y = np.array([1 if label == 'MCIc' else 0 for label in y_df])
        print(X.shape, y.shape)

        # Perform a k-fold
        kf = StratifiedKFold(n_splits=5, random_state=42)
        plt.figure()
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train model
            print(f'--- Training model (Fold {i}/{kf.get_n_splits()}) ---')
            pipeline.fit(X_train, y_train)

            # Test model
            print('--- Testing model ---')
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy}')
            print(classification_report(y_test, y_pred))

            # ROC estimators
            fpr, tpr, threshold = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            # Append to results
            res_series = pd.Series({
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            }, name=f'Fold {i + 1}')
            results = results.append(res_series)

            # Print selected features
            selector = pipeline.named_steps['selector']
            selected_weights = selector.ranking_
            print(f'Weights:\n{selected_weights}')

            features_mask = selector._get_support_mask() + [False]
            print(len(features_mask), len(feature_names))
            selected_features = feature_names[features_mask]
            print(f'Selected features:\n{selected_features}')
            print(f'Number of features selected: {len(selected_weights)}')

            # Plot
            plt.plot(fpr, tpr, label=f'Fold {i + 1} AUC = {roc_auc:0.2f}')

        mean_auc = results['auc'].mean()

        plt.title(f'ROC Curves {t} months (AUC = {mean_auc:0.2f})')
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        os.makedirs('/tmp/results', exist_ok=True)
        figname = f'rocs_{t}_months_tk_{tk}_overlap_{overlap}.png'
        plt.savefig(f'/tmp/results/{figname}')
