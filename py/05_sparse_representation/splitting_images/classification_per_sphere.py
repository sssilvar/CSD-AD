#!/bin/env python3
import os
import logging
import argparse
from os.path import join, basename, dirname, realpath

import pandas as pd
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Define root folder
root = dirname(dirname(dirname(dirname(realpath(__file__)))))


def get_features_and_labels(df_features, confound_correction=False):
    """
    Get labels from dataframe IDs (ADNI only)
    """
    # Clean DataFrame
    df_features = df_features.fillna(0)

    # Load progressions from MCI to Dementia (MCIc)
    logger.info('[  INFO  ] Loading common data...')
    adnimerge = pd.read_csv(join(root, 'param/df_conversions.csv'), index_col='PTID')
    adnimerge = adnimerge[adnimerge['VISCODE'] == 'bl']

    if confound_correction:
        Y = adnimerge.loc[df_features.index, ['PTGENDER', 'AGE']]  # , 'PTRACCAT', 'SITE']#, 'APOE4', 'FDG']
        Y['PTGENDER'] = Y['PTGENDER'].astype('category').cat.codes
        Y['AGE2'] = Y['AGE'] ** 2
        Y = Y.fillna(Y.mean())

        # Load Intra Cranial Volume (ICV)
        icv = adnimerge.loc[df_features.index, 'ICV']
        icv = icv.fillna(icv.mean())

        # Normalize volumes by ICV (if file contains volumes)
        if np.any(['vol' in col for col in df_features.columns]):
            logger.info('[  INFO  ] Normalizing by ICV...')
            df_features = df_features.divide(icv, axis=0)
            logger.info(df_features.head())

        # Create a numpy array
        X = df_features.values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Subtract effect of confounders
        logger.info('[  INFO  ] Subtracting confounders effect...')
        W = np.linalg.inv(Y.T.dot(Y)).dot(Y.T.dot(X))
        X = np.nan_to_num(X - Y.dot(W))

    # Assign labels
    df_features['label'] = adnimerge.loc[df_features.index, 'target'].astype('category')
    return df_features


def parse_args():
    """Parse arguments at the moment of execution"""
    parser = argparse.ArgumentParser(description='Classification per sphere of curvelet features.')
    parser.add_argument('-f',
                        metavar='--file',
                        help='File containing features.')
    parser.add_argument('-clf',
                        metavar='--classifier',
                        help='Classifier used: [svm, random-forest]',
                        default='svm')
    parser.add_argument('-log',
                        metavar='--log-file',
                        help='Log file output')
    return parser.parse_args()


def setup_logger(log_file):
    logger_loc = logging.getLogger(__name__)
    logger_loc.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger_loc.addHandler(handler)

    return logger_loc


def balance_dataset(df_features):
    """Balances the dataset"""
    # Get lengths
    len_mcic = sum(df_features['label'] == 'MCIc')
    len_mcinc = sum(df_features['label'] == 'MCInc')

    # Balance respect to the least amount of subjects
    if len_mcinc > len_mcic:
        index_mcinc = df_features.loc[df_features['label'] == 'MCInc'].sample(n=len_mcic, random_state=21).index
        index_mcic = df_features.loc[df_features['label'] == 'MCIc'].index
    elif len_mcinc < len_mcic:
        index_mcic = df_features.loc[df_features['label'] == 'MCIc'].sample(n=len_mcic, random_state=21).index
        index_mcinc = df_features.loc[df_features['label'] == 'MCInc'].index
    else:
        return df_features

    return df_features.loc[index_mcic].append(df_features.loc[index_mcinc])


if __name__ == "__main__":
    os.system('clear')
    # Parse arguments and load params
    args = parse_args()
    data_file = args.f
    clf_type = args.clf
    log_file = args.log
    n_folds = 7

    # Print some info
    print('========== CLASSIFICATION ==========')
    print('\t- Features file: {}'.format(data_file))
    print('\t- Classifier used: {}'.format(clf_type))
    print('\t- Log file output {}'.format(log_file))
    print('Executing...')

    # Set logger
    logger = setup_logger(log_file=log_file)
    logger.info('\n\nExecuting classification task...')

    # Load dataset
    logger.info('[  INFO  ] Loading dataset...')
    df = pd.read_csv(data_file, index_col=0)
    df['sphere'] = df['sphere'].astype('category')
    logger.info('\t- DataFrame Shape: %s' % str(df.shape))

    for sphere in df['sphere'].cat.categories:
        logger.info('[  INFO  ] Classifying %s' % sphere)
        df_sphere = df[df['sphere'] == sphere].drop('sphere', axis=1)

        # Standardize and correct 
        df_std = get_features_and_labels(df_sphere)

        # Balance dataset
        df_std = balance_dataset(df_std)

        # Get X and y
        X = df_std.drop('label', axis=1).values
        y = df_std['label'] == 'MCIc'

        # Split data into training and testing set
        logger.info('[  INFO  ] Splitting dataset...')
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
        skf = StratifiedKFold(n_splits=n_folds, random_state=42)
        plt.figure(figsize=[12.8, 9.6], dpi=150)

        for fold_i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            logger.info('\n\n[  INFO  ] Processing fold No. %d' % fold_i)
            logger.info('\t- Number of features (%d)' % X.shape[1])
            logger.info('\t- Number of subjects \n%s' % df_std['label'].value_counts())
            logger.info('\t- Total observations (%d)' % len(y))
            logger.info('\t- Training observations (%d)' % len(y_train))
            logger.info('\t- Test observations (%d)' % len(y_test))

            # Setup classification pipeline
            logger.info('[  INFO  ] Setting Classifier up...')
            if clf_type == 'random-forest':
                clf = RandomForestClassifier(random_state=42)
                param_grid = {
                    'clf__n_estimators': [200, 500],
                    'clf__max_features': ['auto', 'sqrt', 'log2'],
                    'clf__max_depth': [4, 5, 6, 7, 8],
                    'clf__criterion': ['gini', 'entropy']
                    }
            else:
                clf = SVC(probability=True)
                # Set up the parameters to evaluate
                param_grid = {
                    'clf__kernel': ['rbf'],
                    'clf__C': [0.001, 0.01, 0.1, 1, 10],
                    'clf__gamma': [0.0001, 0.001, 0.01, 0.1, 1]
                }

            # Define a pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', SelectFromModel(LinearSVC(penalty='l2'))),
                # ('feature_selection', SelectFromModel(LassoCV(cv=5), threshold=1e-2)),
                ('clf', clf)
            ])

            # Set a grit to hypertune the classifier
            logger.info('\t- Tunning classifier...')
            clf_grid = GridSearchCV(pipeline, param_grid,
                                    cv=StratifiedKFold(n_splits=3, random_state=42),
                                    iid=True,
                                    n_jobs=3,
                                    scoring='roc_auc')
            # Train classifier
            clf_grid.fit(X_train, y_train)

            y_pred = clf_grid.predict(X_test)
            y_pred_proba = clf_grid.predict_proba(X_test)[:, 1]

            # Compute rates and plot AUC
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)

            logger.info('[  INFO  ] Classification report: \n%s' % classification_report(y_test, y_pred))
            logger.info('[  INFO  ] Best Params: %s' % clf_grid.best_params_)

            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr, tpr, label='AUC (fold %d) = %.2f' % (fold_i + 1, auc))

        plt.legend()
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(
            'ROC for {} voxels ({})'.format(sphere, basename(data_file).split('_')[0]),
            fontdict={'fontsize': 24})

        fig_file = join(dirname(data_file), 'ROC', basename(data_file)[:-4] + '_roc_%s.png' % sphere)
        plt.savefig(fig_file, bbox_inches='tight')

    print('DONE!\n\n')
    logger.info('DONE!\n\n\n')
