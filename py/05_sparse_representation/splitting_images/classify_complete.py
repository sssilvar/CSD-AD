#!/bin/env python
import os
import sys
import logging
import argparse
from configparser import ConfigParser
from os.path import join, dirname, realpath, basename

import pandas as pd
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import matplotlib
import matplotlib.pyplot as plt

root = dirname(dirname(dirname(dirname(realpath(__file__)))))
plt.style.use('ggplot')
matplotlib.use('Agg')


def parse_args():
    parser = argparse.ArgumentParser(description="Classification task for Currvelet features in MCI to AD conversion")
    parser.add_argument('-time',
                        help='Conversion/Stability time criteria (24, 36, 60) months',
                        default=24,
                        type=int)
    parser.add_argument('-folds',
                        help='Numbers of folds for cross-validation (K-fold)',
                        type=int,
                        default=10)
    parser.add_argument('-clf',
                        help='Type of classifier to be used [svm/rf]',
                        type=str,
                        default='svm')
    parser.add_argument('-imtype',
                        help='Type of the images to be classified [intensity/gradient/sobel]',
                        type=str,
                        default='gradient')
    return parser.parse_args()


def reshape_dataframe(df):
    # Uses 'sphere' to reshape in a single one row
    df['sphere'] = df['sphere'].astype('category')

    df_list = []
    for sphere in df['sphere'].cat.categories:
        df_buff = df.loc[df.sphere == sphere]
        df_buff = df_buff.drop('sphere', axis='columns')
        df_buff.columns = ['{}_{}'.format(i, sphere) for i in df_buff.columns]
        df_list.append(df_buff)

    return pd.concat(df_list, axis='columns', ignore_index=False, sort=True)


def balance_dataset(df_features):
    """Balances the dataset"""
    print_and_log('Balancing the dataset...')
    # Get lengths
    len_mcic = sum(df_features['target'] == 'MCIc')
    len_mcinc = sum(df_features['target'] == 'MCInc')

    # Balance respect to the least amount of subjects
    if len_mcinc > len_mcic:
        index_mcinc = df_features.loc[df_features['target'] == 'MCInc'].sample(n=len_mcic, random_state=21).index
        index_mcic = df_features.loc[df_features['target'] == 'MCIc'].index
    elif len_mcinc < len_mcic:
        index_mcic = df_features.loc[df_features['target'] == 'MCIc'].sample(n=len_mcic, random_state=21).index
        index_mcinc = df_features.loc[df_features['target'] == 'MCInc'].index
    else:
        return df_features

    return df_features.loc[index_mcic].append(df_features.loc[index_mcinc])


def get_times_and_labels(x, time):
    """Obtains labels from indexes"""
    # Load information from ADNIMERGE
    df_common = pd.read_csv(
        join(root, 'param/df_conversions_with_times.csv'),
        index_col='PTID'
    )

    # Assign labels
    x['target'] = df_common.reindex(x.index).target.astype('category')

    # Filter per month
    print('For {} months | Data shape: {}:'.format(time, x.shape))
    df_common = df_common.loc[(df_common['Month.STABLE'] >= time) | (df_common['Month.CONVERSION'] <= time)]
    print_and_log('Number of subjects found in ADNIMERGE for {} months:\n{}'
                  .format(time, df_common['target']
                          .value_counts()))

    # Return Labels
    return x.reindex(df_common.index).dropna(axis='rows', how='all')


def setup_logger(logfile):
    new_logger = logging.getLogger(__name__)
    new_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(logfile)
    handler.setFormatter(formatter)
    new_logger.addHandler(handler)

    return new_logger


def print_and_log(msg, level='INFO'):
    if level == 'INFO':
        logger.info(msg)
    elif level == 'DEBUG':
        logger.debug(msg)
    elif level == 'WARNING':
        logger.warning(msg)
    elif level == 'DEBUG':
        logger.debug(msg)
    else:
        raise KeyError('Log level of type {level} does not exist'.format(level=level))
    print('[  {level}  ] {msg}'.format(level=level, msg=msg))


if __name__ == "__main__":
    # Clear screen
    os.system('clear')

    # Parse arguments
    args = parse_args()
    n_folds = args.folds
    study_time = args.time
    img_type = args.imtype
    clf_type = args.clf.lower()
    clf_name = 'SVM' if clf_type == 'svm' else 'Random Forest'

    # Parse configuration
    cfg = ConfigParser()
    cfg.read(join(root, 'config/config.cfg'))
    data_folder = cfg.get('dirs', 'sphere_mapping')
    n_cores = cfg.getint('resources', 'n_cores')
    n_cores = n_cores if n_cores <= 10 else 10

    # Load features file and set number of folds
    feats_file = join(data_folder, '{}_curvelet_features_4_scales_32_angles.csv'.format(img_type))

    # Create and setup logger
    log_file = join(dirname(feats_file),
                    'ROC', 'classification_{basename}_aio.log'.format(basename=basename(feats_file).split('.')[0]))
    logger = setup_logger(log_file)
    print_and_log('Classification task:')
    print_and_log('Selected classifier {}'.format(clf_name))
    print_and_log('Features file: {}'.format(feats_file))
    print_and_log('Log file: {}'.format(log_file))
    print_and_log('Conversion/stable time: {} months'.format(study_time))

    X = pd.read_csv(feats_file, index_col=0)
    X = reshape_dataframe(X)

    # Get labels, remove unused observations and append them as a column
    X = get_times_and_labels(X, time=study_time)
    # X = balance_dataset(X)

    print_and_log('Preview\n {}'.format(X.head()))
    print_and_log('Data dimensions: {}'.format(X.shape))
    print_and_log('Number of observations:\n{}'.format(X.target.value_counts()))

    # Assign values to X and y
    y = X.target == 'MCIc'
    X = X.drop('target', axis='columns')
    X = X.fillna(X.mean())

    # Split dataset
    print_and_log('Splitting dataset...')
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=21, stratify=y)
    skf = StratifiedKFold(n_splits=n_folds, random_state=42)

    # Create a figure
    plt.figure(figsize=[12.8, 9.6], dpi=150)

    for fold_i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Start classification task
        print_and_log('\n\nProcessing fold No. {}'.format(fold_i + 1))
        print_and_log('Number of features (%d)' % X.shape[1])
        print_and_log('Total observations (%d)' % len(y))
        print_and_log('Training observations (%d)' % len(y_train))
        print_and_log('Test observations (%d)' % len(y_test))

        # Define classifier and grid depending on param
        print_and_log('Setting classifier\'s parameters...')
        if clf_type == 'svm':
            clf = SVC(probability=True)
            param_grid = {
                'clf__kernel': ['rbf'],
                'clf__C': [0.001, 0.01, 0.1, 1, 10],
                'clf__gamma': [0.0001, 0.001, 0.01, 0.1, 1]
            }
        else:
            clf = RandomForestClassifier(random_state=42)
            param_grid = {
                'clf__n_estimators': [200, 500],
                'clf__max_features': ['auto', 'sqrt', 'log2'],
                'clf__max_depth': [4, 5, 6, 7, 8],
                'clf__criterion': ['gini', 'entropy']
            }

        # Start classification
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(LinearSVC(penalty='l2'))),
            ('clf', clf)
        ])

        # Set a grit to hypertune the classifier
        print_and_log('Looking for the best parameters...')
        clf_grid = GridSearchCV(pipeline,
                                param_grid,
                                cv=StratifiedKFold(n_splits=3, random_state=42),
                                iid=True,
                                n_jobs=n_cores)
        clf_grid.fit(X_train, y_train)

        y_pred = clf_grid.predict(X_test)
        y_pred_proba = clf_grid.predict_proba(X_test)[:, 1]

        # Compute rates and plot AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)

        print_and_log('Classification report: \n%s' % classification_report(y_test, y_pred))
        print_and_log('Best Params: %s' % clf_grid.best_params_)

        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Fold {} AUC = {:.2f}'.format((fold_i + 1), auc))

    # Get Classifier name, basename of the features file and
    # the type of images (gradient, intensities, sobel)
    clf_type = str(pipeline.named_steps['clf']).split('(')[0]
    basename_file = basename(feats_file).split('.')[0]
    img_type = basename(feats_file).split('_')[0]

    # Set labels, enable legends and title
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.title('ROC {} ({})'.format(clf_type, img_type))

    fig_file = join(dirname(feats_file),
                    'ROC',
                    '{name}_aio_roc_{folds}_fold.png'.format(name=basename_file, folds=n_folds))
    plt.savefig(fig_file, bbox_inches='tight')
