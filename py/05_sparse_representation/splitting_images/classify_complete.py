#!/bin/env python
import os
import sys
import logging
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

import matplotlib.pyplot as plt

root = dirname(dirname(dirname(dirname(realpath(__file__)))))
plt.style.use('ggplot')


def get_feats_file():
    try:
        return sys.argv[1]
    except IndexError:
        cfg = ConfigParser()
        cfg.read(join(root, 'config/config.cfg'))
        data_folder = cfg.get('dirs', 'sphere_mapping')
        return join(data_folder, 'gradient_curvelet_features_4_scales_32_angles.csv')


def reshape_dataframe(df):
    # Uses 'sphere' to reshape in a single one row
    df['sphere'] = df['sphere'].astype('category')

    df_list = []
    for sphere in df['sphere'].cat.categories:
        df_buff = df.loc[df.sphere == sphere]
        df_buff = df_buff.drop('sphere', axis='columns')
        df_buff.columns = ['{}_{}'.format(i, sphere) for i in df_buff.columns]
        df_list.append(df_buff)

    return pd.concat(df_list, axis='columns', ignore_index=False)


def balance_dataset(df_features):
    """Balances the dataset"""
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


def get_labels(x):
    """Obtains labels from indexes"""
    # Load information from ADNIMERGE
    df_common = pd.read_csv(
        join(root, 'param/df_conversions.csv'),
        index_col='PTID'
    )

    # Return Labels
    return df_common.loc[x.index, 'target'].astype('category')


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

    # Load features file and set number of folds
    feats_file = get_feats_file()
    n_folds = 7

    # Create and setup logger
    log_file = join(dirname(feats_file),
                    'ROC', 'classification_{basename}_aio.log'.format(basename=basename(feats_file).split('.')[0]))
    logger = setup_logger(log_file)
    print_and_log('Classification task:')
    print_and_log('Features file: {}'.format(feats_file))
    print_and_log('Log file: {}'.format(log_file))

    X = pd.read_csv(feats_file, index_col=0)
    X = reshape_dataframe(X)

    # Get labels and append them as a column
    X['target'] = get_labels(X)
    X = balance_dataset(X)

    # Assign values to X and y
    y = X.target == 'MCIc'
    X = X.drop('target', axis='columns')

    print_and_log('Preview\n {}'.format(X.head()))
    print_and_log('Data dimensions: {}'.format(X.shape))
    print_and_log('Number of observations:\n{}'.format(get_labels(X).value_counts()))

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
        print_and_log('Number of features (%d)' % X.shape[1])
        print_and_log('Total observations (%d)' % len(y))
        print_and_log('Training observations (%d)' % len(y_train))
        print_and_log('Test observations (%d)' % len(y_test))

        # Start classification
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(LinearSVC(penalty='l2'))),
            # ('clf', SVC(probability=True))
            ('clf', RandomForestClassifier(random_state=42))
        ])

        # Define a search grid
        print_and_log('Setting classifier\'s parameters...')
        param_grid = {
            'clf__n_estimators': [200, 500],
            'clf__max_features': ['auto', 'sqrt', 'log2'],
            'clf__max_depth': [4, 5, 6, 7, 8],
            'clf__criterion': ['gini', 'entropy']
        }

        # param_grid = {
        #     'clf__kernel': ['rbf'],
        #     'clf__C': [0.001, 0.01, 0.1, 1, 10],
        #     'clf__gamma': [0.0001, 0.001, 0.01, 0.1, 1]
        # }

        # Set a grit to hypertune the classifier
        print_and_log('Looking for the best parameters...')
        clf_grid = GridSearchCV(pipeline,
                                param_grid,
                                cv=StratifiedKFold(n_splits=3, random_state=42),
                                iid=True,
                                n_jobs=3)
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
