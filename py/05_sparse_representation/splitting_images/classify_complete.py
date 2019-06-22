#!/bin/env python3
import os
import shutil
import sys
import logging
import argparse
from configparser import ConfigParser
from os.path import join, dirname, realpath, basename, isdir

import pandas as pd
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, KFold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, confusion_matrix

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
    parser.add_argument('-features',
                        help='Features file',
                        default=None)
    parser.add_argument('-folds',
                        help='Numbers of folds for cross-validation (K-fold)',
                        type=int,
                        default=5)
    parser.add_argument('-clf',
                        help='Type of classifier to be used [svm/rf]',
                        type=str,
                        default='svm')
    parser.add_argument('-imtype',
                        help='Type of the images to be classified [intensity/gradient/sobel]',
                        type=str,
                        default='gradient')
    parser.add_argument('-tune',
                        help='Type of the images to be classified [intensity/gradient/sobel]',
                        type=int,
                        default=0)
    parser.add_argument('-clear',
                        help='Clear all the previous output instead of appending.',
                        type=int,
                        default=0)
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
    os.system('cls' if os.name == 'nt' else 'clear')

    # Parse arguments
    args = parse_args()
    n_folds = args.folds
    study_time = args.time
    img_type = args.imtype
    clf_type = args.clf.lower()
    clf_name = 'SVM' if clf_type == 'svm' else 'Random Forest'
    clf_tuning = bool(args.tune)

    # Parse configuration
    cfg = ConfigParser()
    cfg.read(join(root, 'config/config.cfg'))
    data_folder = cfg.get('dirs', 'sphere_mapping')
    n_cores = cfg.getint('resources', 'n_cores')
    n_cores = n_cores if n_cores <= 20 else 20

    # Load features file and set number of folds
    if not args.features:
        feats_file = join(data_folder, '{}_curvelet_features_4_scales_32_angles.csv'.format(img_type))
    else:
        feats_file = args.features

    # Folders of interest
    out_folder = join(dirname(feats_file), 'ROC')

    # Delete previous output if clear enabled
    if args.clear:
        try:
            shutil.rmtree(out_folder)
        except FileNotFoundError as e:
            print(e)

    # Create folder if does not exist
    if not isdir(out_folder):
        os.mkdir(out_folder)

    # Create and setup logger
    log_file = join(out_folder,
                    'classification_{basename}_aio.log'.format(basename=basename(feats_file).split('.')[0]))
    logger = setup_logger(log_file)
    print_and_log('Classification task:')
    print_and_log('Selected classifier {}'.format(clf_name))
    print_and_log('Hypertunning?: {}'.format(clf_tuning))
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
    # skf = KFold(n_splits=n_folds, random_state=42)

    # Create a figure
    plt.figure(figsize=[12.8, 9.6], dpi=150)

    # Variables of interests
    metrics = pd.DataFrame(columns=['ACC', 'SEN', 'SPE', 'AUC'])
    tpr_df = pd.DataFrame()
    fpr_df = pd.DataFrame()
    thr_df = pd.DataFrame()
    mean_fpr = np.linspace(0, 1, 100)

    for fold_i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        fold_name = 'Fold_{}'.format(fold_i + 1)

        # Start classification task
        print_and_log('\n\nProcessing {}'.format(fold_name))
        print_and_log('Number of features (%d)' % X.shape[1])
        print_and_log('Total observations (%d)' % len(y))
        print_and_log('Training observations (%d)' % len(y_train))
        print_and_log('Test observations (%d)' % len(y_test))

        # Define classifier and grid depending on param
        print_and_log('Setting classifier\'s parameters...')
        if clf_tuning:
            if clf_type == 'svm':
                clf = SVC(kernel='rbf', probability=True)
                param_grid = {
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
        else:
            # No optimization is done
            if clf_type == 'svm':
                clf = SVC(
                    probability=True,
                    kernel='rbf',
                    gamma=0.001
                )  # TODO: Add params
            else:
                clf = RandomForestClassifier(
                    random_state=42,
                    criterion='entropy',
                    max_depth=7,
                    max_features='auto',
                    n_estimators=200,
                    n_jobs=n_cores
                )

        # Start classification
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            # ('feature_selection', SelectFromModel(LinearSVC(penalty='l2'))),
            ('feature_selection', SelectKBest(k=100)),
            ('clf', clf)
        ])

        # Set a grit to hypertune the classifier
        if clf_tuning:
            print_and_log('Looking for the best parameters...')
            clf_grid = GridSearchCV(pipeline,
                                    param_grid,
                                    cv=StratifiedKFold(n_splits=3, random_state=42),
                                    iid=True,
                                    scoring='roc_auc',
                                    n_jobs=n_cores)
            clf_grid.fit(X_train, y_train)

            y_pred = clf_grid.predict(X_test)
            y_pred_proba = clf_grid.predict_proba(X_test)[:, 1]
            print_and_log('Best Params: %s' % clf_grid.best_params_)
        else:
            print_and_log('Performing classification...')
            print_and_log('Training classifier...')
            pipeline.fit(X_train, y_train)

            print_and_log('Classifying..')
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Compute rates and plot AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
        auc = roc_auc_score(y_test, y_pred_proba)

        print_and_log('Classification report: \n%s' %
                      classification_report(
                          y_test, y_pred,
                          labels=[0, 1],
                          target_names=['MCInc', 'MCIc']))

        # Print and log feature importances
        sel_feats_mask = pipeline.named_steps['feature_selection'].get_support()

        if clf_type == 'rf':
            feat_weights = pipeline.named_steps['clf'].feature_importances_
            fi_srt = '\nFeature importances'
            for index, feature_importance in zip(X.columns[sel_feats_mask], feat_weights):
                fi_srt += f', {index}, {feature_importance}'
            print_and_log(fi_srt)

        # Compile extra metrics
        acc = accuracy_score(y_test, y_pred, normalize=True)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)

        print_and_log('ACC: {:.2f}'.format(acc))
        print_and_log('SEN: {:.2f}'.format(sen))
        print_and_log('SPE: {:.2f}'.format(spe))

        # Append metrics to DataFrame
        m_data = pd.Series({'ACC': acc, 'SEN': sen, 'SPE': spe, 'AUC': auc}, name=fold_name)
        metrics = metrics.append(m_data)

        # fpr_df = fpr_df.append(pd.Series(np.interp(mean_fpr, fpr, tpr), name=fold_name))
        tpr_df = tpr_df.append(pd.Series(np.interp(mean_fpr, fpr, tpr), name=fold_name))
        thr_df = thr_df.append(pd.Series(thresholds, name=fold_name))

        print(tpr.shape, tpr_df.shape)
        print(tpr.shape, tpr_df.shape)
        print(metrics.tail())

        # Plot ROC
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='{} AUC = {:.2f}'.format(fold_name.replace('_', ' '), auc))

    # Get Classifier name, basename of the features file and
    # the type of images (gradient, intensities, sobel)
    basename_file = basename(feats_file).split('.')[0]
    img_type = basename(feats_file).split('_')[0]

    # Set labels, enable legends and title
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.title('ROC {} ({})'.format(clf_name, img_type))

    fig_file = join(dirname(feats_file),
                    'ROC',
                    '{name}_aio_roc_{folds}_fold_{clf}_{time}_months.png'
                    .format(name=basename_file, folds=n_folds, clf=clf_type, time=study_time))
    plt.savefig(fig_file, bbox_inches='tight')

    # Get stats
    mean_metrics = metrics.mean()

    tpr_df.iloc[:, 0] = 0
    mean_tpr = tpr_df.mean(axis=0)
    std_tpr = tpr_df.std(axis=0)

    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    # Print final results
    print_and_log('Final mean metrics: \n{}'.format(mean_metrics))

    # Create final results as DataFrame and save them
    f_data = {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'mean_auc': mean_metrics['AUC'],
        'mean_axc': mean_metrics['ACC'],
        'mean_sen': mean_metrics['SEN'],
        'mean_spe': mean_metrics['SPE'],
        'time': '{} Months'.format(study_time)
    }

    csv_file_out = join(out_folder,
                        '{name}_aio_{folds}_fold_{clf}_{time}_months_final.csv'
                        .format(name=basename_file, folds=n_folds, clf=clf_type, time=study_time))

    df_final = pd.DataFrame().from_dict(f_data)
    df_final.to_csv(csv_file_out)

    # Plot final ROC
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--')

    plt.plot(mean_fpr, mean_tpr, label='AUC = {:.2f}'.format(mean_metrics['AUC']))
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=.2)

    # Set labels, enable legends and title
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.title('ROC {} ({})'.format(clf_name, img_type))

    fig_file = join(out_folder,
                    '{name}_aio_roc_{folds}_fold_{clf}_{time}_months_final.png'
                    .format(name=basename_file, folds=n_folds, clf=clf_type, time=study_time))
    plt.savefig(fig_file, bbox_inches='tight')
