#!/bin/env python
import os
import sys
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
    
    df_list=[]
    for sphere in df['sphere'].cat.categories:
        df_buff = df.loc[df.sphere == sphere]
        df_buff = df_buff.drop('sphere', axis='columns')
        df_buff.columns = ['{}_{}'.format(i, sphere) for i in df_buff.columns]
        df_list.append(df_buff)
        
    return pd.concat(df_list, axis='columns', ignore_index=False)

def get_labels(x):
    "Obtains labels from indexes"
    # Load information from ADNIMERGE
    df_common = pd.read_csv(
        join(root, 'param/df_conversions.csv'),
        index_col='PTID'
    )

    # Return Labels
    return df_common.loc[x.index, 'target'].astype('category')



if __name__ == "__main__":
    # Clear screen
    os.system('clear')

    # Load Features file
    feats_file = get_feats_file()
    print('[  INFO  ] Classification task:')
    print('\t- Features file: {}'.format(feats_file))

    X = pd.read_csv(feats_file, index_col=0)
    X = reshape_dataframe(X)
    y = get_labels(X) == 'MCIc'

    print(X.head())
    print(get_labels(X).value_counts())

    # Split dataset
    print('[  INFO  ] Splitting dataset...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=21, stratify=y)
    print('\t- Number of features (%d)' % X.shape[1])
    print('\t- Total observations (%d)' % len(y))
    print('\t- Training observations (%d)' % len(y_train))
    print('\t- Test observations (%d)' % len(y_test))

    # Start classification
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(LinearSVC(penalty='l2'))),
        # ('clf', SVC(probability=True))
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # Define a search grid
    print('[  INFO  ] Setting classifier\'s parammeters...')
    param_grid = {
        'clf__n_estimators': [200, 500],
        'clf__max_features': ['auto', 'sqrt', 'log2'],
        'clf__max_depth': [4, 5, 6, 7, 8],
        'clf__criterion': ['gini', 'entropy']
    }

    # param_grid = {
    #     'clf__kernel': ['rbf', 'linear'],
    #     'clf__C': [0.001, 0.01, 0.1, 1, 10],
    #     'clf__gamma': [0.0001, 0.001, 0.01, 0.1, 1]
    # }

    # Set a grit to hypertune the classifier
    print('[  INFO  ] Looking for the best parammeters...')
    clf_grid = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=5, random_state=42), iid=True)
    clf_grid.fit(X_train, y_train)

    y_pred = clf_grid.predict(X_test)
    y_pred_proba = clf_grid.predict_proba(X_test)[:,1]

    # Compute rates and plot AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    print('[  INFO  ] Classification report: \n%s' % classification_report(y_test, y_pred))
    print('[  INFO  ] Best Params: %s' % clf_grid.best_params_)

    clf_type = str(pipeline.named_steps['clf']).split('(')[0]

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.legend(['AUC = %.2f' % auc])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC {} ({})'.format(clf_type, feats_file.split('_')[0]))

    fig_file = join(dirname(feats_file), 'ROC', basename(feats_file).split('.')[0] + '_roc.png')
    plt.savefig(fig_file, bbox_inches='tight')
    
