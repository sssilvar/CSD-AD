#!/bin/env python3
import os
import sys
from os.path import join, basename, dirname, realpath

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Define root folder
root = dirname(dirname(dirname(realpath(__file__))))


if __name__ == "__main__":
    # Load dataset
    # data_file = '~/Documents/results/sphere_mapped_curvelet_pca/curv_feats_gradient_nscales_5_nangles_8/curv_feats_gradient_nscales_5_nangles_8_pca_20_comp.csv'
    data_file = sys.argv[1]

    print('[  INFO  ] Loading dataset...')
    df = pd.read_csv(data_file, index_col=0)
    X = df.drop('label', axis=1).values
    dxs = pd.Series(
        pd.Categorical(df['label'], categories=['MCInc', 'MCIc'], ordered=False),
        index=df.index)
    y = dxs.cat.codes
    print('\t- Categories: %s' % dxs.cat.categories)
    print('[  INFO  ] Dataset stats:\n%s' % dxs.value_counts())
    print('[  INFO  ] Dataset Preview \n%s' % dxs.head())

    # Split data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Start Classifying: SVM-RBF
    svm = SVC(probability=True)

    # Set up the parameters to evaluate
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': np.logspace(-3, 2, 10),
        'gamma': np.logspace(-3, 2, 10)
    }

    clf_grid = GridSearchCV(svm, param_grid, cv=10, iid=False)
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
    plt.legend(['AUC = ' + str(auc)])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    # plt.title('Breast cancer classification ROC curve')

    fig_file = join(dirname(data_file), basename(data_file)[:-4] + '_roc.pdf')
    plt.savefig(fig_file, bbox_inches='tight')
    # plt.show()


