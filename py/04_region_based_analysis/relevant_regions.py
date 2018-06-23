from __future__ import print_function
import os
import time
import logging

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import numpy as np
import pandas as pd
# from tabulate import tabulate
import matplotlib.pyplot as plt

# Matplotlib setup
plt.switch_backend('agg')
plt.style.use('ggplot')

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set root folder
up = os.path.dirname
root = up(up(up(os.path.realpath(__file__))))

# create a file handler
handler = logging.FileHandler(os.path.join(root, 'output', 'classification.log'))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info = print


if __name__ == '__main__':
    # Load data
    csv_file = os.path.join(root, 'features', 'curvelets', 'curvelet_gmm_3_comp.csv')
    logger.info('[  INFO  ] Datafile located at: ', csv_file)
    logger.info('[  INFO  ] Loading file ...')
    df = pd.read_csv(csv_file, index_col=0)
    logger.info('[  INFO  ] Done!')

    df = df.fillna(0)
    df = df.reset_index(drop=True)

    feature_names = df.drop(['target', 'sid'], axis=1).columns

    # # Split features and labels
    X = df.drop(['target', 'sid'], axis=1).values
    y = np.array(df['target'] == 'MCIc', dtype=np.uint8)

    # Split the data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # ==== LEAVE ONE OUT CV ====
    accuracy = []
    regions = []
    y_tests = []
    precisions = []

    from sklearn.model_selection import LeaveOneOut
    for train_index, test_index in LeaveOneOut().split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        logger.info('\n[  INFO  ] Feature selection ...')

        # from scipy.stats import ttest_ind
        # ix = []
        # for i, col in enumerate(X_train.T):
        #     test = ttest_ind(col[y_train == 1], col[y_train == 0])
        #
        #     if test.pvalue < 0.05:
        #         # logger.info('SIGNIFICANT ROI:', feature_names[i])
        #         ix.append(i)
        #         # plt.figure()
        #         # # plt.boxplot([col[y_train == 1], col[y_train == 0]])
        #         # plt.hist(col[y_train == 1], alpha=0.7, bins=30)
        #         # plt.hist(col[y_train == 0], alpha=0.7, bins=30)
        #         # plt.legend(['MCIc', 'MCInc'])
        #         # plt.title(feature_names[i])
        #         # plt.savefig(os.path.join(
        #         #     '/user/ssilvari/home/Documents/output/feature-behavior/11/', feature_names[i] + '.png'))
        #         # plt.close()
        #         # # plt.show()

        # Feature selection
        lasso = Lasso()
        lasso.fit(X_train, y_train)

        ix = lasso.coef_ != 0
        coefs = lasso.coef_[ix]
        features_selected = feature_names[ix]

        # Update features selected
        X_train, X_test = X_train[:, ix], X_test[:, ix]
        logger.info(X_train.shape, X_test.shape)

        # Print relevant ROIs
        # logger.info('[  OK  ] Relevant features:\n')
        # logger.info(tabulate(zip(features_selected, coefs), headers=['ROI', 'Alpha'], tablefmt='grid'))
        regions.append(features_selected)
        logger.info('[  OK  ] Feature selection done!')

        # ================ CLASSIFICATION ================
        logger.info('\n\n[  INFO  ] Starting Classification')
        # Set pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            # ('scaler', StandardScaler(with_mean=False)),
            # ('mutual_info', SelectKBest(mutual_info_classif, k=10)),
            ('knn', KNeighborsClassifier(n_neighbors=9, algorithm='ball_tree', weights='uniform', p=1, n_jobs=3)),
            # ('svm', SVC(kernel='rbf', gamma=0.001, C=1, probability=True))
        ])

        # Set grid of parameters: grid_param
        # param_grid = [
        #     {'svm__C': [1, 10, 100, 1000], 'svm__kernel': ['linear']},
        #     {'svm__C': [1, 10, 100, 1000], 'svm__gamma': [0.001, 0.0001], 'svm__kernel': ['rbf']},
        # ]
        param_grid = {
            'knn__n_neighbors': range(3, 10),
            'knn__weights': ['uniform', 'distance'],
            'knn__algorithm': ['ball_tree', 'kd_tree', 'brute'],
            'knn__p': [1, 2],
        }

        pipeline = GridSearchCV(
                        pipeline,
                        param_grid,
                        scoring='accuracy',
                        cv=20,
                        n_jobs=-1)

        # Fit model
        logger.info('Fitting model ...')
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        logger.info('Classification report: \n {}'.format(classification_report(y_test, y_pred)))
        # logger.info('Score: {}'.format(pipeline.score(X_test, y_test)))
        logger.info('Best Params: {}'.format(pipeline.best_params_))

        accuracy.append(y_pred_proba)
        y_tests.append(y_test)
        precisions.append(y_pred == y_test)

    logger.info('\n\n[  OK  ] FINAL REPORT')
    logger.info('\t - Mean accuracy: ', np.mean(accuracy))
    logger.info('\t - Precision: %d/%d' % (np.sum(precisions), len(precisions)))

    # Compute AUC
    fpr, tpr, thresholds = roc_curve(y_tests, accuracy)
    auc = roc_auc_score(y_tests, accuracy)

    plt.figure(figsize=(19.2, 10.8), dpi=150)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.legend(['AUC = ' + str(auc)])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Classification ROC curve')
    plt.savefig(os.path.join(root, 'output', os.path.basename(csv_file) + '.png'))
    # plt.show()

    logger.info('[  DONE  ]')
