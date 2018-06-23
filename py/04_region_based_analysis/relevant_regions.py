from __future__ import print_function
import os
import time

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

plt.switch_backend('agg')
plt.style.use('ggplot')


# Set root folder
up = os.path.dirname
root = up(up(up(os.path.realpath(__file__))))

if __name__ == '__main__':
    # Load data
    print('[  INFO  ] Loading file ...')
    csv_file = os.path.join(root, 'features', 'curvelets', 'curvelet_gmm_3_comp.csv')
    df = pd.read_csv(csv_file, index_col=0)
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

        # from scipy.stats import ttest_ind
        # ix = []
        # for i, col in enumerate(X_train.T):
        #     test = ttest_ind(col[y_train == 1], col[y_train == 0])
        #
        #     if test.pvalue < 0.05:
        #         # print('SIGNIFICANT ROI:', feature_names[i])
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
        print(X_train.shape, X_test.shape)

        # Print relevant ROIs
        # print('[  OK  ] Relevant features:\n')
        # print(tabulate(zip(features_selected, coefs), headers=['ROI', 'Alpha'], tablefmt='grid'))
        regions.append(features_selected)

        # ================ CLASSIFICATION ================
        print('\n\n[  INFO  ] Starting Classification')
        # Set pipeline
        pipeline = Pipeline([
            # ('scaler', StandardScaler()),
            # ('scaler', StandardScaler(with_mean=False)),
            # ('mutual_info', SelectKBest(mutual_info_classif, k=10)),
            # ('knn', KNeighborsClassifier(n_neighbors=9, algorithm='ball_tree', weights='uniform', p=1, n_jobs=3)),
            ('svm', SVC(kernel='rbf', probability=True))
        ])

        # Set grid of parameters: grid_param
        param_grid = {
            # 'svm__kernel': ['linear', 'poly', 'rbf'],
            # 'svm__degree': range(5),
            'svm__gamma': np.logspace(-9, 3, 13),
            # 'svm__coef0': np.logspace(0.1, 10, 4),
            'svm__C': np.logspace(-2, 10, 13)
        }
        # param_grid = {
        #     'knn__n_neighbors': range(3, 10),
        #     'knn__weights': ['uniform', 'distance'],
        #     'knn__algorithm': ['ball_tree', 'kd_tree', 'brute'],
        #     'knn__p': [1, 2],
        # }

        pipeline = GridSearchCV(
                        pipeline,
                        param_grid,
                        scoring='accuracy',
                        n_jobs=4)

        # Fit model
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        print('Classification report: \n {}'.format(classification_report(y_test, y_pred)))
        print('Score: {}'.format(pipeline.score(X_test, y_test)))
        print('Best Params: {}'.format(pipeline.best_params_))

        accuracy.append(y_pred_proba)
        y_tests.append(y_test)
        precisions.append(y_pred == y_test)

    print('\n\n[  OK  ] FINAL REPORT')
    print('\t - Mean accuracy: ', np.mean(accuracy))
    print('\t - Precision: %d/%d' % (np.sum(precisions), len(precisions)))

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
    plt.show()

    print('[  DONE  ]')
