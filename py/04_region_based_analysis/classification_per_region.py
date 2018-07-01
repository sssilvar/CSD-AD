from __future__ import print_function
import os

import logging
from tabulate import tabulate

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import matplotlib.pyplot as plt

# Matplotlib setup
plt.switch_backend('agg')
plt.style.use('ggplot')


# Set App root folder
up = os.path.dirname
root = up(up(up(os.path.realpath(__file__))))

# SetUp and start logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.FileHandler(os.path.join(root, 'output', 'classification_per_regions.log'))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

if __name__ == '__main__':
    # Set number of components
    components = [3, 7, 11]
    for n_comp in components:
        # Load region and feature files
        features_file = os.path.join(root, 'features', 'curvelets', 'curvelet_gmm_%d_comp.csv' % n_comp)
        regions = os.path.join(root, 'param', 'FreeSurferColorLUT.csv')
        dataset_file = os.path.join(root, 'param', 'data_df.csv')

        logger.info('Features file located at: ' + features_file)
        logger.info('Loading file ...')

        df_features = pd.read_csv(features_file, index_col=0)
        df_regions = pd.read_csv(regions, index_col=['region_id'])
        df_dataset = pd.read_csv(dataset_file, index_col=0)
        logger.info('Done!')

        # Get regions in feature matrix
        for rid, region in zip(df_regions.index, df_regions['label_name']):
            # Reset statistics
            accuracy = []
            y_tests = []
            precisions = []

            # Look for the features corresponding to the current region
            region_feats_found = [feat for feat in df_features.columns if region in feat]

            if region_feats_found and rid in [10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58,]:
                logger.info('Processing region: ' + region)

                # Create X and y
                X = df_features[region_feats_found].values
                y = np.array(df_features['target'] == 'MCIc', dtype=np.uint8)

                for train_index, test_index in LeaveOneOut().split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    # ============ FEATURE SELECTION ============
                    lasso = Lasso(max_iter=100000, tol=0.001)
                    lasso.fit(X_train, y_train)

                    ix = lasso.coef_ != 0
                    coefs = lasso.coef_[ix]
                    features_selected = np.array(region_feats_found)[ix]
                    # logger.info(tabulate(zip(features_selected, coefs), headers=['ROI', 'Alpha'], tablefmt='grid'))

                    # ============ CLASSIFICATION ============
                    # logger.info('\n\n[  INFO  ] Starting Classification')
                    # Set pipeline
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        # ('scaler', StandardScaler(with_mean=False)),
                        # ('mutual_info', SelectKBest(mutual_info_classif, k=10)),
                        # ('knn', KNeighborsClassifier()),
                        # ('knn', KNeighborsClassifier(n_neighbors=8, algorithm='ball_tree', weights='uniform', p=1, n_jobs=-1)),
                        # ('svm', SVC(probability=True)),
                        # ('svm', SVC(C=1, gamma=0.001, kernel='rbf', probability=True)),
                        ('rfc', RandomForestClassifier(random_state=42)),
                        # ('rfc', RandomForestClassifier(criterion='gini', max_depth=8, max_features='auto', n_estimators=200, n_jobs=-1)),
                    ])

                    # Set grid of parameters: grid_param
                    # param_grid = [
                    #     {'svm__C': [1, 10, 100, 1000], 'svm__kernel': ['linear']},
                    #     {'svm__C': [1, 10, 100, 1000], 'svm__gamma': [0.001, 0.0001], 'svm__kernel': ['rbf']},
                    # ]
                    # param_grid = {
                    #     'knn__n_neighbors': range(3, 10),
                    #     'knn__weights': ['uniform', 'distance'],
                    #     'knn__algorithm': ['ball_tree', 'kd_tree', 'brute'],
                    #     'knn__p': [1, 2],
                    # }
                    param_grid = {
                        'rfc__n_estimators': [200, 500],
                        'rfc__max_features': ['auto', 'sqrt', 'log2'],
                        'rfc__max_depth': [4, 5, 6, 7, 8],
                        'rfc__criterion': ['gini', 'entropy']
                    }

                    pipeline = GridSearchCV(
                                    pipeline,
                                    param_grid,
                                    scoring='accuracy',
                                    cv=10,
                                    n_jobs=-1)

                    # Fit model
                    # logger.info('Fitting model ...')
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

                    # Get Score
                    pipeline_score = pipeline.score(X_test, y_test)

                    # logger.info('Classification report: \n {}'.format(classification_report(y_test, y_pred)))
                    # logger.info('Score: {}'.format(pipeline_score))
                    # logger.info('Best params: {}'.format(pipeline.best_params_))
                    if pipeline_score > 0.5:
                        logger.info('Score: %.2f | Predicted probability: %.2f' % (pipeline_score, y_pred_proba))
                        logger.info('Best params: {}'.format(pipeline.best_params_))

                    accuracy.append(y_pred_proba)
                    y_tests.append(y_test)
                    precisions.append(y_pred == y_test)

                # Report after classifying each region
                logger.info('\n\n[  OK  ] FINAL REPORT')
                logger.info('\t - Mean accuracy: %f ' % np.mean(accuracy))
                logger.info('\t - Precision: %d/%d' % (np.sum(precisions), len(precisions)))

                # Compute AUC
                fpr, tpr, thresholds = roc_curve(y_tests, accuracy)
                auc = roc_auc_score(y_tests, accuracy)

                plt.figure(figsize=(19.2 * 0.75, 10.8 * 0.75), dpi=150)
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(fpr, tpr)
                plt.legend(['AUC = ' + str(auc)])
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title('ROC curve: %s' % region)

                ext = '.png'
                roc_plot_file = os.path.join(root, 'output', 'reg_class', '%04d_roc_%s_%d_comp' %
                                             (rid, str(region).replace('-', '_').lower(), n_comp))
                plt.savefig(roc_plot_file + ext, bbox_inches='tight')
                # plt.show()

                # Save results
                np.savez(roc_plot_file + '.npz', accuracy, precisions, y_tests)

                logger.info('Region %s is done!\nROC curve plot save at: %s' % (region, roc_plot_file + ext))
                print('Region %s is done!\nROC curve plot save at: %s' % (region, roc_plot_file + ext))


