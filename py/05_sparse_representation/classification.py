__author__ = "Santiago Silva"
__copyright__ = "Copyright 2018"
__description__ = """
Classification from curvelet features. USAGE:
    python3 classification.py -f [features_file.h5] -c [common_data.csv]

Curvelet DataFrame column names are structured as follows:
    [radius (vox)]_[scale]_angle
"""

import os
from os.path import dirname as up
import argparse
from tabulate import tabulate

root = up(up(up(os.path.realpath(__file__))))  # Root folder of the software

import numpy as np
import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import *

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def main():
    # Load features DataFrame: fdf
    fdf = pd.read_hdf(features_file, key='features')

    # Common data DataFrame: cdf
    # It is necessary to keep just the baseline info (bl)
    # And just the subjects involved in the studio
    cdf = pd.read_csv(common_data_file, index_col='PTID')
    cdf = cdf[cdf['VISCODE'] == 'bl']
    cdf = cdf.loc[fdf['subject'].tolist()]
    cdf = cdf.dropna(axis=1, how='any')
    cdf['GENDER'] = cdf['PTGENDER'].astype('category').cat.codes
    cdf = cdf.drop(['COLPROT', 'EXAMDATE', 'FSVERSION.bl', 'IMAGEUID.bl'], axis=1)
    print(cdf.columns.tolist())
    print(cdf.head())

    X = fdf.drop(['subject', 'target'], axis=1)
    y = fdf['target'].values

    cols = np.array(X.columns.tolist())
    
    # Scale the data
    print('Scaling data...')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = X.values
    print(X_scaled.shape)

    # Compensate the data with common data
    X = X_scaled

    C = cdf[['AGE', 'GENDER', 'APOE4', 'PTEDUCAT', 'CDRSB', 'MMSE']].values
    w = np.linalg.inv(C.T.dot(C)).dot(C.T.dot(X))
    X = X - C.dot(w)

    kf = StratifiedKFold(n_splits=10)

    results = {
        'auc': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f-score': [],
        ''
    }
    for train_index, test_index in kf.split(X, y=y):
        # Cross validation
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(y_train.shape)
        # Feature selection
        print('Selecting features...')
        
        # Option 1: Lasso
        # lasso = Lasso(normalize=False)
        # lasso.fit(X_train, y_train)

        # ix = abs(lasso.coef_) > 0
        # coefs = lasso.coef_[ix]

        # # Option 2: t-Test
        from scipy.stats import ttest_ind
        ix = []
        coefs = []
        for i, col in enumerate(X_train.T):
            a, b = col[y_train == 1], col[y_train == 0]
            test = ttest_ind(a, b, equal_var=True)
        
            if test.pvalue < 0.05:
                # logger.info('SIGNIFICANT ROI:', feature_names[i])
                coefs.append(test)
                ix.append(i)

                # plt.hist(a, alpha=0.7)
                # plt.hist(b, alpha=0.7)
                # plt.title('Feature histogram: %s' % cols[i])
                # plt.show()

        X_train = X_train[:, ix]
        X_test = X_test[:, ix]
        
        print(tabulate(zip(cols[ix], coefs), headers=['radius_scale_angle', 'Coefficient'], tablefmt='grid'))

        # ########################################
        # Classification
        # ########################################
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rfc', RandomForestClassifier(random_state=42, criterion='gini', max_depth=5, max_features='log2', n_estimators=200))
            # ('rfc', RandomForestClassifier())
        ])

        # Hypertunning Grid Search
        # param_grid = {
        #     'rfc__n_estimators': [200, 500],
        #     'rfc__max_features': ['auto', 'sqrt', 'log2'],
        #     'rfc__max_depth': [5, 8],
        #     'rfc__criterion': ['gini', 'entropy']
        # }

        # pipeline = GridSearchCV(
        #     pipeline,
        #     param_grid,
        #     scoring='accuracy',
        #     cv=10,
        #     n_jobs=2
        # )

        # Fit model
        print('Fitting model ...')
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        score = pipeline.predict_proba(X_test)
        y_pred_prob = score[:, 1]

        # ########################################
        # Analyze output
        # ########################################
        # print('Best params: {}'.format(pipeline.best_params_))

        # Compute AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        auc = roc_auc_score(y_test, y_pred_prob)
        print('\nAUC: %.2f' % auc)
        
        plt.figure(figsize=(19.2 * 0.75, 10.8 * 0.75), dpi=150)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.legend(['AUC = ' + str(auc)])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        
        # ext = '.png'
        # roc_plot_file = os.path.join(root, 'output', 'reg_class', '%04d_roc_%s_%d_comp' %
        #                              (rid, str(region).replace('-', '_').lower(), n_comp))
        # plt.savefig(roc_plot_file + ext, bbox_inches='tight')
        # plt.show()

        # Append experiment results
        results['auc'].append(auc)
        results['accuracy'].append(accuracy_score(y_test, y_pred))
        results['precision'].append(average_precision_score(y_test, y_pred_prob))
        results['recall'].append(recall_score(y_test, y_pred))
        results['f-score'].append(f1_score(y_test, y_pred))

    # Print the results
    rdf = pd.DataFrame.from_dict(results)
    print(rdf)
    print(rdf.mean())



if __name__ == '__main__':
    # --- ARG PARSING ---
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('-f', metavar='--file',
                        help='Features file',
                        default='/user/ssilvari/home/Documents/structural/CSD-AD/output/curvelet_spherical_gradients/spherical_curvelet_features.h5')
    args = parser.parse_args()

    features_file = args.f
    common_data_file = os.path.join(root, 'param', 'common', 'adnimerge.csv')

    os.system('clear')

    # Run main program
    main()