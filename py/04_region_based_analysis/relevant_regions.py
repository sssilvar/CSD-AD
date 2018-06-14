from __future__ import print_function
import os

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# Set root folder
up = os.path.dirname
root = up(up(up(os.path.realpath(__file__))))

if __name__ == '__main__':
    # Load data
    df = pd.read_csv(os.path.join(root, 'features', 'gmm_features.csv'), index_col=0)
    df = df.fillna(0)
    feature_names = df.drop(['target', 'sid'], axis=1).columns

    # # Split features and labels
    X = df.drop(['target', 'sid'], axis=1).values
    y = np.array(df['target'] == 'MCIc', dtype=np.uint8)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    # Feature selection
    lasso = Lasso()
    lasso.fit(X_train, y_train)

    ix = lasso.coef_ != 0
    coefs = lasso.coef_[ix]
    features_selected = feature_names[ix]

    # Update features selected
    X_train, X_test = X_train[:, ix], X_test[:, ix]

    # Print relevant ROIs
    print('[  OK  ] Relevant features:\n')
    print(tabulate(zip(features_selected, coefs), headers=['ROI', 'Alpha'], tablefmt='grid'))

    # ================ CLASSIFICATION ================
    print('\n\n[  INFO  ] Starting Classification')
    # Set pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_jobs=-1))
        # ('svm', SVC(probability=True))
    ])

    # Set grid of parameters: grid_param
    # param_grid = {
    #     'svm__kernel': ['poly', 'rbf', 'sigmoid'],
    #     'svm__degree': range(5),
    #     'svm__coef0': np.logspace(0.1, 2, 10),
    #     'svm__C': np.logspace(0.1, 2, 10)
    # }
    param_grid = {
        'knn__n_neighbors': range(5, 20),
        'knn__weights': ['uniform', 'distance'],
        'knn__algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'knn__p': [1, 2],
    }

    pipeline = GridSearchCV(pipeline, param_grid, scoring='roc_auc')

    # Fit model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    print('Classification report: \n {}'.format(classification_report(y_test, y_pred)))
    print('Score: {}'.format(pipeline.score(X_test, y_test)))
    print('Best Params: {}'.format(pipeline.best_params_))

    # Compute AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.legend(['AUC = ' + str(auc)])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Breast cancer classification ROC curve')
    plt.show()
