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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import seaborn as sns
import matplotlib.pyplot as plt

# Matplotlib setup
# plt.switch_backend('agg')
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

            if region_feats_found:
                logger.info('Processing region: ' + region)

                # Create X and y
                X_df = df_features[region_feats_found]
                X = X_df.values
                y = np.array(df_features['target'] == 'MCIc', dtype=np.uint8)

                # ============ FEATURE SELECTION ============
                lasso = Lasso(max_iter=100000, tol=0.001)
                lasso.fit(X, y)

                ix = lasso.coef_ != 0
                coefs = lasso.coef_[ix]
                features_selected = np.array(region_feats_found)[ix]

                cor = X_df[features_selected].corr(method='pearson')

                try:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plt.title("Correlation Plot")
                    sns.heatmap(cor, mask=np.zeros_like(cor, dtype=np.bool),
                                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                                square=True, ax=ax)
                    plt.show()
                except Exception as e:
                    print('ERROR: ', e)
