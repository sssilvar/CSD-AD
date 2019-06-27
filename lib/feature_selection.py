import pymrmr
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin


class MRMR(BaseEstimator):
    def __init__(self, method='MIQ', k_features=10):
        self.method = method
        self.k_features = k_features
        self.selected_features = []
        self.selected_indexes = []

    def fit(self, X, y):
        # Check if DataFrame
        X = self.check_df(X)
        y = self.check_df(y)

        # Compose new DataFrame
        feat_cols = [f'feat_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(data=X, columns=feat_cols)
        target = pd.Series(y, name='target')
        X_df = X_df.join(target)  # Append labels to dataframe

        # Re-arrange the DataFrame so 'target' is the fisrt column
        ordered_cols = ['target'] + feat_cols
        X_df = X_df[ordered_cols]

        # Perform the feature selection using mRMR
        self.selected_features = pymrmr.mRMR(X_df, self.method, self.k_features)
        self.selected_indexes = [X_df.drop('target', axis='columns').columns.tolist().index(i) for i in self.selected_features]

        return self

    def transform(self, X):
        print('Transforming...')
        X_raw = X.copy()
        # Check if DataFrame
        X = self.check_df(X)

        # Make it one
        feat_cols = [f'feat_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(data=X, columns=feat_cols)

        print('Are the same?')
        print(np.mean(X_raw[:, self.selected_indexes] - X_df[self.selected_features].values))

        # Transform to fitted features
        return X_df[self.selected_features].values

    def get_support(self):
        return self.selected_indexes

    @staticmethod
    def check_df(X):
        if isinstance(X, pd.DataFrame):
            return X.values
        else:
            return np.array(X)
