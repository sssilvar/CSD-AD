#!/bin/env python3
import pymrmr

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if __name__ == '__main__':
    # Load dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    # Create a classification pipeline
    pipeline = Pipeline([
        ('clf', SVC(gamma='scale'))
    ])

    # Split set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Train the model using whole bunch of data
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_true=y_test, y_pred=y_pred))

    # ============================================================
    # Feature selection - mRMR
    # ============================================================
    # Convert data into a DataFrame
    feat_cols = [f'pix_{i}' for i in range(X_train.shape[1])]
    X_train_df = pd.DataFrame(data=X_train, columns=feat_cols)
    X_test_df = pd.DataFrame(data=X_test, columns=feat_cols)
    target_series = pd.Series(y_train, name='target')
    X_train_df = X_train_df.join(target_series)  # Add 'target' column

    # Order columns of DataFrame
    ordered_cols = ['target'] + [i for i in X_train_df.columns if 'target' not in i]
    df_ordered = X_train_df[ordered_cols]
    print(df_ordered.head())

    # Do the magic!
    selected_cols = pymrmr.mRMR(df_ordered, 'MIQ', 30)
    X_train_sel = df_ordered[selected_cols]
    X_test_sel = X_test_df[selected_cols]

    print(f'Fitting model. Dimensions: {X_train_sel.shape}')
    pipeline.fit(X_train_sel, y_train)
    print(f'Testing model. Dimensions: {X_test_sel.shape}')
    y_pred = pipeline.predict(X_test_sel)
    print(classification_report(y_true=y_test, y_pred=y_pred))
