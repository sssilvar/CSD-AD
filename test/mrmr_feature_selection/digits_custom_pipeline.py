#!/bin/env python
from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from lib.feature_selection import MRMR

if __name__ == '__main__':
    # Load dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    # Create pipelines
    pipeline_a = Pipeline([
        ('clf', SVC(gamma='scale'))
    ])
    pipeline_b = Pipeline([
        ('selector', MRMR(method='MIQ', k_features=30)),
        ('clf', SVC(gamma='scale'))
    ])

    # Do the magic
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    # Train models
    print('Training model A (no selector)')
    pipeline_a.fit(X_train, y_train)
    print('Training model B (with selector)')
    pipeline_b.fit(X_train, y_train)

    # Test models
    print('Predicting model A')
    y_pred_a = pipeline_a.predict(X_test)
    print('Predicting with model B')
    y_pred_b = pipeline_b.predict(X_test)

    # Print results
    sep = '=' * 10
    print(f'{sep} PIELINE A (NO MRMR) {sep}')
    print(classification_report(y_true=y_test, y_pred=y_pred_a))
    print(f'{sep} PIELINE B (WITH MRMR) {sep}')
    print(classification_report(y_true=y_test, y_pred=y_pred_b))

