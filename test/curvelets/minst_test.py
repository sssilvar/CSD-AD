import matplotlib.pyplot as plt
import numpy as np
import pyct as ct
from sklearn.datasets import load_digits
from sklearn.linear_model import Lasso
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Set parameters
    number_of_scales = 4
    number_of_angles = 8

    A = ct.fdct2([8, 8], nbs=number_of_scales, nba=number_of_angles, ac=True, norm=False, vec=True, cpx=False)

    # Get indexes
    ix0 = A.index(0)
    ix1 = A.index(1)
    ix2 = A.index(2)
    ix3 = A.index(3)

    features = []
    for digit in X:
        img = np.reshape(digit, [8, 8])

        # Apply curvelet to the image
        f = A.fwd(img)
        features.append(f(2))

        # img_rec = A.inv(f)
        #
        # plt.imshow(img_rec, cmap='gray')
        # plt.show()

    # Cross validation
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)

    # Feature selection
    lasso = Lasso()
    lasso.fit(X_train, y_train)

    # Plot the results
    # plt.plot(lasso.coef_)
    # plt.show()

    feat_ix = np.where(lasso.coef_ != 0)
    X_train = [i[feat_ix] for i in X_train]
    X_test = [i[feat_ix] for i in X_test]

    # Classify!
    knn = KNeighborsClassifier(n_jobs=-1, weights='distance')

    # Set a pipeline
    pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_jobs=-1, weights='distance'))
    ])

    # Set up the parameters to evaluate
    param_grid = {
        'knn__n_neighbors': range(5, 51)
    }

    pipeline = GridSearchCV(pipeline, param_grid)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print('Classification report: \n {}'.format(classification_report(y_test, y_pred)))
    print('Score: {}'.format(pipeline.score(X_test, y_test)))
    print('Best Params: {}'.format(pipeline.best_params_))

    plt.show()
