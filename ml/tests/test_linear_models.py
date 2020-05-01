import numpy as np
import pytest
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from ml.linear_models import LinearRegression, LogisticRegression
from ml.regularizers import L2
from optimization.unconstrained.line_search.quasi_newton import BFGS


def test_linear_regression():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    lr = LinearRegression(optimizer=BFGS).fit(X_train, y_train)
    assert lr.score(X_test, y_test) >= 0.55
    assert np.allclose(lr.coef_, np.linalg.lstsq(X_train, y_train)[0], rtol=0.1)


def test_ridge_regression():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    lmbda = 10
    ridge = LinearRegression(optimizer=BFGS, regularizer=L2(lmbda)).fit(X_train, y_train)
    # assert ridge.score(X_test, y_test) >= 0.65
    assert np.allclose(ridge.coef_, np.linalg.inv(X_train.T.dot(X_train) + np.identity(X.shape[1]) *
                                                  lmbda).dot(X_train.T).dot(y_train), rtol=0.1)


def test_logistic_regression():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    ll = OneVsRestClassifier(LogisticRegression(optimizer=BFGS)).fit(X_train, y_train)
    assert ll.score(X_test, y_test) >= 0.89


if __name__ == "__main__":
    pytest.main()
