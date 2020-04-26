import numpy as np
import pytest
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from ml.linear_models import LinearRegression, LogisticRegression
from optimization.unconstrained.line_search.quasi_newton import BFGS


def test_linear_regression():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    ll = LinearRegression(optimizer=BFGS).fit(X_train, y_train)
    assert np.allclose(ll.w, np.linalg.lstsq(X_train, y_train)[0], rtol=0.1)


def test_logistic_regression():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    ll = OneVsRestClassifier(LogisticRegression(optimizer=BFGS)).fit(X_train, y_train)
    assert ll.score(X_test, y_test) >= 0.85


if __name__ == "__main__":
    pytest.main()
