import numpy as np
import pytest
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split

from ml.learning import (LinearRegressionLearner, LinearModelLossFunction,
                         MultiClassClassifier, LogisticRegressionLearner)
from ml.losses import mean_squared_error
from ml.metrics import accuracy_score, r2_score
from optimization.unconstrained.quasi_newton import BFGS


def test_linear_learner():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
    ll = LinearRegressionLearner(optimizer=BFGS).fit(X_train, y_train)
    assert np.allclose(ll.w, LinearModelLossFunction(X_train, y_train, ll, mean_squared_error).x_star(), rtol=0.1)
    assert r2_score(ll.predict(X_test), y_test) >= 0.6


def test_logistic_learner():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
    ll = MultiClassClassifier(LogisticRegressionLearner(optimizer=BFGS)).fit(X_train, y_train)
    assert accuracy_score(ll.predict(X_test), y_test) >= 0.9


if __name__ == "__main__":
    pytest.main()
