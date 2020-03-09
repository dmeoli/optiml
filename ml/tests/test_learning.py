import numpy as np
import pytest
from sklearn.datasets import load_iris, load_boston

from ml.learning import (LinearRegressionLearner, LinearModelLossFunction,
                         MultiClassClassifier, LogisticRegressionLearner)
from ml.losses import mean_squared_error
from ml.metrics import accuracy_score
from optimization.unconstrained.quasi_newton import BFGS


def test_linear_learner():
    X, y = load_boston(return_X_y=True)
    ll = LinearRegressionLearner(optimizer=BFGS).fit(X, y)
    assert np.allclose(ll.w, LinearModelLossFunction(X, y, ll, mean_squared_error).x_star(), rtol=1e-4)
    assert mean_squared_error(y, ll.predict(X)) <= 24.17


def test_logistic_learner():
    X, y = load_iris(return_X_y=True)
    ll = MultiClassClassifier(LogisticRegressionLearner(optimizer=BFGS)).fit(X, y)
    assert accuracy_score(y, ll.predict(X)) >= 0.96


if __name__ == "__main__":
    pytest.main()
