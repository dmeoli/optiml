import numpy as np
import pytest
from sklearn.datasets import load_iris

from ml.learning import MultiLogisticRegressionLearner, LinearRegressionLearner
from ml.losses import MeanSquaredError
from optimization.unconstrained.quasi_newton import BFGS

X, y = load_iris(return_X_y=True)
X, y = X, y


def test_linear_learner():
    ll = LinearRegressionLearner(optimizer=BFGS).fit(X, y)
    assert np.allclose(ll.w, MeanSquaredError(X, y).x_star(), rtol=1e-3)


def test_logistic_learner():
    ll = MultiLogisticRegressionLearner(optimizer=BFGS).fit(X, y)
    # assert grade_learner(ll, iris_tests) == 1
    # assert np.allclose(err_ratio(ll, X, y), 0.04)


if __name__ == "__main__":
    pytest.main()
