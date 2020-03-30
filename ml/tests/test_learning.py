import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ml.learning import LinearRegressionLearner, MultiClassClassifier, LogisticRegressionLearner, MultiTargetRegressor
from ml.metrics import accuracy_score, mean_euclidean_error
from optimization.unconstrained.quasi_newton import BFGS
from utils import load_ml_cup


def test_linear_learner():
    X, y = load_ml_cup()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    ll = MultiTargetRegressor(LinearRegressionLearner(optimizer=BFGS)).fit(X_train, y_train)
    assert mean_euclidean_error(ll.predict(X_test), y_test) <= 1.3


def test_logistic_learner():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    ll = MultiClassClassifier(LogisticRegressionLearner(optimizer=BFGS)).fit(X_train, y_train)
    assert accuracy_score(ll.predict(X_test), y_test) >= 0.89


if __name__ == "__main__":
    pytest.main()
