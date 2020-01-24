import pytest

from ml.dataset import DataSet
from ml.learning import *


def test_linear_learner():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    n_samples, n_features = len(iris.examples), iris.target
    X, y = np.array([x[:n_features] for x in iris.examples]), \
           np.array([x[n_features] for x in iris.examples])
    ll = LinearRegressionLearner().fit(X, y)
    tests = [([[5.0, 3.1, 0.9, 0.1]], 0),
             ([[5.1, 3.5, 1.0, 0.0]], 0),
             ([[4.9, 3.3, 1.1, 0.1]], 0),
             ([[6.0, 3.0, 4.0, 1.1]], 1),
             ([[6.1, 2.2, 3.5, 1.0]], 1),
             ([[5.9, 2.5, 3.3, 1.1]], 1),
             ([[7.5, 4.1, 6.2, 2.3]], 2),
             ([[7.3, 4.0, 6.1, 2.4]], 2),
             ([[7.0, 3.3, 6.1, 2.5]], 2)]
    assert grade_learner(ll, tests)
    assert err_ratio(ll, X, y)


def test_logistic_learner():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    n_samples, n_features = len(iris.examples), iris.target
    X, y = np.array([x[:n_features] for x in iris.examples]), \
           np.array([x[n_features] for x in iris.examples])
    ll = BinaryLogisticRegressionLearner().fit(X, y)
    tests = [([[5.0, 3.1, 0.9, 0.1]], 0),
             ([[5.1, 3.5, 1.0, 0.0]], 0),
             ([[4.9, 3.3, 1.1, 0.1]], 0),
             ([[6.0, 3.0, 4.0, 1.1]], 1),
             ([[6.1, 2.2, 3.5, 1.0]], 1),
             ([[5.9, 2.5, 3.3, 1.1]], 1),
             ([[7.5, 4.1, 6.2, 2.3]], 2),
             ([[7.3, 4.0, 6.1, 2.4]], 2),
             ([[7.0, 3.3, 6.1, 2.5]], 2)]
    assert grade_learner(ll, tests)
    assert err_ratio(ll, X, y)


if __name__ == "__main__":
    pytest.main()
