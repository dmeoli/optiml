import numpy as np
import pytest

from ml.dataset import DataSet
from ml.learning import WeightedLearner, ada_boost, grade_learner, err_ratio, LinearRegressionLearner
from ml.neural_network.neural_network import PerceptronLearner


def test_ada_boost():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    n_samples, n_features = len(iris.examples), iris.target
    X, y = np.array([x[:n_features] for x in iris.examples]), \
           np.array([x[n_features] for x in iris.examples])
    wl = WeightedLearner(PerceptronLearner)
    ab = ada_boost(wl, X, y, 5)
    tests = [([5, 3, 1, 0.1], 0),
             ([5, 3.5, 1, 0], 0),
             ([6, 3, 4, 1.1], 1),
             ([6, 2, 3.5, 1], 1),
             ([7.5, 4, 6, 2], 2),
             ([7, 3, 6, 2.5], 2)]
    assert grade_learner(ab, tests) > 2 / 3
    assert err_ratio(ab, X, y) < 0.25


def test_linear_learner():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    n_samples, n_features = len(iris.examples), iris.target
    X, y = np.array([x[:n_features] for x in iris.examples]), \
           np.array([x[n_features] for x in iris.examples])[:, np.newaxis]
    ll = LinearRegressionLearner().fit(X, y)


if __name__ == "__main__":
    pytest.main()
