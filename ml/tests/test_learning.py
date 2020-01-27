import numpy as np
import pytest

from ml.dataset import DataSet
from ml.learning import MultiLogisticRegressionLearner, LinearRegressionLearner
from ml.losses import MeanSquaredError


def test_linear_learner():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    n_samples, n_features = len(iris.examples), iris.target
    X, y = np.array([x[:n_features] for x in iris.examples]), \
           np.array([x[n_features] for x in iris.examples])[:, np.newaxis]
    ll = LinearRegressionLearner().fit(X, y)
    assert np.allclose(ll.w, MeanSquaredError(X, y).x_star)


def test_logistic_learner():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    n_samples, n_features = len(iris.examples), iris.target
    X, y = np.array([x[:n_features] for x in iris.examples]), \
           np.array([x[n_features] for x in iris.examples])[:, np.newaxis]
    ll = MultiLogisticRegressionLearner().fit(X, y)


if __name__ == "__main__":
    pytest.main()
