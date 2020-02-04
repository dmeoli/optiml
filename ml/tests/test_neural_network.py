import numpy as np
import pytest

from ml.dataset import DataSet
from ml.neural_network.neural_network import NeuralNetLearner, PerceptronLearner
from ml.validation import err_ratio, grade_learner

iris_tests = [([5.0, 3.1, 0.9, 0.1], 0),
              ([5.1, 3.5, 1.0, 0.0], 0),
              ([4.9, 3.3, 1.1, 0.1], 0),
              ([6.0, 3.0, 4.0, 1.1], 1),
              ([6.1, 2.2, 3.5, 1.0], 1),
              ([5.9, 2.5, 3.3, 1.1], 1),
              ([7.5, 4.1, 6.2, 2.3], 2),
              ([7.3, 4.0, 6.1, 2.4], 2),
              ([7.0, 3.3, 6.1, 2.5], 2)]


def test_neural_net():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    n_samples, n_features = len(iris.examples), iris.target
    X, y = np.array([x[:n_features] for x in iris.examples]), \
           np.array([x[n_features] for x in iris.examples])
    nnl = NeuralNetLearner(iris, [4])
    assert grade_learner(nnl, iris_tests) == 1
    assert np.isclose(err_ratio(nnl, X, y), 0.02)


def test_perceptron():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    n_samples, n_features = len(iris.examples), iris.target
    X, y = np.array([x[:n_features] for x in iris.examples]), \
           np.array([x[n_features] for x in iris.examples])
    pl = PerceptronLearner(iris)
    assert grade_learner(pl, iris_tests) == 1
    assert np.isclose(err_ratio(pl, X, y), 0.04)


if __name__ == "__main__":
    pytest.main()
