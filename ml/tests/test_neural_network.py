import numpy as np
import pytest

from ml.dataset import DataSet
from ml.learning import err_ratio, grade_learner
from ml.neural_network.neural_network import NeuralNetLearner, PerceptronLearner, adam, stochastic_gradient_descent


def test_neural_net():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    n_samples, n_features = len(iris.examples), iris.target
    X, y = np.array([x[:n_features] for x in iris.examples]), \
           np.array([x[n_features] for x in iris.examples])
    tests = [([5.0, 3.1, 0.9, 0.1], 0),
             ([5.1, 3.5, 1.0, 0.0], 0),
             ([4.9, 3.3, 1.1, 0.1], 0),
             ([6.0, 3.0, 4.0, 1.1], 1),
             ([6.1, 2.2, 3.5, 1.0], 1),
             ([5.9, 2.5, 3.3, 1.1], 1),
             ([7.5, 4.1, 6.2, 2.3], 2),
             ([7.3, 4.0, 6.1, 2.4], 2),
             ([7.0, 3.3, 6.1, 2.5], 2)]
    nnl_adam = NeuralNetLearner([4], l_rate=0.001, epochs=200, optimizer=adam).fit(X, y)
    assert grade_learner(nnl_adam, tests) >= 1 / 3
    assert err_ratio(nnl_adam, X, y) < 0.21
    nnl_gd = NeuralNetLearner([4], l_rate=0.15, epochs=100, optimizer=stochastic_gradient_descent).fit(X, y)
    assert grade_learner(nnl_gd, tests) >= 1 / 3
    assert err_ratio(nnl_gd, X, y) < 0.21


def test_perceptron():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    n_samples, n_features = len(iris.examples), iris.target
    X, y = np.array([x[:n_features] for x in iris.examples]), \
           np.array([x[n_features] for x in iris.examples])
    pl = PerceptronLearner(l_rate=0.01, epochs=100).fit(X, y)
    tests = [([5.0, 3.1, 0.9, 0.1], 0),
             ([5.1, 3.5, 1.0, 0.0], 0),
             ([4.9, 3.3, 1.1, 0.1], 0),
             ([6.0, 3.0, 4.0, 1.1], 1),
             ([6.1, 2.2, 3.5, 1.0], 1),
             ([5.9, 2.5, 3.3, 1.1], 1),
             ([7.5, 4.1, 6.2, 2.3], 2),
             ([7.3, 4.0, 6.1, 2.4], 2),
             ([7.0, 3.3, 6.1, 2.5], 2)]
    assert grade_learner(pl, tests) > 1 / 2
    assert err_ratio(pl, X, y) < 0.4


if __name__ == "__main__":
    pytest.main()
