import numpy as np
import pytest

from ml.datasets import DataSet
from ml.learning import err_ratio, grade_learner
from ml.neural_network import NeuralNetLearner, PerceptronLearner, adam, stochastic_gradient_descent


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
    nnl_adam = NeuralNetLearner([4], learning_rate=0.001, epochs=200, optimizer=adam)
    nnl_adam.fit(X, y)
    assert nnl_adam.predict([[5.0, 3.1, 0.9, 0.1]]) == 0
    assert nnl_adam.predict([[5.1, 3.5, 1.0, 0.0]]) == 0
    assert nnl_adam.predict([[4.9, 3.3, 1.1, 0.1]]) == 0
    assert nnl_adam.predict([[6.0, 3.0, 4.0, 1.1]]) == 1
    assert nnl_adam.predict([[6.1, 2.2, 3.5, 1.0]]) == 1
    assert nnl_adam.predict([[5.9, 2.5, 3.3, 1.1]]) == 1
    assert nnl_adam.predict([[7.5, 4.1, 6.2, 2.3]]) == 2
    assert nnl_adam.predict([[7.3, 4.0, 6.1, 2.4]]) == 2
    assert nnl_adam.predict([[7.0, 3.3, 6.1, 2.5]]) == 2
    assert grade_learner(nnl_adam, tests) >= 1 / 3
    assert err_ratio(nnl_adam, iris) < 0.21
    nnl_gd = NeuralNetLearner([4], learning_rate=0.15, epochs=100, optimizer=stochastic_gradient_descent)
    nnl_gd.fit(X, y)
    assert nnl_gd.predict([[5.0, 3.1, 0.9, 0.1]]) == 0
    assert nnl_gd.predict([[5.1, 3.5, 1.0, 0.0]]) == 0
    assert nnl_gd.predict([[4.9, 3.3, 1.1, 0.1]]) == 0
    assert nnl_gd.predict([[6.0, 3.0, 4.0, 1.1]]) == 1
    assert nnl_gd.predict([[6.1, 2.2, 3.5, 1.0]]) == 1
    assert nnl_gd.predict([[5.9, 2.5, 3.3, 1.1]]) == 1
    assert nnl_gd.predict([[7.5, 4.1, 6.2, 2.3]]) == 2
    assert nnl_gd.predict([[7.3, 4.0, 6.1, 2.4]]) == 2
    assert nnl_gd.predict([[7.0, 3.3, 6.1, 2.5]]) == 2
    assert grade_learner(nnl_gd, tests) >= 1 / 3
    assert err_ratio(nnl_gd, iris) < 0.21


def test_perceptron():
    iris = DataSet(name='iris')
    classes = ['setosa', 'versicolor', 'virginica']
    iris.classes_to_numbers(classes)
    n_samples, n_features = len(iris.examples), iris.target
    X, y = np.array([x[:n_features] for x in iris.examples]), \
           np.array([x[n_features] for x in iris.examples])
    pl = PerceptronLearner(learning_rate=0.01, epochs=100)
    pl.fit(X, y)
    tests = [([5, 3, 1, 0.1], 0),
             ([5, 3.5, 1, 0], 0),
             ([6, 3, 4, 1.1], 1),
             ([6, 2, 3.5, 1], 1),
             ([7.5, 4, 6, 2], 2),
             ([7, 3, 6, 2.5], 2)]
    assert pl.predict([[5.0, 3.1, 0.9, 0.1]]) == 0
    assert pl.predict([[5.1, 3.5, 1.0, 0.0]]) == 0
    assert pl.predict([[4.9, 3.3, 1.1, 0.1]]) == 0
    assert pl.predict([[6.0, 3.0, 4.0, 1.1]]) == 1
    assert pl.predict([[6.1, 2.2, 3.5, 1.0]]) == 1
    assert pl.predict([[5.9, 2.5, 3.3, 1.1]]) == 1
    assert pl.predict([[7.5, 4.1, 6.2, 2.3]]) == 2
    assert pl.predict([[7.3, 4.0, 6.1, 2.4]]) == 2
    assert pl.predict([[7.0, 3.3, 6.1, 2.5]]) == 2
    assert grade_learner(pl, tests) > 1 / 2
    assert err_ratio(pl, iris) < 0.4


if __name__ == "__main__":
    pytest.main()
