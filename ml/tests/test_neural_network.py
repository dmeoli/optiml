import pytest
from sklearn.datasets import load_iris, load_boston
from sklearn.metrics import accuracy_score, mean_squared_error

from ml.losses import MeanSquaredError, CrossEntropy
from ml.neural_network.activations import Sigmoid, Tanh
from ml.neural_network.neural_network import NeuralNetwork
from optimization.unconstrained.quasi_newton import BFGS


def test_neural_network_regression():
    X, y = load_boston(return_X_y=True)
    net = NeuralNetwork(hidden_layer_sizes=(5, 3),
                        activations=(Tanh, Tanh),
                        optimizer=BFGS, max_iter=1000)
    net.fit(X, y, loss=MeanSquaredError, optimizer=BFGS, epochs=1000)
    assert mean_squared_error(y, net.predict(X)) <= 433.25


def test_neural_network_classification():
    X, y = load_iris(return_X_y=True)
    net = NeuralNetwork(hidden_layer_sizes=(4, 4),
                        activations=(Sigmoid, Sigmoid),
                        optimizer=BFGS, max_iter=1000)
    net.fit(X, y, loss=CrossEntropy, optimizer=BFGS, epochs=1000)
    assert accuracy_score(y, net.predict(X)) >= 0.97


if __name__ == "__main__":
    pytest.main()
