import pytest
from sklearn.datasets import load_iris, load_boston
from sklearn.metrics import accuracy_score, mean_squared_error

from ml.losses import MeanSquaredError, CrossEntropy
from ml.neural_network.activations import Sigmoid, Softmax, Linear, Tanh
from ml.neural_network.layers import Dense
from ml.neural_network.neural_network import NeuralNetwork
from optimization.unconstrained.quasi_newton import BFGS


def test_neural_network_regression():
    X, y = load_boston(return_X_y=True)
    net = NeuralNetwork(Dense(13, 5, Tanh()),
                        Dense(5, 3, Tanh()),
                        Dense(3, 1, Linear()))
    net.fit(X, y, loss=MeanSquaredError, optimizer=BFGS, epochs=100)
    assert mean_squared_error(y, net.predict(X)) <= 433.25


def test_neural_network_classification():
    X, y = load_iris(return_X_y=True)
    net = NeuralNetwork(Dense(4, 4, Sigmoid()),
                        Dense(4, 4, Sigmoid()),
                        Dense(4, 3, Softmax()))
    net.fit(X, y, loss=CrossEntropy, optimizer=BFGS, epochs=30)
    assert accuracy_score(y, net.predict(X)) >= 0.97


if __name__ == "__main__":
    pytest.main()
