import pytest
from sklearn.datasets import load_iris

from ml.losses import CrossEntropy
from ml.metrics import accuracy_score
from ml.neural_network.activations import Sigmoid
from ml.neural_network.neural_network import NeuralNetwork
from optimization.unconstrained.quasi_newton import BFGS


def test_neural_network_classification():
    X, y = load_iris(return_X_y=True)
    net = NeuralNetwork(hidden_layer_sizes=(4, 4),
                        activations=(Sigmoid, Sigmoid))
    net.fit(X, y, loss=CrossEntropy, optimizer=BFGS)
    assert accuracy_score(y, net.predict(X)) >= 0.9


if __name__ == "__main__":
    pytest.main()
