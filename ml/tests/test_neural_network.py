import pytest
from sklearn.datasets import load_iris

from ml.losses import CrossEntropy
from ml.metrics import accuracy_score
from ml.neural_network.activations import sigmoid, softmax
from ml.neural_network.layers import Dense
from ml.neural_network.neural_network import NeuralNetwork
from optimization.unconstrained.quasi_newton import BFGS


def test_neural_network_classification():
    X, y = load_iris(return_X_y=True)
    net = NeuralNetwork(Dense(4, 4, sigmoid),
                        Dense(4, 4, sigmoid),
                        Dense(4, 3, softmax))
    net.fit(X, y, loss=CrossEntropy, optimizer=BFGS)
    assert accuracy_score(y, net.predict(X)) >= 0.9


if __name__ == "__main__":
    pytest.main()
