import pytest
from sklearn.datasets import load_iris

from ml.losses import categorical_cross_entropy
from ml.metrics import accuracy_score
from ml.neural_network.activations import sigmoid, softmax
from ml.neural_network.layers import FullyConnected
from ml.neural_network.neural_network import NeuralNetwork
from optimization.unconstrained.quasi_newton import BFGS


def test_neural_network_classification():
    X, y = load_iris(return_X_y=True)
    net = NeuralNetwork(FullyConnected(4, 4, sigmoid),
                        FullyConnected(4, 4, sigmoid),
                        FullyConnected(4, 3, softmax))
    net.fit(X, y, loss=categorical_cross_entropy, optimizer=BFGS)
    assert accuracy_score(y, net.predict(X)) >= 0.96


if __name__ == "__main__":
    pytest.main()
