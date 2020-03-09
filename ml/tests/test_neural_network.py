import pytest
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split

from ml.losses import cross_entropy, mean_squared_error
from ml.metrics import accuracy_score, r2_score
from ml.neural_network.activations import sigmoid, softmax, linear
from ml.neural_network.layers import FullyConnected
from ml.neural_network.neural_network import NeuralNetwork
from optimization.unconstrained.quasi_newton import BFGS


def test_neural_network_regression():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    net = NeuralNetwork(FullyConnected(13, 5, sigmoid),
                        FullyConnected(5, 3, sigmoid),
                        FullyConnected(3, 1, linear))
    net.fit(X_train, y_train, loss=mean_squared_error, optimizer=BFGS)
    assert r2_score(net.predict(X_test), y_test) >= 67.45


def test_neural_network_classification():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    net = NeuralNetwork(FullyConnected(4, 4, sigmoid),
                        FullyConnected(4, 4, sigmoid),
                        FullyConnected(4, 3, softmax))
    net.fit(X_train, y_train, loss=cross_entropy, optimizer=BFGS)
    assert accuracy_score(net.predict(X_test), y_test) >= 0.94


if __name__ == "__main__":
    pytest.main()
