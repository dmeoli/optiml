import pytest
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split

from ml.losses import cross_entropy, mean_squared_error
from ml.activations import sigmoid, softmax, linear
from ml.layers import FullyConnected
from ml.neural_network import NeuralNetworkRegressor, NeuralNetworkClassifier
from optimization.unconstrained.quasi_newton import BFGS


def test_neural_network_regressor():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    net = NeuralNetworkRegressor((FullyConnected(13, 13, sigmoid),
                                  FullyConnected(13, 13, sigmoid),
                                  FullyConnected(13, 1, linear)),
                                 loss=mean_squared_error, optimizer=BFGS)
    net.fit(X_train, y_train)
    assert net.score(X_test, y_test) >= 0.1


def test_neural_network_classifier():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    net = NeuralNetworkClassifier((FullyConnected(4, 4, sigmoid),
                                   FullyConnected(4, 4, sigmoid),
                                   FullyConnected(4, 3, softmax)),
                                  loss=cross_entropy, optimizer=BFGS)
    net.fit(X_train, y_train)
    assert net.score(X_test, y_test) >= 0.89


if __name__ == "__main__":
    pytest.main()
