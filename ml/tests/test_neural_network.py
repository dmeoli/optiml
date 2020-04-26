import pytest
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.neural_network.losses import cross_entropy, mean_squared_error
from ml.neural_network.activations import sigmoid, softmax, linear
from ml.neural_network.layers import FullyConnected
from ml.neural_network import NeuralNetworkRegressor, NeuralNetworkClassifier
from optimization.unconstrained.quasi_newton import BFGS


def test_neural_network_regressor():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75)
    net = NeuralNetworkRegressor((FullyConnected(13, 13, sigmoid),
                                  FullyConnected(13, 13, sigmoid),
                                  FullyConnected(13, 1, linear)),
                                 loss=mean_squared_error, optimizer=BFGS)
    net.fit(X_train, y_train)
    assert net.score(X_test, y_test) >= 0.2


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
