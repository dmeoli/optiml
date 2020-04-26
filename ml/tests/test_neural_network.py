import pytest
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ml.neural_network.activations import sigmoid, softmax, linear
from ml.neural_network.layers import FullyConnected
from ml.neural_network.losses import cross_entropy, mean_squared_error
from ml.neural_network.neural_network import NeuralNetworkRegressor, NeuralNetworkClassifier
from optimization.unconstrained.line_search.quasi_newton import BFGS


def test_neural_network_regressor():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    net = NeuralNetworkRegressor((FullyConnected(13, 13, sigmoid),
                                  FullyConnected(13, 13, sigmoid),
                                  FullyConnected(13, 1, linear)),
                                 loss=mean_squared_error, optimizer=BFGS)
    assert cross_val_score(net, X_scaled, y, n_jobs=-1).max() >= 0.2


def test_neural_network_classifier():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    net = NeuralNetworkClassifier((FullyConnected(4, 4, sigmoid),
                                   FullyConnected(4, 4, sigmoid),
                                   FullyConnected(4, 3, softmax)),
                                  loss=cross_entropy, optimizer=BFGS)
    assert cross_val_score(net, X_scaled, y, n_jobs=-1).max() >= 0.99


if __name__ == "__main__":
    pytest.main()
