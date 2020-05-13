import numpy as np
import pytest
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from yase.ml.neural_network import NeuralNetworkRegressor, NeuralNetworkClassifier
from yase.ml.neural_network.activations import sigmoid, softmax, linear, tanh
from yase.ml.neural_network.layers import FullyConnected
from yase.ml.neural_network.losses import mean_squared_error, sparse_categorical_cross_entropy
from yase.ml.neural_network.regularizers import L2
from yase.optimization.unconstrained.line_search import BFGS
from yase.optimization.unconstrained.stochastic import Adam


def test_perceptron_regressor():
    # aka linear regression
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    net = NeuralNetworkRegressor((FullyConnected(13, 1, linear, fit_intercept=False),),
                                 loss=mean_squared_error, optimizer=BFGS)
    net.fit(X_train, y_train)
    assert net.score(X_test, y_test) >= 0.5
    assert np.allclose(net.coefs_[0].ravel(),
                       np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train), rtol=0.1)


def test_perceptron_ridge_regressor():
    # aka ridge regression
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    lmbda = 0.1
    net = NeuralNetworkRegressor((FullyConnected(13, 1, linear, coef_reg=L2(lmbda), fit_intercept=False),),
                                 loss=mean_squared_error, optimizer=BFGS)
    net.fit(X_train, y_train)
    assert net.score(X_test, y_test) >= 0.55
    assert np.allclose(net.coefs_[0].ravel(), np.linalg.inv(X_train.T.dot(X_train) + np.identity(net.loss.ndim) *
                                                            lmbda).dot(X_train.T).dot(y_train), rtol=0.1)


def test_neural_network_regressor():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75)
    net = NeuralNetworkRegressor((FullyConnected(13, 13, sigmoid),
                                  FullyConnected(13, 13, sigmoid),
                                  FullyConnected(13, 1, linear)),
                                 loss=mean_squared_error, optimizer=Adam, learning_rate=0.01)
    net.fit(X_train, y_train)
    assert net.score(X_test, y_test) >= 0.7


def test_neural_network_classifier():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75)
    net = NeuralNetworkClassifier((FullyConnected(4, 4, tanh),
                                   FullyConnected(4, 4, tanh),
                                   FullyConnected(4, 3, softmax)),
                                  loss=sparse_categorical_cross_entropy, optimizer=Adam, learning_rate=0.01)
    net.fit(X_train, y_train)
    assert net.score(X_test, y_test) >= 0.9


if __name__ == "__main__":
    pytest.main()
