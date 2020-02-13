import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from ml.losses import MeanSquaredError
from ml.neural_network.activations import Sigmoid, Softmax
from ml.neural_network.layers import Dense
from ml.neural_network.neural_network import Network
from optimization.unconstrained.quasi_newton import BFGS

X, y = load_iris(return_X_y=True)


def test_neural_network():
    net = Network(Dense(4, 4, Sigmoid()),
                  Dense(4, 4, Sigmoid()),
                  Dense(4, 3, Softmax()))
    net.fit(X, y, loss=MeanSquaredError(X, y), optimizer=BFGS, epochs=30)
    assert accuracy_score(y, net.predict(X)) >= 0.97


if __name__ == "__main__":
    pytest.main()
