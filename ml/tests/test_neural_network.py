import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from ml.neural_network.activations import Sigmoid, SoftMax
from ml.neural_network.layers import Dense
from ml.neural_network.losses import MSE
from ml.neural_network.neural_network import Network
from ml.neural_network.optimizers import Adam

X, y = load_iris(return_X_y=True)


def test_neural_network():
    net = Network(Dense(4, 4, Sigmoid()),
                  Dense(4, 4, Sigmoid()),
                  Dense(4, 3, SoftMax()))
    net.fit(X, y, loss=MSE(), optimizer=Adam(net.params, l_rate=0.1), epochs=30)
    assert accuracy_score(y, net.predict(X)) >= 0.95


if __name__ == "__main__":
    pytest.main()
