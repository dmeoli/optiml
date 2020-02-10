import numpy as np
import pytest
from sklearn.datasets import load_iris

from ml.neural_network.activations import Tanh, Sigmoid
from ml.neural_network.layers import Dense
from ml.neural_network.losses import MSE
from ml.neural_network.neural_network import Network
from ml.neural_network.optimizers import Adam

X, y = load_iris(return_X_y=True)
X, y = X, y[:, np.newaxis]


def test_neural_network():
    net = Network(Dense(4, 4, Tanh()),
                  Dense(4, 4, Tanh()),
                  Dense(4, 2, Sigmoid()))
    net.fit(X, y, loss=MSE(), optimizer=Adam(net.params, l_rate=0.1), epochs=30)
    # net.predict(X)


if __name__ == "__main__":
    pytest.main()
