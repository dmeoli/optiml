import numpy as np
from sklearn.datasets import load_iris

from ml.losses import MeanSquaredError, CrossEntropy
from ml.metrics import mean_euclidean_error, accuracy_score, mean_squared_error
from ml.neural_network.activations import sigmoid, softmax, linear
from ml.neural_network.layers import Dense
from ml.neural_network.neural_network import NeuralNetwork
from ml.regularizers import l2
from optimization.unconstrained.quasi_newton import BFGS

if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)

    net = NeuralNetwork(Dense(4, 4, sigmoid),
                        Dense(4, 4, sigmoid),
                        Dense(4, 3, softmax))

    net.fit(X, y, loss=CrossEntropy, optimizer=BFGS, regularization=l2, epochs=100, batch_size=None, verbose=True)
    pred = net.predict(X)
    print(pred)
    print(accuracy_score(y, pred))

    # ml_cup = np.delete(np.genfromtxt('./ml/data/ML-CUP19/ML-CUP19-TR.csv', delimiter=','), 0, 1)
    # X, y = ml_cup[:, :-2], ml_cup[:, -2:]
    #
    # from sklearn.model_selection import train_test_split
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    #
    # net = NeuralNetwork(Dense(20, 20, sigmoid),
    #                     Dense(20, 20, sigmoid),
    #                     Dense(20, 2, linear))
    #
    # net.fit(X_train, y_train, loss=MeanSquaredError, optimizer=BFGS, learning_rate=0.01,
    #         epochs=1000, batch_size=None, verbose=True)
    # pred = net.predict(X_test)
    # print(mean_squared_error(y_test, pred))
    # print(mean_euclidean_error(y_test, pred))
