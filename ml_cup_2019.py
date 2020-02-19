import numpy as np
from sklearn.datasets import load_iris

from ml.losses import CrossEntropy, MeanSquaredError
from ml.metrics import mean_euclidean_error, accuracy_score, mean_squared_error
from ml.neural_network.activations import Sigmoid
from ml.neural_network.neural_network import NeuralNetwork
from optimization.unconstrained.quasi_newton import BFGS

if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    nn = NeuralNetwork(hidden_layer_sizes=(4, 4),
                       activations=(Sigmoid, Sigmoid))
    nn.fit(X, y, loss=CrossEntropy, optimizer=BFGS)
    pred = nn.predict(X)
    print(pred)
    print(accuracy_score(y, pred))

    ml_cup_train = np.delete(np.genfromtxt('./ml/data/ML-CUP19/ML-CUP19-TR.csv', delimiter=','), 0, 1)
    X_train, y_train = ml_cup_train[:, :-2], ml_cup_train[:, -2:]

    X_exam = np.delete(np.genfromtxt('./ml/data/ML-CUP19/ML-CUP19-TS.csv', delimiter=','), 0, 1)

    nn = NeuralNetwork(hidden_layer_sizes=(20, 20),
                       activations=(Sigmoid, Sigmoid))
    nn.fit(X_train, y_train, loss=MeanSquaredError, optimizer=BFGS)
    pred = nn.predict(X_train)
    print(mean_squared_error(y_train, pred))
    print(mean_euclidean_error(y_train, pred))
