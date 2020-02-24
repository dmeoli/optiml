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

    ml_cup = np.delete(np.genfromtxt('./ml/data/ML-CUP19/ML-CUP19-TR.csv', delimiter=','), 0, 1)
    X, y = ml_cup[:, :-2], ml_cup[:, -2:]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

    nn = NeuralNetwork(hidden_layer_sizes=(20, 20),
                       activations=(Sigmoid, Sigmoid))
    nn.fit(X_train, y_train, loss=MeanSquaredError, optimizer=BFGS)
    pred = nn.predict(X_test)
    print(mean_squared_error(y_test, pred))
    print(mean_euclidean_error(y_test, pred))
