import numpy as np

from ml.neural_network.activations import Sigmoid
from optimization.optimization_function import OptimizationFunction


class LossFunction(OptimizationFunction):
    def __init__(self, X, y, regularization_type, lmbda=0.1):
        super().__init__(X.shape[1])
        self.X = X
        self.y = y
        if regularization_type not in ('l1', 'l2', 'none'):
            raise ValueError('unknown regularization type {}'.format(regularization_type))
        self.regularization_type = regularization_type
        self.lmbda = lmbda

    def args(self):
        return self.X, self.y

    def function(self, theta, X, y):
        raise NotImplementedError

    def regularization(self, theta, X):
        if self.regularization_type is 'l1':
            return self.lmbda * np.mean(np.abs(theta))
        elif self.regularization_type is 'l2':
            return self.lmbda * np.mean(np.square(theta))
        return 0

    def jacobian(self, theta, X, y):
        return np.dot(X.T, np.dot(X, theta) - y) / X.shape[0]


class MeanSquaredError(LossFunction):
    def __init__(self, X, y, regularization_type='l1', lmbda=0.1):
        super().__init__(X, y, regularization_type, lmbda)

    def x_star(self):
        if self.x_opt is not None:
            return self.x_opt
        else:
            self.x_opt = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
            # or np.linalg.lstsq(self.X, self.y)[0]
            return self.x_opt

    def function(self, theta, X, y):
        return np.mean(np.square(np.dot(X, theta) - y)) + self.regularization(theta, X)

    def function(self, predict, y):
        self.prediction = predict
        self.target = y
        return np.mean(np.square(predict - y))

    @property
    def delta(self):
        return self.prediction - self.target


class MeanAbsoluteError(LossFunction):
    def __init__(self, X, y, regularization_type='l2', lmbda=0.1):
        super().__init__(X, y, regularization_type, lmbda)

    def function(self, theta, X, y):
        return np.mean(np.abs(np.dot(X, theta) - y)) + self.regularization(theta, X)

    def function(self, predict, y):
        self.prediction = predict
        self.target = y
        return np.mean(np.abs(predict - y))

    @property
    def delta(self):
        return self.prediction - self.target


class CrossEntropy(LossFunction):
    def __init__(self, X, y, regularization_type='l2', lmbda=0.1):
        super().__init__(X, y, regularization_type, lmbda)

    def function(self, theta, X, y):
        pred = Sigmoid().function(np.dot(X, theta))
        return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred)) + self.regularization(theta, X)

    def function(self, predict, y):
        self.prediction = predict
        self.target = y
        return -np.mean(y * np.log(predict) + (1 - y) * np.log(1 - predict))

    @property
    def delta(self):
        return self.prediction - self.target

    def jacobian(self, theta, X, y):
        return np.dot(X.T, Sigmoid().function(np.dot(X, theta)) - y) / X.shape[0]
