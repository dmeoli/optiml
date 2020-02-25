import numpy as np

from ml.neural_network.activations import Sigmoid
from optimization.optimization_function import OptimizationFunction


class LossFunction(OptimizationFunction):
    def __init__(self, X, y, regularization_type, lmbda=0.001):
        super().__init__(X.shape[1])
        self.X = X
        self.y = y
        if regularization_type not in ('l1', 'l2', 'none'):
            raise ValueError('unknown regularization type {}'.format(regularization_type))
        self.regularization_type = regularization_type
        if not lmbda >= 0:
            raise ValueError('lmbda must be >= 0')
        self.lmbda = lmbda

    def args(self):
        return self.X, self.y

    def predict(self, X, theta):
        return np.dot(X, theta)

    def function(self, theta, X, y):
        raise NotImplementedError

    def regularization(self, theta):
        if self.regularization_type is 'l1':
            return self.lmbda * np.sum(np.abs(theta))
        elif self.regularization_type is 'l2':
            return self.lmbda * np.sum(np.square(theta))
        return 0

    def jacobian(self, theta, X, y):
        return np.dot(X.T, self.predict(X, theta) - y) / X.shape[0]


class MeanSquaredError(LossFunction):
    def __init__(self, X, y, regularization_type='l1', lmbda=0.001):
        super().__init__(X, y, regularization_type, lmbda)

    def x_star(self):
        if self.x_opt is not None:
            return self.x_opt
        else:  # or np.linalg.lstsq(self.X, self.y)[0]
            self.x_opt = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
            return self.x_opt

    def function(self, theta, X, y):
        self.pred = self.predict(X, theta)
        self.target = y
        return np.mean(np.square(self.pred - y)) + self.regularization(theta) / X.shape[0]


class MeanAbsoluteError(LossFunction):
    def __init__(self, X, y, regularization_type='l2', lmbda=0.001):
        super().__init__(X, y, regularization_type, lmbda)

    def function(self, theta, X, y):
        self.pred = self.predict(X, theta)
        self.target = y
        return np.mean(np.abs(self.pred - y)) + self.regularization(theta) / X.shape[0]


class CrossEntropy(LossFunction):
    def __init__(self, X, y, regularization_type='l2', lmbda=0.001):
        super().__init__(X, y, regularization_type, lmbda)

    def predict(self, X, theta):
        return Sigmoid().function(np.dot(X, theta))

    def function(self, theta, X, y):
        self.pred = self.predict(X, theta)
        self.target = y
        return -np.mean(np.nan_to_num(y * np.log(self.pred) +
                                      (1. - y) * np.log(1. - self.pred))) + self.regularization(theta) / X.shape[0]
