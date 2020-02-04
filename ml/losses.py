import numpy as np

from ml.neural_network.activations import Sigmoid
from optimization.optimization_function import OptimizationFunction


class LossFunction(OptimizationFunction):
    def __init__(self, X, y, regularization_type, lmbda=0.1, alpha=0.2):
        super().__init__(X.shape[1])
        self.X = X
        self.y = y
        if regularization_type not in ('l1', 'l2', 'elastic-net', 'none'):
            raise ValueError('unknown regularization type {}'.format(regularization_type))
        self.regularization_type = regularization_type
        self.lmbda = lmbda
        self.alpha = alpha

    def args(self):
        return self.X, self.y

    def function(self, theta, X, y):
        raise NotImplementedError

    @staticmethod
    def predict(X, theta):
        raise NotImplementedError

    def regularization(self, theta, X):
        if self.regularization_type is 'l1':
            return (self.lmbda / 2 * X.shape[0]) * np.sum(np.abs(theta))
        elif self.regularization_type is 'l2':
            return (self.lmbda / 2 * X.shape[0]) * np.sum(theta ** 2)
        elif self.regularization_type is 'elastic-net':
            return (self.lmbda / 2 * X.shape[0]) * np.sum(
                self.alpha * theta ** 2 + (1 - self.alpha) * np.abs(theta))
        else:
            return 0

    def jacobian(self, theta, X, y):
        return (1 / X.shape[0]) * np.dot(X.T, self.predict(X, theta) - y)


class MeanSquaredError(LossFunction):
    def __init__(self, X, y, regularization_type='l1', lmbda=0.1, alpha=0.2):
        super().__init__(X, y, regularization_type, lmbda, alpha)

    def x_star(self):
        if self.x_opt is not None:
            return self.x_opt
        else:
            self.x_opt = np.linalg.inv(self.X.T.dot(self.X)).dot(
                self.X.T).dot(self.y)  # or np.linalg.lstsq(self.X, self.y)[0]
            return self.x_opt

    @staticmethod
    def predict(X, theta):
        return np.dot(X, theta)

    def function(self, theta, X, y):
        return (1 / 2 * X.shape[0]) * np.sum((self.predict(X, theta) - y) ** 2) + self.regularization(theta, X)


class MeanAbsoluteError(LossFunction):
    def __init__(self, X, y, regularization_type='l2', lmbda=0.1, alpha=0.2):
        super().__init__(X, y, regularization_type, lmbda, alpha)

    @staticmethod
    def predict(X, theta):
        return np.dot(X, theta)

    def function(self, theta, X, y):
        return (1 / 2 * X.shape[0]) * np.sum(np.abs(self.predict(X, theta) - y)) + self.regularization(theta, X)


class CrossEntropy(LossFunction):
    def __init__(self, X, y, regularization_type='l2', lmbda=0.1, alpha=0.2):
        super().__init__(X, y, regularization_type, lmbda, alpha)

    @staticmethod
    def predict(X, theta):
        return Sigmoid().function(np.dot(X, theta))

    def function(self, theta, X, y):
        pred = self.predict(X, theta)
        return -(1 / X.shape[0]) * np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred)) + self.regularization(theta, X)
