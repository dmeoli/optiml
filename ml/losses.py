import numpy as np

from ml.neural_network.activations import Sigmoid
from optimization.functions import Function


class LossFunction(Function):
    def __init__(self, regularization_type, lmbda=0.1, alpha=0.2):
        super().__init__()
        if regularization_type not in ('l1', 'l2', 'elastic-net', 'none'):
            raise ValueError('unknown regularization type formula {}'.format(regularization_type))
        self.regularization_type = regularization_type
        self.lmbda = lmbda
        self.alpha = alpha

    def function(self, theta, X=None, y=None):
        return NotImplementedError

    @staticmethod
    def predict(X, theta):
        return NotImplementedError

    def regularization(self, theta, X):
        if self.regularization_type is 'l1':
            return (self.lmbda / 2 * X.shape[0]) * np.sum(np.abs(theta))
        elif self.regularization_type is 'l2':
            return (self.lmbda / 2 * X.shape[0]) * np.sum(np.square(theta))
        elif self.regularization_type is 'elastic-net':
            return (self.lmbda / 2 * X.shape[0]) * np.sum(
                self.alpha * np.square(theta) + (1 - self.alpha) * np.abs(theta))
        else:
            return 0

    def jacobian(self, theta, X=None, y=None):
        return (1 / X.shape[0]) * np.dot(X.T, self.predict(X, theta) - y)


class MeanSquaredError(LossFunction):
    def __init__(self, regularization_type='l1', lmbda=0.1, alpha=0.2):
        super().__init__(regularization_type, lmbda, alpha)

    @staticmethod
    def predict(X, theta):
        return np.dot(X, theta)

    def function(self, theta, X=None, y=None):
        return (1 / 2 * X.shape[0]) * np.sum(np.square(self.predict(X, theta) - y)) + self.regularization(theta, X)


class MeanAbsoluteError(LossFunction):
    def __init__(self, regularization_type='l2', lmbda=0.1, alpha=0.2):
        super().__init__(regularization_type, lmbda, alpha)

    @staticmethod
    def predict(X, theta):
        return np.dot(X, theta)

    def function(self, theta, X=None, y=None):
        return (1 / 2 * X.shape[0]) * np.sum(np.abs(self.predict(X, theta) - y)) + self.regularization(theta, X)


class CrossEntropy(LossFunction):
    def __init__(self, regularization_type='l2', lmbda=0.1, alpha=0.2):
        super().__init__(regularization_type, lmbda, alpha)

    @staticmethod
    def predict(X, theta):
        return Sigmoid().function(np.dot(X, theta))

    def function(self, theta, X=None, y=None):
        pred = self.predict(X, theta)
        return -(1 / X.shape[0]) * np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred)) + self.regularization(theta, X)
