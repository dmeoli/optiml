import numpy as np

from ml.neural_network.activations import Sigmoid
from optimization.functions import Function


class LossFunction(Function):
    def __init__(self, X, y, regularization_type, lmbda=0.1, alpha=0.2):
        super().__init__(X.shape[1])
        self.X = X
        self.y = y
        if regularization_type not in ('l1', 'l2', 'elastic-net', 'none'):
            raise ValueError('unknown regularization type formula {}'.format(regularization_type))
        self.regularization_type = regularization_type
        self.lmbda = lmbda
        self.alpha = alpha

    @staticmethod
    def predict(X, theta):
        return NotImplementedError

    def regularization(self, theta):
        if self.regularization_type is 'l1':
            return (self.lmbda / 2 * self.X.shape[0]) * np.sum(np.abs(theta))
        elif self.regularization_type is 'l2':
            return (self.lmbda / 2 * self.X.shape[0]) * np.sum(np.square(theta))
        elif self.regularization_type is 'elastic-net':
            return (self.lmbda / 2 * self.X.shape[0]) * np.sum(
                self.alpha * np.square(theta) + (1 - self.alpha) * np.abs(theta))
        else:
            return 0


class MeanSquaredError(LossFunction):
    def __init__(self, X, y, regularization_type='l1', lmbda=0.1, alpha=0.2):
        super().__init__(X, y, regularization_type, lmbda, alpha)
        self.x_star = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)  # or np.linalg.lstsq(X, y)[0]

    @staticmethod
    def predict(X, theta):
        return np.dot(X, theta)

    def function(self, theta):
        return (1 / 2 * self.X.shape[0]) * np.sum(
            np.square(self.predict(self.X, theta) - self.y)) + self.regularization(theta)

    def jacobian(self, theta):
        return (1 / self.X.shape[0]) * np.dot(self.X.T, self.predict(self.X, theta) - self.y)


class MeanAbsoluteError(LossFunction):
    def __init__(self, X, y, regularization_type='l2', lmbda=0.1, alpha=0.2):
        super().__init__(X, y, regularization_type, lmbda, alpha)

    @staticmethod
    def predict(X, theta):
        return np.dot(X, theta)

    def function(self, theta):
        return (1 / 2 * self.X.shape[0]) * np.sum(
            np.abs(self.predict(self.X, theta) - self.y)) + self.regularization(theta)

    def jacobian(self, theta):
        return (1 / self.X.shape[0]) * np.dot(self.X.T, self.predict(self.X, theta) - self.y)


class CrossEntropy(LossFunction):
    def __init__(self, X, y, regularization_type='l2', lmbda=0.1, alpha=0.2):
        super().__init__(X, y, regularization_type, lmbda, alpha)

    @staticmethod
    def predict(X, theta, activation=Sigmoid):
        return activation().function(np.dot(X, theta))

    def function(self, theta):
        pred = self.predict(self.X, theta)
        return -(1 / self.X.shape[0]) * np.sum(
            self.y * np.log(pred) + (1 - self.y) * np.log(1 - pred)) + self.regularization(theta)

    def jacobian(self, theta):
        return (1 / self.X.shape[0]) * np.dot(self.X.T, self.predict(self.X, theta) - self.y)
