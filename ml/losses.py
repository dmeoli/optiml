import numpy as np

from ml.neural_network.activations import Sigmoid
from optimization.functions import Function


class LossFunction(Function):
    def __init__(self, X, y):
        super().__init__(X.shape[1])
        self.X = X
        self.y = y

    @staticmethod
    def predict(X, theta):
        return NotImplementedError


class MeanSquaredError(LossFunction):
    def __init__(self, X, y):
        super().__init__(X, y)
        self.x_star = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)  # or np.linalg.lstsq(X, y)[0]

    @staticmethod
    def predict(X, theta):
        return np.dot(X, theta)

    def function(self, theta):
        return (1 / 2 * self.X.shape[0]) * np.sum(np.square(self.predict(self.X, theta) - self.y))

    def jacobian(self, theta):
        return (1 / self.X.shape[0]) * np.dot(self.X.T, self.predict(self.X, theta) - self.y)


class CrossEntropy(LossFunction):
    def __init__(self, X, y):
        super().__init__(X, y)

    @staticmethod
    def predict(X, theta):
        return Sigmoid().function(np.dot(X, theta))

    def function(self, theta):
        pred = self.predict(self.X, theta)
        return -(1 / self.X.shape[0]) * np.sum(self.y * np.log(pred) + (1 - self.y) * np.log(1 - pred))

    def jacobian(self, theta):
        return (1 / self.X.shape[0]) * np.dot(self.X.T, self.predict(self.X, theta) - self.y)
