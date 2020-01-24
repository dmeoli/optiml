import numpy as np

from optimization.functions import Function


def cross_entropy_loss(x, y):
    """Cross entropy loss function. x and y are 1D iterable objects."""
    return (-1 / len(x)) * sum(_x * np.log(_y) + (1 - _x) * np.log(1 - _y) for _x, _y in zip(x, y))


def mean_squared_error_loss(x, y):
    """Mean squared error loss function. x and y are 1D iterable objects."""
    return (1 / len(x)) * sum((_x - _y) ** 2 for _x, _y in zip(x, y))


class MSE(Function):
    def __init__(self, X, y):
        super().__init__(X.shape[1])
        self.X = X
        self.y = y
        self.x_star = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)  # or np.linalg.lstsq(X, y)[0]

    def function(self, theta):
        y_pred = np.dot(self.X, theta)
        residuals = y_pred - self.y
        return (1 / 2 * self.X.shape[0]) * np.sum(residuals ** 2)

    def jacobian(self, theta):
        y_pred = np.dot(self.X, theta)
        residuals = y_pred - self.y
        return (1 / self.X.shape[0]) * np.dot(self.X.T, residuals)


class LogLikelihood(Function):
    def __init__(self, X, y):
        super().__init__(X.shape[1])
        self.X = X
        #  self.y = y
        self.y = y.flatten()

    @staticmethod
    def sigmoid(x):
        # activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def probability(theta, X):
        # calculates the probability that an instance belongs to a particular class
        return LogLikelihood.sigmoid(np.dot(X, theta))

    def function(self, theta):
        # computes the cost function for all the training samples
        return -(1 / self.X.shape[0]) * np.sum(self.y * np.log(self.probability(theta, self.X)) +
                                               (1 - self.y) * np.log(1 - self.probability(theta, self.X)))

    def jacobian(self, theta):
        return (1 / self.X.shape[0]) * np.dot(self.X.T, self.probability(theta, self.X) - self.y)


class CrossEntropy(Function):
    def __init__(self, X, y):
        super().__init__(X.shape[1])
        self.X = X
        self.y = y

    @staticmethod
    def softmax(x):
        return np.exp(x) / sum(np.exp(x))

    def function(self, x):
        return super().function(x)

    def jacobian(self, x):
        return super().jacobian(x)
