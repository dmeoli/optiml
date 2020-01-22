import numpy as np

from optimization.functions import Function


def cross_entropy_loss(x, y):
    """Cross entropy loss function. x and y are 1D iterable objects."""
    return (-1 / len(x)) * sum(_x * np.log(_y) + (1 - _x) * np.log(1 - _y) for _x, _y in zip(x, y))


def mean_squared_error_loss(x, y):
    """Mean squared error loss function. x and y are 1D iterable objects."""
    return (1 / len(x)) * sum((_x - _y) ** 2 for _x, _y in zip(x, y))


class CrossEntropyLoss(Function):
    def __init__(self, X, y):
        super().__init__(X.shape[1])
        self.X = X
        self.y = y

    def function(self, x):
        return super().function(x)

    def jacobian(self, x):
        return super().jacobian(x)


class MSE(Function):
    def __init__(self, X, y):
        super().__init__(X.shape[1])
        self.X = X
        self.y = y
        self.x_star = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)  # or np.linalg.lstsq(X, y)[0]

    def function(self, x):
        y_pred = np.dot(self.X, x)
        residuals = y_pred - self.y
        return (1 / 2 * len(self.y)) * np.sum(residuals ** 2)

    def jacobian(self, x):
        y_pred = np.dot(self.X, x)
        residuals = y_pred - self.y
        return (1 / len(self.y)) * np.dot(self.X.T, residuals)


class LogLoss(Function):
    def __init__(self, X, y):
        super().__init__(X.shape[1])
        self.X = X
        self.y = y

    def function(self, x):
        return super().function(x)

    def jacobian(self, x):
        return super().jacobian(x)
