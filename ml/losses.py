import numpy as np
from jax.scipy.special import xlogy

from optimization.optimization_function import OptimizationFunction


class LossFunction(OptimizationFunction):
    def __init__(self, X, y):
        super().__init__(X.shape[1])
        self.X = X
        self.y = y

    def args(self):
        return self.X, self.y

    def function(self, y_pred, y_true):
        raise NotImplementedError

    # def jacobian(self, theta, X, y):
    #     return np.dot(X.T, self.predict(X, theta) - y) / X.shape[0]


class MeanSquaredError(LossFunction):
    def __init__(self, X, y):
        super().__init__(X, y)

    def x_star(self):
        if self.x_opt is not None:
            return self.x_opt
        else:  # or np.linalg.lstsq(self.X, self.y)[0]
            self.x_opt = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
            return self.x_opt

    def function(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))


class MeanAbsoluteError(LossFunction):
    def __init__(self, X, y):
        super().__init__(X, y)

    def function(self, y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))


class CrossEntropy(LossFunction):
    def __init__(self, X, y):
        super().__init__(X, y)

    def function(self, y_pred, y_true):
        return -np.mean(xlogy(y_true, y_pred) + xlogy(1. - y_true, 1. - y_pred))
