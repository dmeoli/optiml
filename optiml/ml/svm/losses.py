import numpy as np

from optiml.optimization import OptimizationFunction


class SVMLoss(OptimizationFunction):

    def __init__(self, svm, X, y):
        super().__init__(X.shape[1])
        self.svm = svm
        self.X = X
        self.y = y

    def args(self):
        return self.X, self.y

    def loss(self, coef, X, y):
        raise NotImplementedError

    def function(self, coef, X_batch=None, y_batch=None):
        if X_batch is None:
            X_batch = self.X
        if y_batch is None:
            y_batch = self.y

        n_samples = X_batch.shape[0]
        return (1 / (2. * n_samples) * np.linalg.norm(coef) + self.svm.C / n_samples *
                np.sum(self.loss(coef, X_batch, y_batch)))

    def jacobian(self, coef, X_batch=None, y_batch=None):
        if X_batch is None:
            X_batch = self.X
        if y_batch is None:
            y_batch = self.y

        n_samples = X_batch.shape[0]
        one_rows = np.where(y_batch * np.dot(X_batch, coef) < 1.)[0]
        return 1. / n_samples * coef - self.svm.C / n_samples * np.dot(y_batch[one_rows], X_batch[one_rows])


class Hinge(SVMLoss):

    def loss(self, coef, X, y):
        return np.maximum(0, 1 - y * np.dot(X, coef))


class SquaredHinge(Hinge):

    def loss(self, coef, X, y):
        return np.square(super().loss(coef, X, y))


class EpsilonInsensitive(SVMLoss):

    def loss(self, coef, X, y):
        return


class SquaredEpsilonInsensitive(EpsilonInsensitive):

    def loss(self, coef, X, y):
        return np.square(super().loss(coef, X, y))


hinge = Hinge
squared_hinge = SquaredHinge
epsilon_insensitive = EpsilonInsensitive
squared_epsilon_insensitive = SquaredEpsilonInsensitive
