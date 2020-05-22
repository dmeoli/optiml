from abc import ABC

import numpy as np

from optiml.optimization import OptimizationFunction


class SVMLoss(OptimizationFunction, ABC):

    def __init__(self, svm, X, y):
        super().__init__(X.shape[1])
        self.svm = svm
        self.X = X
        self.y = y

    def args(self):
        return self.X, self.y

    def loss(self, y_pred, y_true):
        raise NotImplementedError

    def loss_derivative(self, X, y):
        raise NotImplementedError


class SVCLoss(SVMLoss, ABC):

    def __init__(self, svm, X, y, penalty='l2'):
        super().__init__(svm, X, y)
        self.penalty = penalty

    def function(self, coef, X_batch=None, y_batch=None):
        if X_batch is None:
            X_batch = self.X
        if y_batch is None:
            y_batch = self.y

        self.svm.coef_ = coef

        norm = 1 if self.penalty == 'l1' else 2
        return (0.5 * np.linalg.norm(coef, ord=norm) + self.svm.C *
                np.mean(self.loss(self.svm.decision_function(X_batch), y_batch)))

    def jacobian(self, coef, X_batch=None, y_batch=None):
        if X_batch is None:
            X_batch = self.X
        if y_batch is None:
            y_batch = self.y

        n_samples = X_batch.shape[0]
        return (1 / n_samples * (coef if self.penalty == 'l2' else np.sign(coef)) -
                self.svm.C / n_samples * self.loss_derivative(X_batch, y_batch))

    def __call__(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


class Hinge(SVCLoss):
    """
    Compute the Hinge loss for classification as:

        L(y_pred, y_true) = max(0, 1 - y_true * y_pred)
    """

    def loss(self, y_pred, y_true):
        return np.maximum(0, 1 - y_true * y_pred)

    def loss_derivative(self, X, y):
        one_rows = np.where(y * self.svm.decision_function(X) < 1.)[0]
        return np.dot(y[one_rows], X[one_rows])


class SquaredHinge(Hinge):
    """
    Compute the squared Hinge loss for classification as:

        L(y_pred, y_true) = max(0, 1 - y_true * y_pred)^2
    """

    def loss(self, y_pred, y_true):
        return np.square(super().loss(y_pred, y_true))

    def jacobian(self, coef, X_batch=None, y_batch=None):
        return 2 * super().jacobian(coef, X_batch, y_batch)


class SVRLoss(SVMLoss, ABC):

    def function(self, coef, X_batch=None, y_batch=None):
        if X_batch is None:
            X_batch = self.X
        if y_batch is None:
            y_batch = self.y

        self.svm.coef_ = coef

        return (0.5 * np.linalg.norm(coef) + self.svm.C *
                np.mean(self.loss(self.svm.predict(X_batch), y_batch)))

    def jacobian(self, coef, X_batch=None, y_batch=None):
        if X_batch is None:
            X_batch = self.X
        if y_batch is None:
            y_batch = self.y

        n_samples = X_batch.shape[0]
        return 1 / n_samples * coef - self.svm.C / n_samples * self.loss_derivative(X_batch, y_batch)

    def __call__(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


class EpsilonInsensitive(SVRLoss):
    """
    Compute the epsilon-insensitive loss for regression as:

        L(y_pred, y_true) = max(0, |y_true - y_pred| - epsilon)
    """

    def __init__(self, svm, X, y, epsilon=0.1):
        super().__init__(svm, X, y)
        self.epsilon = epsilon

    def loss(self, y_pred, y_true):
        return np.maximum(0, np.abs(y_pred - y_true) - self.epsilon)

    def loss_derivative(self, X, y):
        y_pred = self.svm.predict(X)
        one_rows = np.where(np.abs(y - y_pred) > self.epsilon)[0]
        return np.sign(y_pred[one_rows] - y[one_rows])


class SquaredEpsilonInsensitive(EpsilonInsensitive):
    """
    Compute the squared epsilon-insensitive loss for regression as:

        L(y_pred, y_true) = max(0, |y_true - y_pred| - epsilon)^2
    """

    def loss(self, y_pred, y_true):
        return np.square(super().loss(y_pred, y_true))

    def jacobian(self, coef, X_batch=None, y_batch=None):
        return 2 * super().jacobian(coef, X_batch, y_batch)


hinge = Hinge
squared_hinge = SquaredHinge
epsilon_insensitive = EpsilonInsensitive
squared_epsilon_insensitive = SquaredEpsilonInsensitive
