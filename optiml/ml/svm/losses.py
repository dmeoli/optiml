from abc import ABC

import numpy as np

from ...optimization import OptimizationFunction


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

    def loss_jacobian(self, X_batch, y_batch):
        raise NotImplementedError

    def __call__(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


class SVCLoss(SVMLoss, ABC):

    def __init__(self, svm, X, y, penalty='l2'):
        super().__init__(svm, X, y)
        self.penalty = penalty

    def function(self, packed_coef_inter, X_batch=None, y_batch=None):
        if X_batch is None:
            X_batch = self.X
        if y_batch is None:
            y_batch = self.y

        self.svm._unpack(packed_coef_inter)

        n_samples = X_batch.shape[0]
        if self.penalty == 'l1':
            return (1 / (2 * n_samples) * np.linalg.norm(packed_coef_inter, ord=1) +
                    self.svm.C / n_samples * np.sum(self.loss(self.svm.decision_function(X_batch), y_batch)))
        elif self.penalty == 'l2':
            return (1 / (2 * n_samples) * np.linalg.norm(packed_coef_inter) ** 2 +
                    self.svm.C / n_samples * np.sum(self.loss(self.svm.decision_function(X_batch), y_batch)))

    def jacobian(self, packed_coef_inter, X_batch=None, y_batch=None):
        if X_batch is None:
            X_batch = self.X
        if y_batch is None:
            y_batch = self.y

        n_samples = X_batch.shape[0]
        if self.penalty == 'l1':
            return (1 / n_samples * np.sign(packed_coef_inter) -
                    self.svm.C / n_samples * self.loss_jacobian(X_batch, y_batch))
        elif self.penalty == 'l2':
            return ((1 / n_samples) * 2 * packed_coef_inter -
                    self.svm.C / n_samples * self.loss_jacobian(X_batch, y_batch))


class Hinge(SVCLoss):
    """
    Compute the Hinge loss for classification as:

        L(y_pred, y_true) = max(0, 1 - y_true * y_pred)
    """

    def loss(self, y_pred, y_true):
        return np.maximum(0, 1 - y_true * y_pred)

    def loss_jacobian(self, X_batch, y_batch):
        mask = y_batch * self.svm.decision_function(X_batch) < 1.
        return np.dot(y_batch[mask], X_batch[mask])


class SquaredHinge(Hinge):
    """
    Compute the squared Hinge loss for classification as:

        L(y_pred, y_true) = max(0, 1 - y_true * y_pred)^2
    """

    def loss(self, y_pred, y_true):
        return np.square(super().loss(y_pred, y_true))

    def loss_jacobian(self, X_batch, y_batch):
        return 2 * super().loss_jacobian(X_batch, y_batch)


class SVRLoss(SVMLoss, ABC):

    def function(self, packed_coef_inter, X_batch=None, y_batch=None):
        if X_batch is None:
            X_batch = self.X
        if y_batch is None:
            y_batch = self.y

        self.svm._unpack(packed_coef_inter)

        n_samples = X_batch.shape[0]
        return (1 / (2 * n_samples) * np.linalg.norm(packed_coef_inter) ** 2 +
                self.svm.C / n_samples * np.sum(self.loss(self.svm.predict(X_batch), y_batch)))

    def jacobian(self, packed_coef_inter, X_batch=None, y_batch=None):
        if X_batch is None:
            X_batch = self.X
        if y_batch is None:
            y_batch = self.y

        n_samples = X_batch.shape[0]
        return ((1 / n_samples) * 2 * packed_coef_inter -
                self.svm.C / n_samples * self.loss_jacobian(X_batch, y_batch))


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

    def loss_jacobian(self, X_batch, y_batch):
        y_pred = self.svm.predict(X_batch)
        mask = np.abs(y_pred - y_batch) > self.epsilon
        return np.dot(np.sign(y_batch[mask] - y_pred[mask]), X_batch[mask])


class SquaredEpsilonInsensitive(EpsilonInsensitive):
    """
    Compute the squared epsilon-insensitive loss for regression as:

        L(y_pred, y_true) = max(0, |y_true - y_pred| - epsilon)^2
    """

    def loss(self, y_pred, y_true):
        return np.square(super().loss(y_pred, y_true))

    def loss_jacobian(self, X_batch, y_batch):
        return 2 * super().loss_jacobian(X_batch, y_batch)


hinge = Hinge
squared_hinge = SquaredHinge
epsilon_insensitive = EpsilonInsensitive
squared_epsilon_insensitive = SquaredEpsilonInsensitive
