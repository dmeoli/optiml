from abc import ABC

import autograd.numpy as np

from .kernels import linear
from ...opti import OptimizationFunction


class SVMLoss(OptimizationFunction, ABC):

    def __init__(self, svm, X, y):
        super(SVMLoss, self).__init__(X.shape[1])
        self.svm = svm
        self.X = X
        self.y = y

    def args(self):
        return self.X, self.y

    def f_star(self):
        if self.svm.fit_intercept:
            return self.function(self.x_star())
        return super(SVMLoss, self).f_star()

    def x_star(self):
        if self.svm.fit_intercept:
            if not hasattr(self, 'x_opt'):
                if self.svm.loss._loss_type == 'classifier':
                    dual_svm = self.svm.dual(loss=self.svm.loss.__class__,
                                             kernel=linear,
                                             C=self.svm.C,
                                             reg_intercept=True,
                                             optimizer='cvxopt',
                                             verbose=-1)
                elif self.svm.loss._loss_type == 'regressor':
                    dual_svm = self.svm.dual(loss=self.svm.loss.__class__,
                                             epsilon=self.svm.epsilon,
                                             kernel=linear,
                                             C=self.svm.C,
                                             reg_intercept=True,
                                             optimizer='cvxopt',
                                             verbose=-1)
                dual_svm.fit(self.X[:, :-1], self.y)
                self.x_opt = np.hstack((dual_svm.coef_, dual_svm.intercept_))
            return self.x_opt
        return super(SVMLoss, self).x_star()

    def function(self, packed_coef_inter, X_batch=None, y_batch=None):
        if X_batch is None:
            X_batch = self.X
        if y_batch is None:
            y_batch = self.y

        n_samples = X_batch.shape[0]
        y_pred = np.dot(X_batch, packed_coef_inter)  # svm decision function
        return (1 / (2 * n_samples) * np.linalg.norm(packed_coef_inter) ** 2 +  # regularization term
                self.svm.C / n_samples * np.sum(self.loss(y_pred, y_batch)))  # loss term

    def loss(self, y_pred, y_true):
        raise NotImplementedError

    def jacobian(self, packed_coef_inter, X_batch=None, y_batch=None):
        if X_batch is None:
            X_batch = self.X
        if y_batch is None:
            y_batch = self.y

        n_samples = X_batch.shape[0]
        return ((1 / n_samples) * packed_coef_inter -
                self.svm.C / n_samples * self.loss_jacobian(packed_coef_inter, X_batch, y_batch))

    def loss_jacobian(self, packed_coef_inter, X_batch, y_batch):
        raise NotImplementedError


class Hinge(SVMLoss):
    """
    Compute the Hinge loss for classification as:

        L(y_pred, y_true) = max(0, 1 - y_true * y_pred)
    """

    _loss_type = 'classifier'

    def loss(self, y_pred, y_true):
        return np.maximum(0, 1 - y_true * y_pred)

    def loss_jacobian(self, packed_coef_inter, X_batch, y_batch):
        y_pred = np.dot(X_batch, packed_coef_inter)  # svm decision function
        idx = np.argwhere(y_batch * y_pred < 1.).ravel()
        return np.dot(y_batch[idx], X_batch[idx])


class SquaredHinge(Hinge):
    """
    Compute the squared Hinge loss for classification as:

        L(y_pred, y_true) = max(0, 1 - y_true * y_pred)^2
    """

    def loss(self, y_pred, y_true):
        return np.square(super(SquaredHinge, self).loss(y_pred, y_true))

    def loss_jacobian(self, packed_coef_inter, X_batch, y_batch):
        return 2 * super(SquaredHinge, self).loss_jacobian(packed_coef_inter, X_batch, y_batch)


class EpsilonInsensitive(SVMLoss):
    """
    Compute the epsilon-insensitive loss for regression as:

        L(y_pred, y_true) = max(0, |y_true - y_pred| - epsilon)
    """

    _loss_type = 'regressor'

    def __init__(self, svm, X, y, epsilon):
        super(EpsilonInsensitive, self).__init__(svm, X, y)
        self.epsilon = epsilon

    def loss(self, y_pred, y_true):
        return np.maximum(0, np.abs(y_true - y_pred) - self.epsilon)

    def loss_jacobian(self, packed_coef_inter, X_batch, y_batch):
        y_pred = np.dot(X_batch, packed_coef_inter)  # svm decision function
        idx = np.argwhere(np.abs(y_batch - y_pred) >= self.epsilon).ravel()
        z = y_batch[idx] - y_pred[idx]
        return np.dot(np.sign(z), X_batch[idx])  # or np.dot(np.divide(z, np.abs(z)), X_batch[idx])


class SquaredEpsilonInsensitive(EpsilonInsensitive):
    """
    Compute the squared epsilon-insensitive loss for regression as:

        L(y_pred, y_true) = max(0, |y_true - y_pred| - epsilon)^2
    """

    def loss(self, y_pred, y_true):
        return np.square(super(SquaredEpsilonInsensitive, self).loss(y_pred, y_true))

    def loss_jacobian(self, packed_coef_inter, X_batch, y_batch):
        y_pred = np.dot(X_batch, packed_coef_inter)  # svm decision function
        idx = np.argwhere(np.abs(y_batch - y_pred) >= self.epsilon).ravel()
        z = y_batch[idx] - y_pred[idx]
        return 2 * np.dot(np.sign(z) * (np.abs(z) - self.epsilon), X_batch[idx])


hinge = Hinge
squared_hinge = SquaredHinge
epsilon_insensitive = EpsilonInsensitive
squared_epsilon_insensitive = SquaredEpsilonInsensitive
