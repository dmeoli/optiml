from abc import ABC

import autograd.numpy as np
import cvxpy as cp

from ...opti import OptimizationFunction


class SVMLoss(OptimizationFunction, ABC):
    """
    Base abstract class for all SVM loss functions. It defines the
    primal objective, i.e., the regularization term plus the loss term
    averaged over the training samples, together with its jacobian.

    Subclasses must implement ``loss``, ``loss_jacobian`` and ``step_size``.
    """

    def __init__(self, svm, X, y):
        """
        Parameters
        ----------

        svm : `SVM` instance
            The SVM estimator this loss is attached to. It provides the
            hyper-parameters used by the objective, e.g., ``C`` and
            ``fit_intercept``.

        X : ndarray of shape (n_samples, n_features)
            Training data over which the loss is evaluated.

        y : ndarray of shape (n_samples,)
            Target values associated with ``X``.
        """
        super(SVMLoss, self).__init__(X.shape[1])
        self.svm = svm
        self.X = X
        self.y = y

    def args(self):
        return self.X, self.y

    def x_star(self):
        # Compute the exact minimizer of the *same* primal objective that the
        # optimizers minimize, i.e., 1/(2n) ||theta||^2 + C/n sum(loss), by solving
        # it directly as a convex program to high accuracy with a reliable conic
        # solver, instead of recovering it (less accurately) from the dual. This
        # makes f_star() = function(x_star()) a genuine, solver-certified optimum.
        if not hasattr(self, 'x_opt'):
            n_samples = self.X.shape[0]
            theta = cp.Variable(self.X.shape[1])
            objective = cp.Minimize(1 / (2 * n_samples) * cp.sum_squares(theta) +  # regularization term
                                    self.svm.C / n_samples * self._cvxpy_loss(theta))  # loss term
            problem = cp.Problem(objective)
            # solve to high accuracy, falling back to other available solvers if needed
            for solver in (cp.CLARABEL, cp.ECOS, cp.OSQP, cp.SCS):
                try:
                    problem.solve(solver=solver)
                except (cp.error.SolverError, cp.error.DCPError, KeyError):
                    continue
                if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    break
            if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raise ValueError(f'could not compute the optimal solution x_star '
                                 f'(solver status: {problem.status})')
            self.x_opt = np.asarray(theta.value, dtype=float)
        return self.x_opt

    def f_star(self):
        return self.function(self.x_star())

    def _cvxpy_loss(self, theta):
        """
        The cvxpy expression of the (summed over the samples) loss term as a
        function of the optimization variable ``theta``, used to build the convex
        primal program whose optimum defines f_star.

        :param theta: the cvxpy variable of the packed coefficients and intercept.
        :return:      the cvxpy expression of sum(loss(y, X theta)).
        """
        raise NotImplementedError

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
        return ((1 / n_samples) * packed_coef_inter -  # jacobian wrt the regularization term
                self.svm.C / n_samples * self.loss_jacobian(
                    packed_coef_inter, X_batch, y_batch))  # jacobian wrt the loss term

    def loss_jacobian(self, packed_coef_inter, X_batch, y_batch):
        raise NotImplementedError

    def step_size(self, X_batch, y_batch):
        raise NotImplementedError


class Hinge(SVMLoss):
    """
    Compute the hinge loss for classification as:

        L(y_pred, y_true) = max(0, 1 - y_true * y_pred)
    """

    _loss_type = 'classifier'

    def loss(self, y_pred, y_true):
        return np.maximum(0, 1 - y_true * y_pred)

    def _cvxpy_loss(self, theta):
        return cp.sum(cp.pos(1 - cp.multiply(self.y, self.X @ theta)))

    def loss_jacobian(self, packed_coef_inter, X_batch, y_batch):
        y_pred = np.dot(X_batch, packed_coef_inter)  # svm decision function
        idx = np.argwhere(y_batch * y_pred < 1.).ravel()
        return np.dot(y_batch[idx], X_batch[idx])

    def step_size(self, X_batch, y_batch):
        if np.array_equal(X_batch, self.X):  # no mini batches
            if not hasattr(self, '_step_size'):
                n_samples = self.X.shape[0]
                L = self.svm.C / n_samples * np.linalg.norm(self.X) ** 2
                self._step_size = 1 / L
            yield self._step_size
        else:
            n_samples = X_batch.shape[0]
            L = self.svm.C / n_samples * np.linalg.norm(X_batch) ** 2
            yield 1 / L


class SquaredHinge(Hinge):
    """
    Compute the squared hinge loss for classification as:

        L(y_pred, y_true) = max(0, 1 - y_true * y_pred)^2
    """

    def loss(self, y_pred, y_true):
        return np.square(super(SquaredHinge, self).loss(y_pred, y_true))

    def _cvxpy_loss(self, theta):
        return cp.sum(cp.square(cp.pos(1 - cp.multiply(self.y, self.X @ theta))))

    def loss_jacobian(self, packed_coef_inter, X_batch, y_batch):
        y_pred = np.dot(X_batch, packed_coef_inter)  # svm decision function
        idx = np.argwhere(y_batch * y_pred < 1.).ravel()
        return 2 * np.dot(np.maximum(0, 1 - y_batch[idx] * y_pred[idx]) * y_batch[idx], X_batch[idx])

    def step_size(self, X_batch, y_batch):
        if np.array_equal(X_batch, self.X):  # no mini batches
            if not hasattr(self, '_step_size'):
                mu = 1
                n_samples = self.X.shape[0]
                L = (1 / n_samples * mu +  # Lipschitz constant wrt the regularization term (strictly convex)
                     self.svm.C / n_samples * np.linalg.norm(self.X) ** 2)  # Lipschitz constant wrt the loss term
                self._step_size = 1 / L
            yield self._step_size
        else:
            mu = 1
            n_samples = X_batch.shape[0]
            L = (1 / n_samples * mu +  # Lipschitz constant wrt the regularization term (strictly convex)
                 self.svm.C / n_samples * np.linalg.norm(X_batch) ** 2)  # Lipschitz constant wrt the loss term
            yield 1 / L


class EpsilonInsensitive(SVMLoss):
    """
    Compute the epsilon-insensitive loss for regression as:

        L(y_pred, y_true) = max(0, abs(y_true - y_pred) - epsilon)
    """

    _loss_type = 'regressor'

    def __init__(self, svm, X, y, epsilon):
        """
        Parameters
        ----------

        svm : `SVM` instance
            The SVM estimator this loss is attached to.

        X : ndarray of shape (n_samples, n_features)
            Training data over which the loss is evaluated.

        y : ndarray of shape (n_samples,)
            Target values associated with ``X``.

        epsilon : float
            Width of the epsilon-tube within which no penalty is associated
            with points predicted within a distance epsilon from the actual value.
        """
        super(EpsilonInsensitive, self).__init__(svm, X, y)
        self.epsilon = epsilon

    def loss(self, y_pred, y_true):
        return np.maximum(0, np.abs(y_true - y_pred) - self.epsilon)

    def _cvxpy_loss(self, theta):
        return cp.sum(cp.pos(cp.abs(self.y - self.X @ theta) - self.epsilon))

    def loss_jacobian(self, packed_coef_inter, X_batch, y_batch):
        y_pred = np.dot(X_batch, packed_coef_inter)  # svm decision function
        idx = np.argwhere(np.abs(y_batch - y_pred) >= self.epsilon).ravel()
        z = y_batch[idx] - y_pred[idx]
        return np.dot(np.sign(z), X_batch[idx])  # or np.dot(np.divide(z, np.abs(z)), X_batch[idx])

    def step_size(self, X_batch, y_batch):
        if np.array_equal(X_batch, self.X):  # no mini batches
            if not hasattr(self, '_step_size'):
                n_samples = self.X.shape[0]
                L = self.svm.C / n_samples * np.linalg.norm(self.X) ** 2
                self._step_size = 1 / L
            yield self._step_size
        else:
            n_samples = X_batch.shape[0]
            L = self.svm.C / n_samples * np.linalg.norm(X_batch) ** 2
            yield 1 / L


class SquaredEpsilonInsensitive(EpsilonInsensitive):
    """
    Compute the squared epsilon-insensitive loss for regression as:

        L(y_pred, y_true) = max(0, abs(y_true - y_pred) - epsilon)^2
    """

    def loss(self, y_pred, y_true):
        return np.square(super(SquaredEpsilonInsensitive, self).loss(y_pred, y_true))

    def _cvxpy_loss(self, theta):
        return cp.sum(cp.square(cp.pos(cp.abs(self.y - self.X @ theta) - self.epsilon)))

    def loss_jacobian(self, packed_coef_inter, X_batch, y_batch):
        y_pred = np.dot(X_batch, packed_coef_inter)  # svm decision function
        idx = np.argwhere(np.abs(y_batch - y_pred) >= self.epsilon).ravel()
        z = y_batch[idx] - y_pred[idx]
        return 2 * np.dot(np.sign(z) * (np.abs(z) - self.epsilon), X_batch[idx])

    def step_size(self, X_batch, y_batch):
        if np.array_equal(X_batch, self.X):  # no mini batches
            if not hasattr(self, '_step_size'):
                mu = 1
                n_samples = self.X.shape[0]
                L = (1 / n_samples * mu +  # Lipschitz constant wrt the regularization term (strictly convex)
                     self.svm.C / n_samples * np.linalg.norm(self.X) ** 2)  # Lipschitz constant wrt the loss term
                self._step_size = 1 / L
            yield self._step_size
        else:
            mu = 1
            n_samples = X_batch.shape[0]
            L = (1 / n_samples * mu +  # Lipschitz constant wrt the regularization term (strictly convex)
                 self.svm.C / n_samples * np.linalg.norm(X_batch) ** 2)  # Lipschitz constant wrt the loss term
            yield 1 / L


hinge = Hinge
squared_hinge = SquaredHinge
epsilon_insensitive = EpsilonInsensitive
squared_epsilon_insensitive = SquaredEpsilonInsensitive
