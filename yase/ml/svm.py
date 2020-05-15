import warnings

import numpy as np
from qpsolvers import solve_qp
from sklearn.base import ClassifierMixin, BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.multiclass import unique_labels

from ..optimization import Optimizer
from ..optimization.constrained import (SMO, SMOClassifier, SMORegression, BoxConstrainedQuadratic,
                                        BoxConstrainedOptimizer, LagrangianBoxConstrainedQuadratic)
from ..optimization.unconstrained import ProximalBundle
from ..optimization.unconstrained.line_search import LineSearchOptimizer
from ..optimization.unconstrained.stochastic import StochasticOptimizer


class SVM(BaseEstimator):
    def __init__(self, kernel='rbf', degree=3., gamma='scale', coef0=0., C=1., tol=1e-3, optimizer=SMO,
                 max_iter=1000, learning_rate=0.01, momentum_type='none', momentum=0.9, max_f_eval=1000,
                 master_solver='ecos', verbose=False):
        self.kernels = {'linear': self.linear,
                        'poly': self.poly,
                        'rbf': self.rbf,
                        'laplacian': self.laplacian,
                        'sigmoid': self.sigmoid}
        if kernel not in self.kernels.keys():
            raise ValueError(f'unknown kernel type {kernel}')
        self.kernel = kernel
        if not np.isscalar(degree):
            raise ValueError('degree is not an integer scalar')
        if not degree > 0:
            raise ValueError('degree must be > 0')
        self.degree = degree
        if isinstance(gamma, str):
            if gamma not in ('scale', 'auto'):
                raise ValueError(f'unknown gamma type {gamma}')
        else:
            if not np.isscalar(gamma):
                raise ValueError('gamma is not a real scalar')
            if not gamma > 0:
                raise ValueError('gamma must be > 0')
        self.gamma = gamma
        if not np.isscalar(coef0):
            raise ValueError('coef0 is not a real scalar')
        self.coef0 = coef0
        if not np.isscalar(C):
            raise ValueError('C is not a real scalar')
        if not C >= 0:
            raise ValueError('C must be >= 0')
        self.C = C  # penalty or regularization term
        if not np.isscalar(tol):
            raise ValueError('tol is not a real scalar')
        if not tol > 0:
            raise ValueError('tol must be > 0')
        self.tol = tol  # tolerance for KKT conditions
        if not isinstance(optimizer, str) and not issubclass(optimizer, SMO) and not issubclass(optimizer, Optimizer):
            raise TypeError('optimizer is not an allowed optimizer')
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.momentum_type = momentum_type
        self.momentum = momentum
        self.max_f_eval = max_f_eval
        self.master_solver = master_solver
        self.verbose = verbose
        if kernel == 'linear':
            self.coef_ = 0.
        self.intercept_ = 0.

    # kernels

    def linear(self, X, Y=None):
        """
        Compute the linear kernel between X and Y:

            K(X, Y) = <X, Y>
        """
        if Y is None:
            Y = X
        return np.dot(X, Y.T)

    def poly(self, X, Y=None):
        """
        Compute the polynomial kernel between X and Y:

            K(X, Y) = (gamma <X, Y> + coef0)^degree
        """
        if Y is None:
            Y = X
        gamma = (1. / (X.shape[1] * X.var()) if self.gamma == 'scale' else  # auto
                 1. / X.shape[1] if isinstance(self.gamma, str) else self.gamma)
        return (gamma * np.dot(X, Y.T) + self.coef0) ** self.degree

    def rbf(self, X, Y=None):
        """
        Compute the rbf (gaussian) kernel between X and Y:

            K(x, y) = exp(-gamma ||x-y||_2^2)
        """
        if Y is None:
            Y = X
        gamma = (1. / (X.shape[1] * X.var()) if self.gamma == 'scale' else  # auto
                 1. / X.shape[1] if isinstance(self.gamma, str) else self.gamma)
        return np.exp(-gamma * np.linalg.norm(X[:, np.newaxis] - Y[np.newaxis, :], axis=2) ** 2)

    def laplacian(self, X, Y=None):
        """
        Compute the laplacian kernel between X and Y:

            K(x, y) = exp(-gamma ||x-y||_1)
        """
        if Y is None:
            Y = X
        gamma = (1. / (X.shape[1] * X.var()) if self.gamma == 'scale' else  # auto
                 1. / X.shape[1] if isinstance(self.gamma, str) else self.gamma)
        return np.exp(-gamma * np.linalg.norm(X[:, np.newaxis] - Y[np.newaxis, :], ord=1, axis=2))

    def sigmoid(self, X, Y=None):
        """
        Compute the sigmoid kernel between X and Y:

            K(X, Y) = tanh(gamma <X, Y> + coef0)
        """
        if Y is None:
            Y = X
        gamma = (1. / (X.shape[1] * X.var()) if self.gamma == 'scale' else  # auto
                 1. / X.shape[1] if isinstance(self.gamma, str) else self.gamma)
        return np.tanh(gamma * np.dot(X, Y.T) + self.coef0)


class SVC(ClassifierMixin, SVM):
    def __init__(self, kernel='rbf', degree=3., gamma='scale', coef0=0., C=1., tol=1e-3, optimizer=SMOClassifier,
                 max_iter=1000, learning_rate=0.01, momentum_type='none', momentum=0.9, max_f_eval=1000,
                 master_solver='ecos', verbose=False):
        super().__init__(kernel, degree, gamma, coef0, C, tol, optimizer, max_iter, learning_rate,
                         momentum_type, momentum, max_f_eval, master_solver, verbose)

    def fit(self, X, y):
        """
        Trains the model by solving a constrained quadratic programming problem.
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples] holding the class labels
        """
        self.labels = unique_labels(y)
        if len(self.labels) > 2:
            raise ValueError('use OneVsOneClassifier or OneVsRestClassifier from sklearn.multiclass '
                             'to train a model over more than two labels')
        y = np.where(y == self.labels[0], -1., 1.)

        n_samples = len(y)

        # kernel matrix
        K = self.kernels[self.kernel](X)

        P = K * np.outer(y, y)
        P = (P + P.T) / 2  # ensure P is symmetric
        q = -np.ones(n_samples)

        ub = np.ones(n_samples) * self.C  # upper bounds

        bcqp = BoxConstrainedQuadratic(P, q, ub)

        if self.optimizer == SMOClassifier:

            smo = SMOClassifier(bcqp, X, y, K, self.kernel, self.C, self.tol, self.verbose).minimize()
            alphas = smo.alphas
            if self.kernel == 'linear':
                self.coef_ = smo.w
            self.intercept_ = smo.b

        elif isinstance(self.optimizer, str):

            G = np.vstack((-np.identity(n_samples), np.identity(n_samples)))  # inequality matrix
            lb = np.zeros(n_samples)  # lower bounds
            h = np.hstack((lb, ub))  # inequality vector

            A = y  # equality matrix
            b = np.zeros(1)  # equality vector

            alphas = solve_qp(bcqp.Q, bcqp.q, G, h, A, b, solver=self.optimizer, verbose=self.verbose)

        elif issubclass(self.optimizer, BoxConstrainedOptimizer):

            self.optimizer = self.optimizer(f=bcqp, max_iter=self.max_iter, verbose=self.verbose)
            alphas = self.optimizer.minimize().x

        elif issubclass(self.optimizer, Optimizer):

            dual = LagrangianBoxConstrainedQuadratic(bcqp)

            if issubclass(self.optimizer, LineSearchOptimizer):

                self.optimizer = self.optimizer(f=dual, x=np.zeros(dual.ndim), max_iter=self.max_iter,
                                                max_f_eval=self.max_f_eval, verbose=self.verbose).minimize()

                if self.optimizer.status == 'stopped':
                    warnings.warn('max_iter reached but the optimization has not converged yet', ConvergenceWarning)

            elif issubclass(self.optimizer, StochasticOptimizer):

                self.optimizer = self.optimizer(f=dual, x=np.zeros(dual.ndim), step_size=self.learning_rate,
                                                epochs=self.max_iter, momentum_type=self.momentum_type,
                                                momentum=self.momentum, verbose=self.verbose)
                self.optimizer.minimize()

            elif issubclass(self.optimizer, ProximalBundle):

                self.optimizer = self.optimizer(f=dual, x=np.zeros(dual.ndim), max_iter=self.max_iter,
                                                master_solver=self.master_solver, momentum_type=self.momentum_type,
                                                momentum=self.momentum, verbose=self.verbose)
                self.optimizer.minimize()

            alphas = dual.primal_solution

        else:

            raise ValueError(f'unknown optimizer {self.optimizer}')

        sv = alphas > 1e-5
        self.support_ = np.arange(len(alphas))[sv]
        self.support_vectors_, self.sv_y, self.alphas = X[sv], y[sv], alphas[sv]
        self.dual_coef_ = self.alphas * self.sv_y

        if self.optimizer != SMOClassifier:

            if self.kernel == 'linear':
                self.coef_ = np.dot(self.dual_coef_, self.support_vectors_)

            for n in range(len(self.alphas)):
                self.intercept_ += self.sv_y[n]
                self.intercept_ -= np.sum(self.dual_coef_ * K[self.support_[n], sv])
            self.intercept_ /= len(self.alphas)

        return self

    def decision_function(self, X):
        if self.kernel != 'linear':
            return np.dot(self.dual_coef_, self.kernels[self.kernel](self.support_vectors_, X)) + self.intercept_
        return np.dot(X, self.coef_) + self.intercept_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.labels[1], self.labels[0])


class SVR(RegressorMixin, SVM):
    def __init__(self, kernel='rbf', degree=3., gamma='scale', coef0=0., C=1., tol=1e-3, epsilon=0.1,
                 optimizer=SMORegression, max_iter=1000, learning_rate=0.01, momentum_type='none', momentum=0.9,
                 max_f_eval=1000, master_solver='ecos', verbose=False):
        super().__init__(kernel, degree, gamma, coef0, C, tol, optimizer, max_iter, learning_rate,
                         momentum_type, momentum, max_f_eval, master_solver, verbose)
        self.epsilon = epsilon  # epsilon insensitive loss value

    def fit(self, X, y):
        """
        Trains the model by solving a constrained quadratic programming problem.
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples] holding the class labels
        """
        self.targets = y.shape[1] if y.ndim > 1 else 1
        if self.targets > 1:
            raise ValueError('use sklearn.multioutput.MultiOutputRegressor to train a model over more than one target')

        n_samples = len(y)

        # kernel matrix
        K = self.kernels[self.kernel](X)

        P = np.vstack((np.hstack((K, -K)),  # alphas_p, alphas_n
                       np.hstack((-K, K))))  # alphas_n, alphas_p
        P = (P + P.T) / 2  # ensure P is symmetric
        q = np.hstack((-y, y)) + self.epsilon

        ub = np.ones(2 * n_samples) * self.C  # upper bounds

        bcqp = BoxConstrainedQuadratic(P, q, ub)

        if self.optimizer == SMORegression:

            smo = SMORegression(bcqp, X, y, K, self.kernel, self.C, self.epsilon, self.tol, self.verbose).minimize()
            alphas_p, alphas_n = smo.alphas_p, smo.alphas_n
            if self.kernel == 'linear':
                self.coef_ = smo.w
            self.intercept_ = smo.b

        else:

            if isinstance(self.optimizer, str):

                G = np.vstack((-np.identity(2 * n_samples), np.identity(2 * n_samples)))  # inequality matrix
                lb = np.zeros(2 * n_samples)  # lower bounds
                h = np.hstack((lb, ub))  # inequality vector

                A = np.hstack((np.ones(n_samples), -np.ones(n_samples)))  # equality matrix
                b = np.zeros(1)  # equality vector

                alphas = solve_qp(bcqp.Q, bcqp.q, G, h, A, b, solver=self.optimizer, verbose=self.verbose)

            elif issubclass(self.optimizer, BoxConstrainedOptimizer):

                self.optimizer = self.optimizer(f=bcqp, max_iter=self.max_iter, verbose=self.verbose)
                alphas = self.optimizer.minimize().x

            elif issubclass(self.optimizer, Optimizer):

                dual = LagrangianBoxConstrainedQuadratic(bcqp)

                if issubclass(self.optimizer, LineSearchOptimizer):

                    self.optimizer = self.optimizer(f=dual, x=np.zeros(dual.ndim), max_iter=self.max_iter,
                                                    max_f_eval=self.max_f_eval, verbose=self.verbose).minimize()

                    if self.optimizer.status == 'stopped':
                        warnings.warn('max_iter reached but the optimization has not converged yet', ConvergenceWarning)

                elif issubclass(self.optimizer, StochasticOptimizer):

                    self.optimizer = self.optimizer(f=dual, x=np.zeros(dual.ndim), step_size=self.learning_rate,
                                                    epochs=self.max_iter, momentum_type=self.momentum_type,
                                                    momentum=self.momentum, verbose=self.verbose)
                    self.optimizer.minimize()

                elif issubclass(self.optimizer, ProximalBundle):

                    self.optimizer = self.optimizer(f=dual, x=np.zeros(dual.ndim), max_iter=self.max_iter,
                                                    master_solver=self.master_solver, momentum_type=self.momentum_type,
                                                    momentum=self.momentum, verbose=self.verbose)
                    self.optimizer.minimize()

                alphas = dual.primal_solution

            else:

                raise ValueError(f'unknown optimizer {self.optimizer}')

            alphas_p = alphas[:n_samples]
            alphas_n = alphas[n_samples:]

        sv = np.logical_or(alphas_p > 1e-5, alphas_n > 1e-5)
        self.support_ = np.arange(len(alphas_p))[sv]
        self.support_vectors_, self.sv_y, self.alphas_p, self.alphas_n = X[sv], y[sv], alphas_p[sv], alphas_n[sv]
        self.dual_coef_ = self.alphas_p - self.alphas_n

        if self.optimizer != SMORegression:

            if self.kernel == 'linear':
                self.coef_ = np.dot(self.dual_coef_, self.support_vectors_)

            for n in range(len(self.alphas_p)):
                self.intercept_ += self.sv_y[n]
                self.intercept_ -= np.sum(self.dual_coef_ * K[self.support_[n], sv])
            self.intercept_ -= self.epsilon
            self.intercept_ /= len(self.alphas_p)

        return self

    def predict(self, X):
        if self.kernel != 'linear':
            return np.dot(self.dual_coef_, self.kernels[self.kernel](self.support_vectors_, X)) + self.intercept_
        return np.dot(X, self.coef_) + self.intercept_
