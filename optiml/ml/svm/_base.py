import warnings

import numpy as np
from qpsolvers import solve_qp
from sklearn.base import ClassifierMixin, BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.multiclass import unique_labels

from .kernels import rbf, Kernel, LinearKernel
from .losses import SVMLoss
from ...optimization import Optimizer
from ...optimization.constrained import SMO, SMOClassifier, SMORegression, LagrangianConstrainedQuadratic
from ...optimization.unconstrained import Quadratic
from ...optimization.unconstrained.line_search import LineSearchOptimizer
from ...optimization.unconstrained.stochastic import StochasticOptimizer, StochasticGradientDescent


class SVM(BaseEstimator):
    def __init__(self, kernel=rbf, C=1., optimizer=None, max_iter=1000, learning_rate=0.01, momentum_type='none',
                 momentum=0.9, batch_size=None, max_f_eval=1000, shuffle=True, random_state=None, verbose=False):
        if not issubclass(type(kernel), Kernel):
            raise TypeError(f'{kernel} is not an allowed kernel function')
        self.kernel = kernel
        if not C > 0:
            raise ValueError('C must be > 0')
        self.C = C
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.momentum_type = momentum_type
        self.momentum = momentum
        self.batch_size = batch_size
        self.max_f_eval = max_f_eval
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        if isinstance(self.kernel, LinearKernel):
            self.coef_ = 0.
        self.intercept_ = 0.


class PrimalSVM(SVM):

    def __init__(self, kernel=rbf, C=1., loss=None, optimizer=StochasticGradientDescent, max_iter=1000,
                 learning_rate=0.01, momentum_type='none', momentum=0.9, batch_size=None, max_f_eval=1000,
                 shuffle=True, random_state=None, verbose=False):
        super().__init__(kernel, C, optimizer, max_iter, learning_rate, momentum_type, momentum,
                         batch_size, max_f_eval, shuffle, random_state, verbose)
        if not issubclass(loss, SVMLoss):
            raise TypeError(f'{loss} is not an allowed svm loss function')
        self.loss = loss
        if not issubclass(self.optimizer, Optimizer):
            raise TypeError(f'{optimizer} is not an allowed optimization method')


class DualSVM(SVM):

    def __init__(self, kernel=rbf, C=1., tol=1e-3, optimizer=SMO, max_iter=1000, learning_rate=0.01,
                 momentum_type='none', momentum=0.9, batch_size=None, max_f_eval=1000, shuffle=True,
                 random_state=None, verbose=False):
        super().__init__(kernel, C, optimizer, max_iter, learning_rate, momentum_type, momentum,
                         batch_size, max_f_eval, shuffle, random_state, verbose)
        if not tol > 0:
            raise ValueError('tol must be > 0')
        self.tol = tol
        if not (isinstance(optimizer, str) or
                not issubclass(optimizer, SMO) or
                not issubclass(optimizer, Optimizer)):
            raise TypeError(f'{optimizer} is not an allowed optimization method')


class PrimalSVC(ClassifierMixin, PrimalSVM):
    pass


class DualSVC(ClassifierMixin, DualSVM):
    def __init__(self, kernel=rbf, C=1., tol=1e-3, optimizer=SMOClassifier, max_iter=1000, learning_rate=0.01,
                 momentum_type='none', momentum=0.9, batch_size=None, max_f_eval=1000, shuffle=True,
                 random_state=None, verbose=False):
        super().__init__(kernel, C, tol, optimizer, max_iter, learning_rate, momentum_type, momentum,
                         batch_size, max_f_eval, shuffle, random_state, verbose)

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
        K = self.kernel(X)

        Q = K * np.outer(y, y)
        Q = (Q + Q.T) / 2  # ensure Q is symmetric
        q = -np.ones(n_samples)

        self.quad = Quadratic(Q, q)

        if self.optimizer == SMOClassifier:

            self.optimizer = SMOClassifier(self.quad, X, y, K, self.kernel, self.C,
                                           self.tol, self.verbose).minimize()
            alphas = self.optimizer.alphas
            if isinstance(self.kernel, LinearKernel):
                self.coef_ = self.optimizer.w
            self.intercept_ = self.optimizer.b

        else:

            A = y  # equality matrix
            ub = np.ones(n_samples) * self.C  # upper bounds

            if isinstance(self.optimizer, str):

                b = np.zeros(1)  # equality vector
                lb = np.zeros(n_samples)  # lower bounds

                alphas = solve_qp(Q, q, A=A, b=b, lb=lb, ub=ub, solver=self.optimizer, verbose=self.verbose)

            elif issubclass(self.optimizer, Optimizer):

                self.quad = LagrangianConstrainedQuadratic(self.quad, A, ub)

                if issubclass(self.optimizer, LineSearchOptimizer):

                    self.optimizer = self.optimizer(f=self.quad, x=np.zeros(self.quad.ndim), max_iter=self.max_iter,
                                                    max_f_eval=self.max_f_eval, verbose=self.verbose).minimize()

                    if self.optimizer.status == 'stopped':
                        warnings.warn('max_iter reached but the optimization has not converged yet', ConvergenceWarning)

                elif issubclass(self.optimizer, StochasticOptimizer):

                    self.optimizer = self.optimizer(f=self.quad, x=np.zeros(self.quad.ndim), epochs=self.max_iter,
                                                    step_size=self.learning_rate, momentum_type=self.momentum_type,
                                                    momentum=self.momentum, verbose=self.verbose).minimize()

        sv = alphas > 1e-5
        self.support_ = np.arange(len(alphas))[sv]
        self.support_vectors_, self.sv_y, self.alphas = X[sv], y[sv], alphas[sv]
        self.dual_coef_ = self.alphas * self.sv_y

        if self.optimizer != SMOClassifier:

            if isinstance(self.kernel, LinearKernel):
                self.coef_ = np.dot(self.dual_coef_, self.support_vectors_)

            for n in range(len(self.alphas)):
                self.intercept_ += self.sv_y[n]
                self.intercept_ -= np.sum(self.dual_coef_ * K[self.support_[n], sv])
            self.intercept_ /= len(self.alphas)

        return self

    def decision_function(self, X):
        if not isinstance(self.kernel, LinearKernel):
            return np.dot(self.dual_coef_, self.kernel(self.support_vectors_, X)) + self.intercept_
        return np.dot(X, self.coef_) + self.intercept_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.labels[1], self.labels[0])


class PrimalSVR(RegressorMixin, PrimalSVM):
    pass


class DualSVR(RegressorMixin, DualSVM):
    def __init__(self, kernel=rbf, C=1., epsilon=0.1, tol=1e-3, optimizer=SMORegression, max_iter=1000,
                 learning_rate=0.01, momentum_type='none', momentum=0.9, batch_size=None, max_f_eval=1000,
                 shuffle=True, random_state=None, verbose=False):
        super().__init__(kernel, C, tol, optimizer, max_iter, learning_rate, momentum_type, momentum,
                         batch_size, max_f_eval, shuffle, random_state, verbose)
        if not epsilon > 0:
            raise ValueError('epsilon must be > 0')
        self.epsilon = epsilon

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
        K = self.kernel(X)

        Q = np.vstack((np.hstack((K, -K)),  # alphas_p, alphas_n
                       np.hstack((-K, K))))  # alphas_n, alphas_p
        Q = (Q + Q.T) / 2  # ensure Q is symmetric
        q = np.hstack((-y, y)) + self.epsilon

        self.quad = Quadratic(Q, q)

        if self.optimizer == SMORegression:

            self.optimizer = SMORegression(self.quad, X, y, K, self.kernel, self.C,
                                           self.epsilon, self.tol, self.verbose).minimize()
            alphas_p, alphas_n = self.optimizer.alphas_p, self.optimizer.alphas_n
            if isinstance(self.kernel, LinearKernel):
                self.coef_ = self.optimizer.w
            self.intercept_ = self.optimizer.b

        else:

            A = np.hstack((np.ones(n_samples), -np.ones(n_samples)))  # equality matrix
            ub = np.ones(2 * n_samples) * self.C  # upper bounds

            if isinstance(self.optimizer, str):

                lb = np.zeros(2 * n_samples)  # lower bounds
                b = np.zeros(1)  # equality vector

                alphas = solve_qp(Q, q, A=A, b=b, lb=lb, ub=ub, solver=self.optimizer, verbose=self.verbose)

            elif issubclass(self.optimizer, Optimizer):

                self.quad = LagrangianConstrainedQuadratic(self.quad, A, ub)

                if issubclass(self.optimizer, LineSearchOptimizer):

                    self.optimizer = self.optimizer(f=self.quad, x=np.zeros(self.quad.ndim), max_iter=self.max_iter,
                                                    max_f_eval=self.max_f_eval, verbose=self.verbose).minimize()

                    if self.optimizer.status == 'stopped':
                        warnings.warn('max_iter reached but the optimization has not converged yet', ConvergenceWarning)

                elif issubclass(self.optimizer, StochasticOptimizer):

                    self.optimizer = self.optimizer(f=self.quad, x=np.zeros(self.quad.ndim), epochs=self.max_iter,
                                                    step_size=self.learning_rate, momentum_type=self.momentum_type,
                                                    momentum=self.momentum, verbose=self.verbose).minimize()

            alphas_p = alphas[:n_samples]
            alphas_n = alphas[n_samples:]

        sv = np.logical_or(alphas_p > 1e-5, alphas_n > 1e-5)
        self.support_ = np.arange(len(alphas_p))[sv]
        self.support_vectors_, self.sv_y, self.alphas_p, self.alphas_n = X[sv], y[sv], alphas_p[sv], alphas_n[sv]
        self.dual_coef_ = self.alphas_p - self.alphas_n

        if self.optimizer != SMORegression:

            if isinstance(self.kernel, LinearKernel):
                self.coef_ = np.dot(self.dual_coef_, self.support_vectors_)

            for n in range(len(self.alphas_p)):
                self.intercept_ += self.sv_y[n]
                self.intercept_ -= np.sum(self.dual_coef_ * K[self.support_[n], sv])
            self.intercept_ -= self.epsilon
            self.intercept_ /= len(self.alphas_p)

        return self

    def predict(self, X):
        if not isinstance(self.kernel, LinearKernel):
            return np.dot(self.dual_coef_, self.kernel(self.support_vectors_, X)) + self.intercept_
        return np.dot(X, self.coef_) + self.intercept_
