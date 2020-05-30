import warnings

import numpy as np
from qpsolvers import solve_qp
from sklearn.base import ClassifierMixin, BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model._base import LinearClassifierMixin, SparseCoefMixin, LinearModel
from sklearn.utils.multiclass import unique_labels

from .kernels import rbf, Kernel, LinearKernel
from .losses import squared_hinge, SVMLoss, SVCLoss, SVRLoss, epsilon_insensitive
from .smo import SMO, SMOClassifier, SMORegression
from ...opti import Optimizer
from ...opti import Quadratic
from ...opti.qp import LagrangianConstrainedQuadratic
from ...opti.qp import LagrangianEqualityConstrainedQuadratic
from ...opti.qp.bcqp import BoxConstrainedQuadraticOptimizer, LagrangianBoxConstrainedQuadratic
from ...opti.unconstrained import ProximalBundle
from ...opti.unconstrained.line_search import LineSearchOptimizer
from ...opti.unconstrained.stochastic import StochasticOptimizer, StochasticGradientDescent, AdaGrad


class SVM(BaseEstimator):

    def __init__(self, C=1., tol=1e-3, optimizer=None, max_iter=1000, learning_rate=0.1,
                 momentum_type='none', momentum=0.9, batch_size=None, max_f_eval=1000,
                 shuffle=True, random_state=None, verbose=False):
        if not C > 0:
            raise ValueError('C must be > 0')
        self.C = C
        if not tol > 0:
            raise ValueError('tol must be > 0')
        self.tol = tol
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


class PrimalSVM(SVM):

    def __init__(self, C=1., tol=1e-4, loss=SVMLoss, optimizer=StochasticGradientDescent, max_iter=1000,
                 learning_rate=0.1, momentum_type='none', momentum=0.9, batch_size=None, max_f_eval=1000,
                 fit_intercept=True, shuffle=True, random_state=None, verbose=False):
        super().__init__(C, tol, optimizer, max_iter, learning_rate, momentum_type, momentum,
                         batch_size, max_f_eval, shuffle, random_state, verbose)
        self.loss = loss
        if not issubclass(self.optimizer, Optimizer):
            raise TypeError(f'{optimizer} is not an allowed optimization method')
        self.coef_ = np.zeros(0)
        self.intercept_ = 0.
        self.fit_intercept = fit_intercept

    def _unpack(self, packed_coef_inter):
        if self.fit_intercept:
            self.coef_, self.intercept_ = packed_coef_inter[:-1], packed_coef_inter[-1]
        else:
            self.coef_ = packed_coef_inter


class DualSVM(SVM):

    def __init__(self, kernel=rbf, C=1., tol=1e-3, optimizer=SMO, max_iter=1000, learning_rate=0.1,
                 momentum_type='none', momentum=0.9, batch_size=None, max_f_eval=1000, master_solver='ecos',
                 master_verbose=False, shuffle=True, random_state=None, verbose=False):
        super().__init__(C, tol, optimizer, max_iter, learning_rate, momentum_type,
                         momentum, batch_size, max_f_eval, shuffle, random_state, verbose)
        if not isinstance(kernel, Kernel):
            raise TypeError(f'{kernel} is not an allowed kernel function')
        self.kernel = kernel
        if not (isinstance(optimizer, str) or
                not issubclass(optimizer, SMO) or
                not issubclass(optimizer, Optimizer)):
            raise TypeError(f'{optimizer} is not an allowed optimization method')
        self.master_solver = master_solver
        self.master_verbose = master_verbose
        if isinstance(self.kernel, LinearKernel):
            self.coef_ = np.zeros(0)
        self.intercept_ = 0.


class PrimalSVC(LinearClassifierMixin, SparseCoefMixin, PrimalSVM):

    def __init__(self, C=1., tol=1e-4, loss=squared_hinge, penalty='l2', optimizer=StochasticGradientDescent,
                 max_iter=1000, learning_rate=0.1, momentum_type='none', momentum=0.9, batch_size=None,
                 max_f_eval=1000, fit_intercept=True, shuffle=True, random_state=None, verbose=False):
        super().__init__(C, tol, loss, optimizer, max_iter, learning_rate, momentum_type, momentum,
                         batch_size, max_f_eval, fit_intercept, shuffle, random_state, verbose)
        if not issubclass(loss, SVCLoss):
            raise TypeError(f'{loss} is not an allowed LinearSVC loss function')
        if penalty not in ('l1', 'l2'):
            raise TypeError(f'{penalty} is not an allowed penalty')
        self.penalty = penalty

    def fit(self, X, y):
        self.labels = unique_labels(y)
        if len(self.labels) > 2:
            raise ValueError('use OneVsOneClassifier or OneVsRestClassifier from sklearn.multiclass '
                             'to train a model over more than two labels')
        y = np.where(y == self.labels[0], -1., 1.)

        if self.fit_intercept:
            X_train = np.c_[X, np.ones_like(y)]
        else:
            X_train = X

        self.loss = self.loss(self, X_train, y, self.penalty)

        if issubclass(self.optimizer, LineSearchOptimizer):

            self.optimizer = self.optimizer(f=self.loss, x=np.zeros(self.loss.ndim), max_iter=self.max_iter,
                                            max_f_eval=self.max_f_eval, verbose=self.verbose).minimize()

            if self.optimizer.status == 'stopped':
                if self.optimizer.iter >= self.max_iter:
                    warnings.warn('max_iter reached but the optimization has not converged yet', ConvergenceWarning)
                elif self.optimizer.f_eval >= self.max_f_eval:
                    warnings.warn('max_f_eval reached but the optimization has not converged yet', ConvergenceWarning)

        elif issubclass(self.optimizer, StochasticOptimizer):

            self.optimizer = self.optimizer(f=self.loss, x=np.zeros(self.loss.ndim), epochs=self.max_iter,
                                            step_size=self.learning_rate, momentum_type=self.momentum_type,
                                            momentum=self.momentum, verbose=self.verbose).minimize()

        self._unpack(self.optimizer.x)

        if self.fit_intercept:
            self.loss.X = X

        return self

    def decision_function(self, X):
        return np.dot(X, self.coef_) + self.intercept_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.labels[1], self.labels[0])


class DualSVC(ClassifierMixin, DualSVM):

    def __init__(self, kernel=rbf, C=1., tol=1e-3, optimizer=SMOClassifier, max_iter=1000, learning_rate=0.1,
                 momentum_type='none', momentum=0.9, batch_size=None, max_f_eval=1000, master_solver='ecos',
                 master_verbose=False, shuffle=True, random_state=None, verbose=False):
        super().__init__(kernel, C, tol, optimizer, max_iter, learning_rate, momentum_type, momentum, batch_size,
                         max_f_eval, master_solver, master_verbose, shuffle, random_state, verbose)

    def fit(self, X, y):
        self.labels = unique_labels(y)
        if len(self.labels) > 2:
            raise ValueError('use OneVsOneClassifier or OneVsRestClassifier from sklearn.multiclass '
                             'to train a model over more than two labels')
        y = np.where(y == self.labels[0], -1., 1.)

        n_samples = len(y)

        # kernel matrix
        K = self.kernel(X)

        Q = K * np.outer(y, y)
        q = -np.ones(n_samples)

        ub = np.ones(n_samples) * self.C  # upper bounds

        self.obj = Quadratic(Q, q)

        if self.optimizer == SMOClassifier:

            self.optimizer = SMOClassifier(self.obj, X, y, K, self.kernel, self.C,
                                           self.tol, self.verbose).minimize()
            alphas = self.optimizer.alphas
            if isinstance(self.kernel, LinearKernel):
                self.coef_ = self.optimizer.w
            self.intercept_ = self.optimizer.b

        elif isinstance(self.optimizer, str):

            lb = np.zeros(n_samples)  # lower bounds
            alphas = solve_qp(Q, q, lb=lb, ub=ub, solver=self.optimizer, verbose=self.verbose)

        else:

            if issubclass(self.optimizer, BoxConstrainedQuadraticOptimizer):

                self.optimizer = self.optimizer(f=self.obj, ub=ub, max_iter=self.max_iter,
                                                verbose=self.verbose).minimize()
                alphas = self.optimizer.x

            elif issubclass(self.optimizer, Optimizer):

                self.obj = LagrangianBoxConstrainedQuadratic(self.obj, ub)

                if issubclass(self.optimizer, LineSearchOptimizer):

                    self.optimizer = self.optimizer(f=self.obj, x=np.zeros(self.obj.ndim), max_iter=self.max_iter,
                                                    max_f_eval=self.max_f_eval, verbose=self.verbose).minimize()

                    if self.optimizer.status == 'stopped':
                        if self.optimizer.iter >= self.max_iter:
                            warnings.warn('max_iter reached but the optimization has not converged yet',
                                          ConvergenceWarning)
                        elif self.optimizer.f_eval >= self.max_f_eval:
                            warnings.warn('max_f_eval reached but the optimization has not converged yet',
                                          ConvergenceWarning)

                elif issubclass(self.optimizer, ProximalBundle):

                    self.optimizer = self.optimizer(f=self.obj, x=np.zeros(self.obj.ndim), max_iter=self.max_iter,
                                                    master_solver=self.master_solver, verbose=self.verbose,
                                                    master_verbose=self.master_verbose).minimize()

                    if self.optimizer.status == 'stopped':
                        if self.optimizer.iter >= self.max_iter:
                            warnings.warn('max_iter reached but the optimization has not converged yet',
                                          ConvergenceWarning)
                        elif self.optimizer.f_eval >= self.max_f_eval:
                            warnings.warn('max_f_eval reached but the optimization has not converged yet',
                                          ConvergenceWarning)

                elif issubclass(self.optimizer, StochasticOptimizer):

                    self.optimizer = self.optimizer(f=self.obj, x=np.zeros(self.obj.ndim), epochs=self.max_iter,
                                                    step_size=self.learning_rate, momentum_type=self.momentum_type,
                                                    momentum=self.momentum, verbose=self.verbose).minimize()

                alphas = self.obj.primal_solution

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


class PrimalSVR(RegressorMixin, LinearModel, PrimalSVM):

    def __init__(self, C=1., epsilon=0., tol=1e-4, loss=epsilon_insensitive, optimizer=AdaGrad, max_iter=1000,
                 learning_rate=0.1, momentum_type='none', momentum=0.9, batch_size=None, max_f_eval=1000,
                 fit_intercept=True, shuffle=True, random_state=None, verbose=False):
        super().__init__(C, tol, loss, optimizer, max_iter, learning_rate, momentum_type, momentum,
                         batch_size, max_f_eval, fit_intercept, shuffle, random_state, verbose)
        if not issubclass(loss, SVRLoss):
            raise TypeError(f'{loss} is not an allowed LinearSVR loss function')
        if not epsilon >= 0:
            raise ValueError('epsilon must be >= 0')
        self.epsilon = epsilon

    def fit(self, X, y):
        targets = y.shape[1] if y.ndim > 1 else 1
        if targets > 1:
            raise ValueError('use sklearn.multioutput.MultiOutputRegressor '
                             'to train a model over more than one target')

        if self.fit_intercept:
            X_train = np.c_[X, np.ones_like(y)]
        else:
            X_train = X

        self.loss = self.loss(self, X_train, y, self.epsilon)

        if issubclass(self.optimizer, LineSearchOptimizer):

            self.optimizer = self.optimizer(f=self.loss, x=np.zeros(self.loss.ndim), max_iter=self.max_iter,
                                            max_f_eval=self.max_f_eval, verbose=self.verbose).minimize()

            if self.optimizer.status == 'stopped':
                if self.optimizer.iter >= self.max_iter:
                    warnings.warn('max_iter reached but the optimization has not converged yet', ConvergenceWarning)
                elif self.optimizer.f_eval >= self.max_f_eval:
                    warnings.warn('max_f_eval reached but the optimization has not converged yet', ConvergenceWarning)

        elif issubclass(self.optimizer, StochasticOptimizer):

            self.optimizer = self.optimizer(f=self.loss, x=np.zeros(self.loss.ndim), epochs=self.max_iter,
                                            step_size=self.learning_rate, momentum_type=self.momentum_type,
                                            momentum=self.momentum, verbose=self.verbose).minimize()

        self._unpack(self.optimizer.x)

        if self.fit_intercept:
            self.loss.X = X

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_


class DualSVR(RegressorMixin, DualSVM):
    def __init__(self, kernel=rbf, C=1., epsilon=0.1, tol=1e-3, optimizer=SMORegression, max_iter=1000,
                 learning_rate=0.1, momentum_type='none', momentum=0.9, batch_size=None, max_f_eval=1000,
                 master_solver='ecos', master_verbose=False, shuffle=True, random_state=None, verbose=False):
        super().__init__(kernel, C, tol, optimizer, max_iter, learning_rate, momentum_type, momentum, batch_size,
                         max_f_eval, master_solver, master_verbose, shuffle, random_state, verbose)
        if not epsilon >= 0:
            raise ValueError('epsilon must be >= 0')
        self.epsilon = epsilon

    def fit(self, X, y):
        targets = y.shape[1] if y.ndim > 1 else 1
        if targets > 1:
            raise ValueError('use sklearn.multioutput.MultiOutputRegressor '
                             'to train a model over more than one target')

        n_samples = len(y)

        # kernel matrix
        K = self.kernel(X)

        Q = np.vstack((np.hstack((K, -K)),
                       np.hstack((-K, K))))
        q = np.hstack((-y, y)) + self.epsilon

        A = np.hstack((np.ones(n_samples), -np.ones(n_samples)))  # equality matrix

        ub = np.ones(2 * n_samples) * self.C  # upper bounds

        self.obj = Quadratic(Q, q)

        if self.optimizer == SMORegression:

            self.optimizer = SMORegression(self.obj, X, y, K, self.kernel, self.C,
                                           self.epsilon, self.tol, self.verbose).minimize()
            alphas_p, alphas_n = self.optimizer.alphas_p, self.optimizer.alphas_n
            if isinstance(self.kernel, LinearKernel):
                self.coef_ = self.optimizer.w
            self.intercept_ = self.optimizer.b

        else:

            if isinstance(self.optimizer, str):

                b = np.zeros(1)  # equality vector
                lb = np.zeros(2 * n_samples)  # lower bounds
                alphas = solve_qp(Q, q, A=A, b=b, lb=lb, ub=ub, solver=self.optimizer, verbose=self.verbose)

            else:

                if issubclass(self.optimizer, BoxConstrainedQuadraticOptimizer):

                    self.obj = LagrangianEqualityConstrainedQuadratic(self.obj, A)
                    self.optimizer = self.optimizer(f=self.obj, ub=ub, max_iter=self.max_iter,
                                                    verbose=self.verbose).minimize()

                elif issubclass(self.optimizer, Optimizer):

                    self.obj = LagrangianConstrainedQuadratic(self.obj, A, ub)

                    if issubclass(self.optimizer, LineSearchOptimizer):

                        self.optimizer = self.optimizer(f=self.obj, x=np.zeros(self.obj.ndim), max_iter=self.max_iter,
                                                        max_f_eval=self.max_f_eval, verbose=self.verbose).minimize()

                        if self.optimizer.status == 'stopped':
                            if self.optimizer.iter >= self.max_iter:
                                warnings.warn('max_iter reached but the optimization has not converged yet',
                                              ConvergenceWarning)
                            elif self.optimizer.f_eval >= self.max_f_eval:
                                warnings.warn('max_f_eval reached but the optimization has not converged yet',
                                              ConvergenceWarning)

                    elif issubclass(self.optimizer, ProximalBundle):

                        self.optimizer = self.optimizer(f=self.obj, x=np.zeros(self.obj.ndim), max_iter=self.max_iter,
                                                        master_solver=self.master_solver, verbose=self.verbose,
                                                        master_verbose=self.master_verbose).minimize()

                    elif issubclass(self.optimizer, StochasticOptimizer):

                        self.optimizer = self.optimizer(f=self.obj, x=np.zeros(self.obj.ndim), epochs=self.max_iter,
                                                        step_size=self.learning_rate, momentum_type=self.momentum_type,
                                                        momentum=self.momentum, verbose=self.verbose).minimize()

                alphas = self.obj.primal_solution

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
