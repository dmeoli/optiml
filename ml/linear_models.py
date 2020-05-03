import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.base import MultiOutputMixin, RegressorMixin, BaseEstimator, ClassifierMixin

from ml.neural_network.activations import Sigmoid
from ml.neural_network.initializers import random_uniform
from ml.neural_network.losses import mean_squared_error, binary_cross_entropy, MeanSquaredError
from ml.regularizers import l2, L2
from optimization.optimization_function import OptimizationFunction
from optimization.unconstrained.proximal_bundle import ProximalBundle
from optimization.unconstrained.line_search.line_search_optimizer import LineSearchOptimizer
from optimization.unconstrained.stochastic import StochasticGradientDescent
from optimization.unconstrained.stochastic.stochastic_optimizer import StochasticOptimizer

plt.style.use('ggplot')


class LinearModelLossFunction(OptimizationFunction):

    def __init__(self, linear_model, X, y):
        super().__init__(X.shape[1], linear_model.loss.x_min, linear_model.loss.x_max,
                         linear_model.loss.y_min, linear_model.loss.y_max)
        self.X = X
        self.y = y
        self.linear_model = linear_model

    def x_star(self):
        if isinstance(self.linear_model.loss, MeanSquaredError) and isinstance(self.linear_model.regularizer, L2):
            if not hasattr(self, 'x_opt'):
                if self.linear_model.regularizer.lmbda == 0.:
                    self.x_opt = np.linalg.lstsq(self.X, self.y)[0]
                else:
                    self.x_opt = np.linalg.inv(self.X.T.dot(self.X) + np.identity(self.ndim) *
                                               self.linear_model.regularizer.lmbda).dot(self.X.T).dot(self.y)
            return self.x_opt

    def f_star(self):
        if self.x_star() is not None:
            return self.linear_model.loss(self.linear_model.predict(self.X, self.x_star()), self.y)
        return super().f_star()

    def args(self):
        return self.X, self.y

    def function(self, theta, X=None, y=None):
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        return (self.linear_model.loss(self.linear_model.predict(X, theta), y) +
                self.linear_model.regularizer(theta) / X.shape[0])

    def jacobian(self, theta, X=None, y=None):
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        return (np.dot(X.T, self.linear_model.predict(X, theta) - y) +
                self.linear_model.regularizer.jacobian(theta) / X.shape[0])


class LinearRegression(BaseEstimator, MultiOutputMixin, RegressorMixin):

    def __init__(self, loss=mean_squared_error, optimizer=StochasticGradientDescent, learning_rate=0.01,
                 momentum_type='none', momentum=0.9, max_iter=1000, batch_size=None, max_f_eval=1000,
                 regularizer=l2, master_solver='cvxopt', verbose=False, plot=False):
        self.loss = loss
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum_type = momentum_type
        self.momentum = momentum
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.max_f_eval = max_f_eval
        self.master_solver = master_solver
        self.verbose = verbose
        self.plot = plot

    def fit(self, X, y):
        self.targets = y.shape[1] if y.ndim > 1 else 1
        if self.targets > 1:
            raise ValueError('use sklearn.multioutput.MultiOutputRegressor to train a model over more than one target')

        loss = LinearModelLossFunction(self, X, y)

        if isinstance(self.optimizer, str):  # scipy optimization
            res = minimize(fun=loss.function, jac=loss.jacobian, args=loss.args(),
                           x0=random_uniform(loss.ndim), method=self.optimizer,
                           options={'disp': self.verbose,
                                    'maxiter': self.max_iter,
                                    'maxfun': self.max_f_eval})
            if res.status != 0:
                warnings.warn('max_iter reached but the optimization has not converged yet')
            self.coef_ = res.x
        else:
            if issubclass(self.optimizer, LineSearchOptimizer):
                res = self.optimizer(f=loss, max_iter=self.max_iter, max_f_eval=self.max_f_eval,
                                     verbose=self.verbose, plot=self.plot).minimize()
                if res[2] != 'optimal':
                    warnings.warn('max_iter reached but the optimization has not converged yet')
            elif issubclass(self.optimizer, StochasticOptimizer):
                res = self.optimizer(f=loss, batch_size=self.batch_size, step_size=self.learning_rate,
                                     momentum_type=self.momentum_type, momentum=self.momentum,
                                     epochs=self.max_iter, verbose=self.verbose, plot=self.plot).minimize()
            elif issubclass(self.optimizer, ProximalBundle):
                res = self.optimizer(f=loss, max_iter=self.max_iter, master_solver=self.master_solver,
                                     momentum_type=self.momentum_type, momentum=self.momentum,
                                     verbose=self.verbose, plot=self.plot).minimize()
            else:
                raise ValueError(f'unknown optimizer {self.optimizer}')

            self.coef_ = res[0]

        return self

    def predict(self, X, theta=None):
        return np.dot(X, theta if theta is not None else self.coef_)


class LogisticRegression(BaseEstimator, ClassifierMixin):

    def __init__(self, loss=binary_cross_entropy, optimizer=StochasticGradientDescent, learning_rate=0.01,
                 momentum_type='none', momentum=0.9, max_iter=1000, batch_size=None, max_f_eval=1000,
                 regularizer=l2, master_solver='cvxopt', verbose=False, plot=False):
        self.loss = loss
        self.learning_rate = learning_rate
        self.momentum_type = momentum_type
        self.momentum = momentum
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.max_f_eval = max_f_eval
        self.regularizer = regularizer
        self.master_solver = master_solver
        self.verbose = verbose
        self.plot = plot

    def fit(self, X, y):
        self.labels = np.unique(y)
        if self.labels.size > 2:
            raise ValueError('use OneVsOneClassifier or OneVsRestClassifier from sklearn.multiclass '
                             'to train a model over more than two labels')
        y = np.where(y == self.labels[0], 0, 1)

        loss = LinearModelLossFunction(self, X, y)
        if issubclass(self.optimizer, LineSearchOptimizer):
            self.coef_ = self.optimizer(f=loss, max_iter=self.max_iter, max_f_eval=self.max_f_eval,
                                        verbose=self.verbose, plot=self.plot).minimize()[0]
        elif issubclass(self.optimizer, StochasticOptimizer):
            self.coef_ = self.optimizer(f=loss, batch_size=self.batch_size, step_size=self.learning_rate,
                                        momentum_type=self.momentum_type, momentum=self.momentum,
                                        epochs=self.max_iter, verbose=self.verbose, plot=self.plot).minimize()[0]
        return self

    def decision_function(self, X, theta=None):
        return Sigmoid().function(np.dot(X, theta if theta is not None else self.coef_))

    def predict(self, X, theta=None):
        return np.where(self.decision_function(X, theta) >= 0.5, self.labels[1], self.labels[0])
