import re
import time
import warnings
from abc import ABC
from io import StringIO

import numpy as np
from qpsolvers import solve_qp
from sklearn.base import ClassifierMixin, BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model._base import LinearClassifierMixin, SparseCoefMixin, LinearModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from wurlitzer import pipes, STDOUT

from .kernels import gaussian, Kernel, LinearKernel
from .losses import hinge, squared_hinge, SVMLoss, epsilon_insensitive, squared_epsilon_insensitive, Hinge, \
    SquaredHinge, SquaredEpsilonInsensitive, EpsilonInsensitive
from .smo import SMO, SMOClassifier, SMORegression
from ...opti import Optimizer
from ...opti import Quadratic
from ...opti.constrained import BoxConstrainedQuadraticOptimizer, LagrangianBoxConstrainedQuadratic
from ...opti.constrained._base import LagrangianEqualityBoxConstrainedQuadratic, LagrangianEqualityConstrainedQuadratic, \
    LagrangianQuadratic
from ...opti.unconstrained import ProximalBundle
from ...opti.unconstrained.line_search import LineSearchOptimizer
from ...opti.unconstrained.stochastic import StochasticOptimizer, StochasticGradientDescent, StochasticMomentumOptimizer


class SVM(BaseEstimator, ABC):
    """
    Base abstract class for all SVM-type estimator.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    optimizer : LineSearchOptimizer or StochasticOptimizer subclass, default=None
        The solver for optimization. It can be a subclass of the `LineSearchOptimizer`
        which can converge faster and perform better for small datasets, e.g., the
        `LBFGS` quasi-Newton method or, alternatively, a subclass of the `StochasticOptimizer`
        e.g, the `StochasticGradientDescent` or `Adam`, which works well on relatively
        large datasets (with thousands of training samples or more) in terms of both
        training time and validation score.

    max_iter : int, default=1000
        Maximum number of iterations. The solver iterates until convergence
        (determined by ``tol``) or this number of iterations. If the optimizer
        is a subclass of `StochasticOptimizer`, this value determines the number
        of epochs (how many times each data point will be used), not the number
        of gradient steps.

    learning_rate : double, default=0.1
        The initial learning rate used for weight update. It controls the
        step-size in updating the weights. Only used when solver is a
        subclass of `StochasticOptimizer`.

    momentum_type : {'none', 'standard', 'nesterov'}, default='none'
        Momentum type used for weight update. Only used when solver is
        a subclass of `StochasticOptimizer`.

    momentum : float, default=0.9
        Momentum for weight update. Should be between 0 and 1. Only used when
        solver is a subclass of `StochasticOptimizer`.

    max_f_eval : int, default=15000
        Only used when ``optimizer`` is a subclass of `LineSearchOptimizer`.
        Maximum number of loss function calls. The solver iterates until
        convergence (determined by ``tol``), number of iterations reaches
        ``max_iter``, or this number of loss function calls. Note that number
        of loss function calls will be greater than or equal to the number
        of iterations.

    mu :

    master_solver :

    master_verbose : bool or int, default=False

    verbose : bool or int, default=False
        Controls the verbosity of progress messages to stdout. Use a boolean value
        to switch on/off or an int value to show progress each ``verbose`` time
        optimization steps.
    """

    def __init__(self,
                 loss=SVMLoss,
                 C=1.,
                 tol=1e-3,
                 optimizer=None,
                 max_iter=1000,
                 learning_rate=0.1,
                 momentum_type='none',
                 momentum=0.9,
                 max_f_eval=15000,
                 fit_intercept=True,
                 mu=1,
                 master_solver='ecos',
                 master_verbose=False,
                 verbose=False):
        if not C > 0:
            raise ValueError('C must be > 0')
        self.C = C
        if not tol > 0:
            raise ValueError('tol must be > 0')
        self.tol = tol
        self.loss = loss
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.momentum_type = momentum_type
        self.momentum = momentum
        self.max_f_eval = max_f_eval
        self.fit_intercept = fit_intercept
        self.intercept_ = 0.
        self.mu = mu
        self.master_solver = master_solver
        self.master_verbose = master_verbose
        self.verbose = verbose

    def fit(self, X, y):
        raise NotImplementedError


class PrimalSVM(SVM, ABC):
    """

    Parameters
    ----------

    shuffle : bool, default=True
    Whether to shuffle samples for batch sampling in each iteration. Only
    used when the ``optimizer`` is a subclass of `StochasticOptimizer`.

    random_state : int, default=None
        Controls the pseudo random number generation for train-test split if
        early stopping is used and shuffling the data for batch sampling when
        an instance of StochasticOptimizer class is used as ``optimizer`` value.
        Pass an int for reproducible output across multiple function calls.
    """

    def __init__(self,
                 loss=SVMLoss,
                 C=1.,
                 tol=1e-4,
                 optimizer=StochasticGradientDescent,
                 max_iter=1000,
                 learning_rate=0.1,
                 momentum_type='none',
                 momentum=0.9,
                 validation_split=0.,
                 batch_size=None,
                 max_f_eval=15000,
                 early_stopping=False,
                 patience=5,
                 fit_intercept=True,
                 shuffle=True,
                 random_state=None,
                 mu=1,
                 master_solver='ecos',
                 master_verbose=False,
                 verbose=False):
        super().__init__(loss=loss,
                         C=C,
                         tol=tol,
                         optimizer=optimizer,
                         max_iter=max_iter,
                         learning_rate=learning_rate,
                         momentum_type=momentum_type,
                         momentum=momentum,
                         max_f_eval=max_f_eval,
                         fit_intercept=fit_intercept,
                         mu=mu,
                         master_solver=master_solver,
                         master_verbose=master_verbose,
                         verbose=verbose)
        if not issubclass(self.optimizer, Optimizer):
            raise TypeError(f'{optimizer} is not an allowed optimization method')
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.shuffle = shuffle
        self.random_state = random_state
        self.coef_ = np.zeros(0)
        if issubclass(self.optimizer, StochasticOptimizer):
            self.train_loss_history = []
            self.train_time_history = []
            self.train_score_history = []
            self._no_improvement_count = 0
            self._avg_epoch_loss = 0
            if self.validation_split:
                self.val_loss_history = []
                self.val_score_history = []
                self.best_val_score = -np.inf
            else:
                self.best_loss = np.inf

    def _unpack(self, packed_coef_inter):
        if self.fit_intercept:
            self.coef_, self.intercept_ = packed_coef_inter[:-1], packed_coef_inter[-1]
        else:
            self.coef_ = packed_coef_inter

    def _store_train_val_info(self, opt, X_batch, y_batch, X_val, y_val):
        self._unpack(opt.x)
        self._avg_epoch_loss += opt.f_x * X_batch.shape[0]
        if opt.is_batch_end():
            self._avg_epoch_loss /= opt.f.X.shape[0]  # n_samples
            self.train_loss_history.append(self._avg_epoch_loss)
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            self.train_time_history.append(elapsed_time)
            if opt.is_verbose() and opt.epoch != opt.iter:
                print('\tavg_loss: {: 1.4e}'.format(self._avg_epoch_loss), end='')
            self._avg_epoch_loss = 0.
            if self.validation_split:
                val_loss = self.loss.function(opt.x, X_val, y_val)
                self.val_loss_history.append(val_loss)
                if opt.is_verbose():
                    print('\tval_loss: {: 1.4e}'.format(val_loss), end='')

    def _update_no_improvement_count(self, opt):
        if self.early_stopping:

            if self.validation_split:  # monitor val_score

                if self.val_score_history[-1] < self.best_val_score + self.tol:
                    self._no_improvement_count += 1
                else:
                    self._no_improvement_count = 0
                if self.val_score_history[-1] > self.best_val_score:
                    self.best_val_score = self.val_score_history[-1]
                    self._best_coef = self.coef_.copy()

            else:  # monitor train_loss

                if self.train_loss_history[-1] > self.best_loss - self.tol:
                    self._no_improvement_count += 1
                else:
                    self._no_improvement_count = 0
                if self.train_loss_history[-1] < self.best_loss:
                    self.best_loss = self.train_loss_history[-1]

            if self._no_improvement_count >= self.patience:

                if self.validation_split:
                    opt.x = self._best_coef

                if self.verbose:
                    if self.validation_split:
                        print(f'\ntraining stopped since validation score did not improve more than '
                              f'tol={self.tol} for {self.patience} consecutive epochs')
                    else:
                        print('\ntraining stopped since training loss did not improve more than '
                              f'tol={self.tol} for {self.patience} consecutive epochs')

                raise StopIteration


class DualSVM(SVM, ABC):
    """

    Parameters
    ----------

    kernel : Kernel instance like {linear, poly, gaussian, laplacian, sigmoid}, default=gaussian
        Specifies the kernel type to be used in the algorithm.
        It must be one of linear, poly, gaussian, laplacian, sigmoid or
        a custom one which extend the method ``__call__`` of the ``Kernel`` class.
        If none is given, 'gaussian' will be used. If a custom is given it is
        used to pre-compute the kernel matrix from data matrices; that matrix
        should be an array of shape ``(n_samples, n_samples)``.
    """

    def __init__(self,
                 loss=SVMLoss,
                 kernel=gaussian,
                 C=1.,
                 tol=1e-3,
                 optimizer=SMO,
                 max_iter=1000,
                 learning_rate=0.1,
                 momentum_type='none',
                 momentum=0.9,
                 max_f_eval=15000,
                 fit_intercept=True,
                 mu=1,
                 master_solver='ecos',
                 master_verbose=False,
                 lagrangian_solver='minres',
                 lagrangian_solver_verbose=False,
                 verbose=False):
        super().__init__(loss=loss,
                         C=C,
                         tol=tol,
                         optimizer=optimizer,
                         max_iter=max_iter,
                         learning_rate=learning_rate,
                         momentum_type=momentum_type,
                         momentum=momentum,
                         max_f_eval=max_f_eval,
                         fit_intercept=fit_intercept,
                         mu=mu,
                         master_solver=master_solver,
                         master_verbose=master_verbose,
                         verbose=verbose)
        if not isinstance(kernel, Kernel):
            raise TypeError(f'{kernel} is not an allowed kernel function')
        self.kernel = kernel
        if not (isinstance(optimizer, str) or
                not issubclass(optimizer, SMO) or
                not issubclass(optimizer, Optimizer)):
            raise TypeError(f'{optimizer} is not an allowed optimization method')
        if isinstance(self.kernel, LinearKernel):
            self.coef_ = np.zeros(0)
        self.lagrangian_solver = lagrangian_solver
        self.lagrangian_solver_verbose = lagrangian_solver_verbose
        if not isinstance(optimizer, str) and issubclass(optimizer, StochasticOptimizer):
            self.train_loss_history = []
            self.train_time_history = []

    def _store_train_info(self, opt):
        if opt.is_lagrangian_dual():
            self.train_loss_history.append(opt.primal_f_x)
        else:
            self.train_loss_history.append(opt.f_x)
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        self.train_time_history.append(elapsed_time)


class PrimalSVC(LinearClassifierMixin, SparseCoefMixin, PrimalSVM):

    def __init__(self,
                 loss=squared_hinge,
                 C=1.,
                 tol=1e-4,
                 optimizer=StochasticGradientDescent,
                 max_iter=1000,
                 learning_rate=0.1,
                 momentum_type='none',
                 momentum=0.9,
                 validation_split=0.,
                 batch_size=None,
                 max_f_eval=15000,
                 early_stopping=False,
                 patience=5,
                 fit_intercept=True,
                 shuffle=True,
                 random_state=None,
                 mu=1,
                 master_solver='ecos',
                 master_verbose=False,
                 verbose=False):
        super().__init__(loss=loss,
                         C=C,
                         tol=tol,
                         optimizer=optimizer,
                         max_iter=max_iter,
                         learning_rate=learning_rate,
                         momentum_type=momentum_type,
                         momentum=momentum,
                         validation_split=validation_split,
                         batch_size=batch_size,
                         max_f_eval=max_f_eval,
                         early_stopping=early_stopping,
                         patience=patience,
                         fit_intercept=fit_intercept,
                         shuffle=shuffle,
                         random_state=random_state,
                         mu=mu,
                         master_solver=master_solver,
                         master_verbose=master_verbose,
                         verbose=verbose)
        if not loss._loss_type == 'classifier':
            raise TypeError(f'{loss} is not an allowed SVC loss function')
        self.lb = LabelBinarizer(neg_label=-1)

    def _store_train_val_info(self, opt, X_batch, y_batch, X_val, y_val):
        super()._store_train_val_info(opt, X_batch, y_batch, X_val, y_val)
        if opt.is_batch_end():
            acc = self.score(X_batch[:, :-1], y_batch)
            self.train_score_history.append(acc)
            if opt.is_verbose():
                print('\tacc: {:1.4f}'.format(acc), end='')
            if self.validation_split:
                val_acc = self.score(X_val[:, :-1], y_val)
                self.val_score_history.append(val_acc)
                if opt.is_verbose():
                    print('\tval_acc: {:1.4f}'.format(val_acc), end='')
            self._update_no_improvement_count(opt)

    def fit(self, X, y):
        self.start_time = time.time()
        self.lb.fit(y)
        if len(self.lb.classes_) > 2:
            raise ValueError('use OneVsOneClassifier or OneVsRestClassifier from sklearn.multiclass '
                             'to train a model over more than two labels')
        y = self.lb.transform(y).ravel()

        if issubclass(self.optimizer, LineSearchOptimizer):

            if self.fit_intercept:
                X_biased = np.c_[X, np.ones_like(y)]
            else:
                X_biased = X

            self.loss = self.loss(self, X_biased, y)
            self.optimizer = self.optimizer(f=self.loss,
                                            max_iter=self.max_iter,
                                            max_f_eval=self.max_f_eval,
                                            random_state=self.random_state,
                                            verbose=self.verbose).minimize()

            if self.optimizer.status == 'stopped':
                if self.optimizer.iter >= self.max_iter:
                    warnings.warn('max_iter reached but the optimization has not converged yet', ConvergenceWarning)
                elif self.optimizer.f_eval >= self.max_f_eval:
                    warnings.warn('max_f_eval reached but the optimization has not converged yet', ConvergenceWarning)

            self._unpack(self.optimizer.x)

        elif issubclass(self.optimizer, ProximalBundle):

            if self.fit_intercept:
                X_biased = np.c_[X, np.ones_like(y)]
            else:
                X_biased = X

            self.loss = self.loss(self, X_biased, y)
            self.optimizer = self.optimizer(f=self.loss,
                                            mu=self.mu,
                                            max_iter=self.max_iter,
                                            master_solver=self.master_solver,
                                            master_verbose=self.master_verbose,
                                            random_state=self.random_state,
                                            verbose=self.verbose).minimize()

            if self.optimizer.status == 'error':
                warnings.warn('failure while computing direction for the master problem', ConvergenceWarning)

            self._unpack(self.optimizer.x)

        elif issubclass(self.optimizer, StochasticOptimizer):

            if self.validation_split:
                X, X_val, y, y_val = train_test_split(X, y,
                                                      test_size=self.validation_split,
                                                      random_state=self.random_state)

                if self.fit_intercept:
                    X_val_biased = np.c_[X_val, np.ones_like(y_val)]
                else:
                    X_val_biased = X_val

            else:
                X_val_biased = None
                y_val = None

            if self.fit_intercept:
                X_biased = np.c_[X, np.ones_like(y)]
            else:
                X_biased = X

            self.loss = self.loss(self, X_biased, y)

            if issubclass(self.optimizer, StochasticMomentumOptimizer):

                self.optimizer = self.optimizer(f=self.loss,
                                                epochs=self.max_iter,
                                                step_size=self.learning_rate,
                                                momentum_type=self.momentum_type,
                                                momentum=self.momentum,
                                                batch_size=self.batch_size,
                                                callback=self._store_train_val_info,
                                                callback_args=(X_val_biased, y_val),
                                                shuffle=self.shuffle,
                                                random_state=self.random_state,
                                                verbose=self.verbose).minimize()
            else:

                self.optimizer = self.optimizer(f=self.loss,
                                                epochs=self.max_iter,
                                                step_size=self.learning_rate,
                                                batch_size=self.batch_size,
                                                callback=self._store_train_val_info,
                                                callback_args=(X_val_biased, y_val),
                                                shuffle=self.shuffle,
                                                random_state=self.random_state,
                                                verbose=self.verbose).minimize()

        else:

            raise TypeError(f'{self.optimizer} is not an allowed optimizer')

        if self.fit_intercept and X.shape[1] > 1:
            self.loss.X = X

        return self

    def decision_function(self, X):
        return np.dot(X, self.coef_) + self.intercept_

    def predict(self, X):
        return self.lb.inverse_transform(self.decision_function(X))


class DualSVC(ClassifierMixin, DualSVM):

    def __init__(self,
                 loss=hinge,
                 kernel=gaussian,
                 C=1.,
                 tol=1e-3,
                 optimizer=SMO,
                 max_iter=1000,
                 learning_rate=0.1,
                 momentum_type='none',
                 momentum=0.9,
                 max_f_eval=15000,
                 fit_intercept=True,
                 mu=1,
                 master_solver='ecos',
                 master_verbose=False,
                 lagrangian_solver='minres',
                 lagrangian_solver_verbose=False,
                 verbose=False):
        super().__init__(loss=loss,
                         kernel=kernel,
                         C=C,
                         tol=tol,
                         optimizer=optimizer,
                         max_iter=max_iter,
                         learning_rate=learning_rate,
                         momentum_type=momentum_type,
                         momentum=momentum,
                         max_f_eval=max_f_eval,
                         fit_intercept=fit_intercept,
                         mu=mu,
                         master_solver=master_solver,
                         master_verbose=master_verbose,
                         lagrangian_solver=lagrangian_solver,
                         lagrangian_solver_verbose=lagrangian_solver_verbose,
                         verbose=verbose)
        if not loss._loss_type == 'classifier':
            raise TypeError(f'{loss} is not an allowed SVC loss function')
        self.lb = LabelBinarizer(neg_label=-1)

    def fit(self, X, y):
        self.start_time = time.time()
        self.lb.fit(y)
        if len(self.lb.classes_) > 2:
            raise ValueError('use OneVsOneClassifier or OneVsRestClassifier from sklearn.multiclass '
                             'to train a model over more than two labels')
        y = self.lb.transform(y).ravel()

        n_samples = len(y)

        # kernel matrix
        K = self.kernel(X)

        Q = K * np.outer(y, y)
        q = -np.ones(n_samples)

        if self.loss == Hinge:

            ub = np.ones(n_samples) * self.C  # upper bounds

            if self.optimizer == 'smo' or self.optimizer == SMO:

                self.obj = Quadratic(Q, q)

                self.optimizer = SMOClassifier(self.obj, X, y, K, self.kernel, self.C,
                                               self.tol, self.verbose).minimize()
                self.alphas_ = self.optimizer.alphas
                if isinstance(self.kernel, LinearKernel):
                    self.coef_ = self.optimizer.w
                self.intercept_ = self.optimizer.b

            elif isinstance(self.optimizer, str):

                lb = np.zeros(n_samples)  # lower bounds

                if not self.fit_intercept:

                    self.obj = Quadratic(Q, q)

                    out = StringIO()
                    with pipes(stdout=out, stderr=STDOUT):
                        self.alphas_ = solve_qp(P=Q,
                                                q=q,
                                                A=y.astype(float),
                                                b=np.zeros(1),
                                                lb=lb,
                                                ub=ub,
                                                solver=self.optimizer,
                                                verbose=True)

                else:

                    Q += np.outer(y, y)
                    self.obj = Quadratic(Q, q)

                    out = StringIO()
                    with pipes(stdout=out, stderr=STDOUT):
                        self.alphas_ = solve_qp(P=Q,
                                                q=q,
                                                lb=lb,
                                                ub=ub,
                                                solver=self.optimizer,
                                                verbose=True)

                stdout = out.getvalue()
                if stdout:
                    self.iter = int(max(re.findall(r'(\d+):', stdout)))
                    if self.verbose:
                        print(stdout)

            else:

                if issubclass(self.optimizer, BoxConstrainedQuadraticOptimizer):

                    Q += np.outer(y, y)
                    self.obj = Quadratic(Q, q)

                    self.optimizer = self.optimizer(quad=self.obj,
                                                    ub=ub,
                                                    max_iter=self.max_iter,
                                                    verbose=self.verbose).minimize()

                elif issubclass(self.optimizer, Optimizer):

                    if not self.fit_intercept:

                        self.obj = LagrangianEqualityBoxConstrainedQuadratic(primal=Quadratic(Q, q),
                                                                             A=y,
                                                                             ub=ub,
                                                                             lagrangian_solver=self.lagrangian_solver,
                                                                             lagrangian_solver_verbose=self.lagrangian_solver_verbose)

                    else:

                        Q += np.outer(y, y)
                        self.obj = LagrangianBoxConstrainedQuadratic(primal=Quadratic(Q, q),
                                                                     ub=ub,
                                                                     lagrangian_solver=self.lagrangian_solver,
                                                                     lagrangian_solver_verbose=self.lagrangian_solver_verbose)

                    if issubclass(self.optimizer, LineSearchOptimizer):

                        self.optimizer = self.optimizer(f=self.obj,
                                                        max_iter=self.max_iter,
                                                        max_f_eval=self.max_f_eval,
                                                        verbose=self.verbose).minimize()

                        if self.optimizer.status == 'stopped':
                            if self.optimizer.iter >= self.max_iter:
                                warnings.warn('max_iter reached but the optimization has not converged yet',
                                              ConvergenceWarning)
                            elif self.optimizer.f_eval >= self.max_f_eval:
                                warnings.warn('max_f_eval reached but the optimization has not converged yet',
                                              ConvergenceWarning)

                    elif issubclass(self.optimizer, ProximalBundle):

                        self.optimizer = self.optimizer(f=self.obj,
                                                        mu=self.mu,
                                                        max_iter=self.max_iter,
                                                        master_solver=self.master_solver,
                                                        master_verbose=self.master_verbose,
                                                        verbose=self.verbose).minimize()

                        if self.optimizer.status == 'error':
                            warnings.warn('failure while computing direction for the master problem',
                                          ConvergenceWarning)

                    elif issubclass(self.optimizer, StochasticOptimizer):

                        if issubclass(self.optimizer, StochasticMomentumOptimizer):

                            self.optimizer = self.optimizer(f=self.obj,
                                                            step_size=self.learning_rate,
                                                            epochs=self.max_iter,
                                                            momentum_type=self.momentum_type,
                                                            momentum=self.momentum,
                                                            callback=self._store_train_info,
                                                            verbose=self.verbose).minimize()

                        else:

                            self.optimizer = self.optimizer(f=self.obj,
                                                            step_size=self.learning_rate,
                                                            epochs=self.max_iter,
                                                            callback=self._store_train_info,
                                                            verbose=self.verbose).minimize()

                    else:

                        raise TypeError(f'{self.optimizer} is not an allowed optimizer')

                self.alphas_ = self.optimizer.primal_x if self.optimizer.is_lagrangian_dual() else self.optimizer.x

        elif self.loss == SquaredHinge:

            D = np.diag(np.ones(n_samples) / (2 * self.C))
            Q += D

            if isinstance(self.optimizer, str):

                lb = np.zeros(n_samples)  # lower bounds

                if not self.fit_intercept:

                    self.obj = Quadratic(Q, q)

                    out = StringIO()
                    with pipes(stdout=out, stderr=STDOUT):
                        self.alphas_ = solve_qp(P=Q,
                                                q=q,
                                                A=y.astype(float),
                                                b=np.zeros(1),
                                                lb=lb,
                                                solver=self.optimizer,
                                                verbose=True)

                else:

                    Q += np.outer(y, y)
                    self.obj = Quadratic(Q, q)

                    out = StringIO()
                    with pipes(stdout=out, stderr=STDOUT):
                        self.alphas_ = solve_qp(P=Q,
                                                q=q,
                                                lb=lb,
                                                solver=self.optimizer,
                                                verbose=True)

                stdout = out.getvalue()
                if stdout:
                    self.iter = int(max(re.findall(r'(\d+):', stdout)))
                    if self.verbose:
                        print(stdout)

            else:

                if issubclass(self.optimizer, Optimizer):

                    if not self.fit_intercept:

                        self.obj = LagrangianEqualityConstrainedQuadratic(primal=Quadratic(Q, q),
                                                                          A=y,
                                                                          lagrangian_solver=self.lagrangian_solver,
                                                                          lagrangian_solver_verbose=self.lagrangian_solver_verbose)

                    else:

                        Q += np.outer(y, y)
                        self.obj = LagrangianQuadratic(primal=Quadratic(Q, q),
                                                       lagrangian_solver=self.lagrangian_solver,
                                                       lagrangian_solver_verbose=self.lagrangian_solver_verbose)

                    if issubclass(self.optimizer, LineSearchOptimizer):

                        self.optimizer = self.optimizer(f=self.obj,
                                                        max_iter=self.max_iter,
                                                        max_f_eval=self.max_f_eval,
                                                        verbose=self.verbose).minimize()

                        if self.optimizer.status == 'stopped':
                            if self.optimizer.iter >= self.max_iter:
                                warnings.warn('max_iter reached but the optimization has not converged yet',
                                              ConvergenceWarning)
                            elif self.optimizer.f_eval >= self.max_f_eval:
                                warnings.warn('max_f_eval reached but the optimization has not converged yet',
                                              ConvergenceWarning)

                    elif issubclass(self.optimizer, ProximalBundle):

                        self.optimizer = self.optimizer(f=self.obj,
                                                        mu=self.mu,
                                                        max_iter=self.max_iter,
                                                        master_solver=self.master_solver,
                                                        master_verbose=self.master_verbose,
                                                        verbose=self.verbose).minimize()

                        if self.optimizer.status == 'error':
                            warnings.warn('failure while computing direction for the master problem',
                                          ConvergenceWarning)

                    elif issubclass(self.optimizer, StochasticOptimizer):

                        if issubclass(self.optimizer, StochasticMomentumOptimizer):

                            self.optimizer = self.optimizer(f=self.obj,
                                                            step_size=self.learning_rate,
                                                            epochs=self.max_iter,
                                                            momentum_type=self.momentum_type,
                                                            momentum=self.momentum,
                                                            callback=self._store_train_info,
                                                            verbose=self.verbose).minimize()

                        else:

                            self.optimizer = self.optimizer(f=self.obj,
                                                            step_size=self.learning_rate,
                                                            epochs=self.max_iter,
                                                            callback=self._store_train_info,
                                                            verbose=self.verbose).minimize()

                    else:

                        raise TypeError(f'{self.optimizer} is not an allowed optimizer')

                self.alphas_ = self.optimizer.primal_x if self.optimizer.is_lagrangian_dual() else self.optimizer.x

        else:

            raise TypeError(f'{self.loss} is not an allowed loss')

        sv = self.alphas_ > 1e-6
        self.support_ = np.arange(len(self.alphas_))[sv]
        self.support_vectors_, self.sv_y, self.alphas = X[sv], y[sv], self.alphas_[sv]
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
        return self.lb.inverse_transform(self.decision_function(X))


class PrimalSVR(RegressorMixin, LinearModel, PrimalSVM):

    def __init__(self,
                 loss=squared_epsilon_insensitive,
                 epsilon=0.,
                 C=1.,
                 tol=1e-4,
                 optimizer=StochasticGradientDescent,
                 max_iter=1000,
                 learning_rate=0.1,
                 momentum_type='none',
                 momentum=0.9,
                 validation_split=0.,
                 batch_size=None,
                 max_f_eval=15000,
                 early_stopping=False,
                 patience=5,
                 fit_intercept=True,
                 shuffle=True,
                 random_state=None,
                 mu=1,
                 master_solver='ecos',
                 master_verbose=False,
                 verbose=False):
        super().__init__(loss=loss,
                         C=C,
                         tol=tol,
                         optimizer=optimizer,
                         max_iter=max_iter,
                         learning_rate=learning_rate,
                         momentum_type=momentum_type,
                         momentum=momentum,
                         validation_split=validation_split,
                         batch_size=batch_size,
                         max_f_eval=max_f_eval,
                         early_stopping=early_stopping,
                         patience=patience,
                         fit_intercept=fit_intercept,
                         shuffle=shuffle,
                         random_state=random_state,
                         mu=mu,
                         master_solver=master_solver,
                         master_verbose=master_verbose,
                         verbose=verbose)
        if not loss._loss_type == 'regressor':
            raise TypeError(f'{loss} is not an allowed SVR loss function')
        if not epsilon >= 0:
            raise ValueError('epsilon must be >= 0')
        self.epsilon = epsilon

    def _store_train_val_info(self, opt, X_batch, y_batch, X_val, y_val):
        super()._store_train_val_info(opt, X_batch, y_batch, X_val, y_val)
        if opt.is_batch_end():
            r2 = self.score(X_batch[:, :-1], y_batch)
            self.train_score_history.append(r2)
            if opt.is_verbose():
                print('\tr2: {: 1.4f}'.format(r2), end='')
            if self.validation_split:
                val_r2 = self.score(X_val[:, :-1], y_val)
                self.val_score_history.append(val_r2)
                if opt.is_verbose():
                    print('\tval_r2: {: 1.4f}'.format(val_r2), end='')
            self._update_no_improvement_count(opt)

    def fit(self, X, y):
        self.start_time = time.time()
        targets = y.shape[1] if y.ndim > 1 else 1
        if targets > 1:
            raise ValueError('use sklearn.multioutput.MultiOutputRegressor '
                             'to train a model over more than one target')

        if issubclass(self.optimizer, LineSearchOptimizer):

            if self.fit_intercept:
                X_biased = np.c_[X, np.ones_like(y)]
            else:
                X_biased = X

            self.loss = self.loss(self, X_biased, y, self.epsilon)
            self.optimizer = self.optimizer(f=self.loss,
                                            max_iter=self.max_iter,
                                            max_f_eval=self.max_f_eval,
                                            random_state=self.random_state,
                                            verbose=self.verbose).minimize()

            if self.optimizer.status == 'stopped':
                if self.optimizer.iter >= self.max_iter:
                    warnings.warn('max_iter reached but the optimization has not converged yet', ConvergenceWarning)
                elif self.optimizer.f_eval >= self.max_f_eval:
                    warnings.warn('max_f_eval reached but the optimization has not converged yet', ConvergenceWarning)

            self._unpack(self.optimizer.x)

        elif issubclass(self.optimizer, ProximalBundle):

            if self.fit_intercept:
                X_biased = np.c_[X, np.ones_like(y)]
            else:
                X_biased = X

            self.loss = self.loss(self, X_biased, y, self.epsilon)
            self.optimizer = self.optimizer(f=self.loss,
                                            mu=self.mu,
                                            max_iter=self.max_iter,
                                            master_solver=self.master_solver,
                                            master_verbose=self.master_verbose,
                                            random_state=self.random_state,
                                            verbose=self.verbose).minimize()

            if self.optimizer.status == 'error':
                warnings.warn('failure while computing direction for the master problem', ConvergenceWarning)

            self._unpack(self.optimizer.x)

        elif issubclass(self.optimizer, StochasticOptimizer):

            if self.validation_split:
                X, X_val, y, y_val = train_test_split(X, y,
                                                      test_size=self.validation_split,
                                                      random_state=self.random_state)

                if self.fit_intercept:
                    X_val_biased = np.c_[X_val, np.ones_like(y_val)]
                else:
                    X_val_biased = X_val

            else:
                X_val_biased = None
                y_val = None

            if self.fit_intercept:
                X_biased = np.c_[X, np.ones_like(y)]
            else:
                X_biased = X

            self.loss = self.loss(self, X_biased, y, self.epsilon)

            if issubclass(self.optimizer, StochasticMomentumOptimizer):

                self.optimizer = self.optimizer(f=self.loss,
                                                epochs=self.max_iter,
                                                step_size=self.learning_rate,
                                                momentum_type=self.momentum_type,
                                                momentum=self.momentum,
                                                batch_size=self.batch_size,
                                                callback=self._store_train_val_info,
                                                callback_args=(X_val_biased, y_val),
                                                shuffle=self.shuffle,
                                                random_state=self.random_state,
                                                verbose=self.verbose).minimize()

            else:

                self.optimizer = self.optimizer(f=self.loss,
                                                epochs=self.max_iter,
                                                step_size=self.learning_rate,
                                                batch_size=self.batch_size,
                                                callback=self._store_train_val_info,
                                                callback_args=(X_val_biased, y_val),
                                                shuffle=self.shuffle,
                                                random_state=self.random_state,
                                                verbose=self.verbose).minimize()

        else:

            raise TypeError(f'{self.optimizer} is not an allowed optimizer')

        if self.fit_intercept and X.shape[1] > 1:
            self.loss.X = X

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_


class DualSVR(RegressorMixin, DualSVM):

    def __init__(self,
                 loss=epsilon_insensitive,
                 epsilon=0.1,
                 kernel=gaussian,
                 C=1.,
                 tol=1e-3,
                 optimizer=SMO,
                 max_iter=1000,
                 learning_rate=0.1,
                 momentum_type='none',
                 momentum=0.9,
                 max_f_eval=15000,
                 fit_intercept=True,
                 mu=1,
                 master_solver='ecos',
                 master_verbose=False,
                 lagrangian_solver='minres',
                 lagrangian_solver_verbose=False,
                 verbose=False):
        super().__init__(loss=loss,
                         kernel=kernel,
                         C=C,
                         tol=tol,
                         optimizer=optimizer,
                         max_iter=max_iter,
                         learning_rate=learning_rate,
                         momentum_type=momentum_type,
                         momentum=momentum,
                         max_f_eval=max_f_eval,
                         fit_intercept=fit_intercept,
                         mu=mu,
                         master_solver=master_solver,
                         master_verbose=master_verbose,
                         lagrangian_solver=lagrangian_solver,
                         lagrangian_solver_verbose=lagrangian_solver_verbose,
                         verbose=verbose)
        if not loss._loss_type == 'regressor':
            raise TypeError(f'{loss} is not an allowed SVR loss function')
        if not epsilon >= 0:
            raise ValueError('epsilon must be >= 0')
        self.epsilon = epsilon

    def fit(self, X, y):
        self.start_time = time.time()
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

        if self.loss == EpsilonInsensitive:

            ub = np.ones(2 * n_samples) * self.C  # upper bounds

            if self.optimizer == 'smo' or self.optimizer == SMO:

                self.obj = Quadratic(Q, q)

                self.optimizer = SMORegression(self.obj, X, y, K, self.kernel, self.C,
                                               self.epsilon, self.tol, self.verbose).minimize()
                alphas_p, alphas_n = self.optimizer.alphas_p, self.optimizer.alphas_n
                if isinstance(self.kernel, LinearKernel):
                    self.coef_ = self.optimizer.w
                self.intercept_ = self.optimizer.b

            else:

                e = np.hstack((np.ones(n_samples), -np.ones(n_samples)))  # equality matrix

                if isinstance(self.optimizer, str):

                    lb = np.zeros(2 * n_samples)  # lower bounds

                    if not self.fit_intercept:

                        self.obj = Quadratic(Q, q)

                        out = StringIO()
                        with pipes(stdout=out, stderr=STDOUT):
                            self.alphas_ = solve_qp(P=Q,
                                                    q=q,
                                                    A=e,
                                                    b=np.zeros(1),
                                                    lb=lb,
                                                    ub=ub,
                                                    solver=self.optimizer,
                                                    verbose=True)

                    else:

                        Q += np.outer(e, e)
                        self.obj = Quadratic(Q, q)

                        out = StringIO()
                        with pipes(stdout=out, stderr=STDOUT):
                            self.alphas_ = solve_qp(P=Q,
                                                    q=q,
                                                    lb=lb,
                                                    ub=ub,
                                                    solver=self.optimizer,
                                                    verbose=True)

                    stdout = out.getvalue()
                    if stdout:
                        self.iter = int(max(re.findall(r'(\d+):', stdout)))
                        if self.verbose:
                            print(stdout)

                else:

                    if issubclass(self.optimizer, BoxConstrainedQuadraticOptimizer):

                        Q += np.outer(e, e)
                        self.obj = Quadratic(Q, q)

                        self.optimizer = self.optimizer(quad=self.obj,
                                                        ub=ub,
                                                        max_iter=self.max_iter,
                                                        verbose=self.verbose).minimize()

                    elif issubclass(self.optimizer, Optimizer):

                        if not self.fit_intercept:

                            self.obj = LagrangianEqualityBoxConstrainedQuadratic(primal=Quadratic(Q, q),
                                                                                 A=e,
                                                                                 ub=ub,
                                                                                 lagrangian_solver=self.lagrangian_solver,
                                                                                 lagrangian_solver_verbose=self.lagrangian_solver_verbose)

                        else:

                            Q += np.outer(e, e)
                            self.obj = LagrangianBoxConstrainedQuadratic(primal=Quadratic(Q, q),
                                                                         ub=ub,
                                                                         lagrangian_solver=self.lagrangian_solver,
                                                                         lagrangian_solver_verbose=self.lagrangian_solver_verbose)

                        if issubclass(self.optimizer, LineSearchOptimizer):

                            self.optimizer = self.optimizer(f=self.obj,
                                                            max_iter=self.max_iter,
                                                            max_f_eval=self.max_f_eval,
                                                            verbose=self.verbose).minimize()

                            if self.optimizer.status == 'stopped':
                                if self.optimizer.iter >= self.max_iter:
                                    warnings.warn('max_iter reached but the optimization has not converged yet',
                                                  ConvergenceWarning)
                                elif self.optimizer.f_eval >= self.max_f_eval:
                                    warnings.warn('max_f_eval reached but the optimization has not converged yet',
                                                  ConvergenceWarning)

                        elif issubclass(self.optimizer, ProximalBundle):

                            self.optimizer = self.optimizer(f=self.obj,
                                                            mu=self.mu,
                                                            max_iter=self.max_iter,
                                                            master_solver=self.master_solver,
                                                            master_verbose=self.master_verbose,
                                                            verbose=self.verbose).minimize()

                            if self.optimizer.status == 'error':
                                warnings.warn('failure while computing direction for the master problem',
                                              ConvergenceWarning)

                        elif issubclass(self.optimizer, StochasticOptimizer):

                            if issubclass(self.optimizer, StochasticMomentumOptimizer):

                                self.optimizer = self.optimizer(f=self.obj,
                                                                step_size=self.learning_rate,
                                                                epochs=self.max_iter,
                                                                momentum_type=self.momentum_type,
                                                                momentum=self.momentum,
                                                                callback=self._store_train_info,
                                                                verbose=self.verbose).minimize()

                            else:

                                self.optimizer = self.optimizer(f=self.obj,
                                                                step_size=self.learning_rate,
                                                                epochs=self.max_iter,
                                                                callback=self._store_train_info,
                                                                verbose=self.verbose).minimize()

                        else:

                            raise TypeError(f'{self.optimizer} is not an allowed optimizer')

                    self.alphas_ = self.optimizer.primal_x if self.optimizer.is_lagrangian_dual() else self.optimizer.x

                alphas_p, alphas_n = np.split(self.alphas_, 2)

        elif self.loss == SquaredEpsilonInsensitive:

            D = np.diag(np.ones(2 * n_samples) / (2 * self.C))
            Q += D

            e = np.hstack((np.ones(n_samples), -np.ones(n_samples)))  # equality matrix

            if isinstance(self.optimizer, str):

                lb = np.zeros(2 * n_samples)  # lower bounds

                if not self.fit_intercept:

                    self.obj = Quadratic(Q, q)

                    out = StringIO()
                    with pipes(stdout=out, stderr=STDOUT):
                        self.alphas_ = solve_qp(P=Q,
                                                q=q,
                                                A=e,
                                                b=np.zeros(1),
                                                lb=lb,
                                                solver=self.optimizer,
                                                verbose=True)

                else:

                    Q += np.outer(e, e)
                    self.obj = Quadratic(Q, q)

                    out = StringIO()
                    with pipes(stdout=out, stderr=STDOUT):
                        self.alphas_ = solve_qp(P=Q,
                                                q=q,
                                                lb=lb,
                                                solver=self.optimizer,
                                                verbose=True)

                stdout = out.getvalue()
                if stdout:
                    self.iter = int(max(re.findall(r'(\d+):', stdout)))
                    if self.verbose:
                        print(stdout)

            else:

                if issubclass(self.optimizer, Optimizer):

                    if not self.fit_intercept:

                        self.obj = LagrangianEqualityConstrainedQuadratic(primal=Quadratic(Q, q),
                                                                          A=e,
                                                                          lagrangian_solver=self.lagrangian_solver,
                                                                          lagrangian_solver_verbose=self.lagrangian_solver_verbose)

                    else:

                        Q += np.outer(e, e)
                        self.obj = LagrangianQuadratic(primal=Quadratic(Q, q),
                                                       lagrangian_solver=self.lagrangian_solver,
                                                       lagrangian_solver_verbose=self.lagrangian_solver_verbose)

                    if issubclass(self.optimizer, LineSearchOptimizer):

                        self.optimizer = self.optimizer(f=self.obj,
                                                        max_iter=self.max_iter,
                                                        max_f_eval=self.max_f_eval,
                                                        verbose=self.verbose).minimize()

                        if self.optimizer.status == 'stopped':
                            if self.optimizer.iter >= self.max_iter:
                                warnings.warn('max_iter reached but the optimization has not converged yet',
                                              ConvergenceWarning)
                            elif self.optimizer.f_eval >= self.max_f_eval:
                                warnings.warn('max_f_eval reached but the optimization has not converged yet',
                                              ConvergenceWarning)

                    elif issubclass(self.optimizer, ProximalBundle):

                        self.optimizer = self.optimizer(f=self.obj,
                                                        mu=self.mu,
                                                        max_iter=self.max_iter,
                                                        master_solver=self.master_solver,
                                                        master_verbose=self.master_verbose,
                                                        verbose=self.verbose).minimize()

                        if self.optimizer.status == 'error':
                            warnings.warn('failure while computing direction for the master problem',
                                          ConvergenceWarning)

                    elif issubclass(self.optimizer, StochasticOptimizer):

                        if issubclass(self.optimizer, StochasticMomentumOptimizer):

                            self.optimizer = self.optimizer(f=self.obj,
                                                            step_size=self.learning_rate,
                                                            epochs=self.max_iter,
                                                            momentum_type=self.momentum_type,
                                                            momentum=self.momentum,
                                                            callback=self._store_train_info,
                                                            verbose=self.verbose).minimize()

                        else:

                            self.optimizer = self.optimizer(f=self.obj,
                                                            step_size=self.learning_rate,
                                                            epochs=self.max_iter,
                                                            callback=self._store_train_info,
                                                            verbose=self.verbose).minimize()

                    else:

                        raise TypeError(f'{self.optimizer} is not an allowed optimizer')

                self.alphas_ = self.optimizer.primal_x if self.optimizer.is_lagrangian_dual() else self.optimizer.x

            alphas_p, alphas_n = np.split(self.alphas_, 2)

        else:

            raise TypeError(f'{self.loss} is not an allowed loss')

        sv = np.logical_or(alphas_p > 1e-6, alphas_n > 1e-6)
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
