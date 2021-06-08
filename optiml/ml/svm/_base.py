import re
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
from ...opti.constrained import BoxConstrainedQuadraticOptimizer, LagrangianQuadratic
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
        inversely proportional to C. Must be strictly positive.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    optimizer : LineSearchOptimizer or StochasticOptimizer subclass, default=None
        The solver for optimization. It can be a subclass of the `LineSearchOptimizer`
        which can converge faster and perform better for small datasets, e.g., the
        `BFGS` quasi-Newton method or, alternatively, a subclass of the `StochasticOptimizer`
        e.g., the `StochasticGradientDescent` or `Adam`, which works well on relatively
        large datasets (with thousands of training samples or more) in terms of both
        training time and validation score.

    max_iter : int, default=1000
        Maximum number of iterations. The solver iterates until convergence
        (determined by `tol`) or this number of iterations. If the optimizer
        is a subclass of `StochasticOptimizer`, this value determines the number
        of epochs (how many times each data point will be used), not the number
        of gradient steps.

    learning_rate : double, default=0.1
        The initial learning rate used for weight update. It controls the
        step-size in updating the weights. Only used when `optimizer` is a
        subclass of `StochasticOptimizer`.

    momentum_type : {'none', 'polyak', 'nesterov'}, default='none'
        Momentum type used for weight update. Only used when `optimizer` is
        a subclass of `StochasticOptimizer`.

    momentum : float, default=0.9
        Momentum for weight update. Should be between 0 and 1. Only used when
        `optimizer` is a subclass of `StochasticOptimizer`.

    max_f_eval : int, default=15000
        Only used when `optimizer` is a subclass of `LineSearchOptimizer`.
        Maximum number of loss function calls. The solver iterates until
        convergence (determined by `tol`), number of iterations reaches
        `max_iter`, or this number of loss function calls. Note that number
        of loss function calls will be greater than or equal to the number
        of iterations.

    mu : float, default=1.
        Mu parameter for the proximal bundle method.
        Only used when `optimizer` is `ProximalBundle`.

    master_solver : string, default='ecos'
        Master solver for the proximal bundle method for the `cvxpy` interface.
        Only used when `optimizer` is `ProximalBundle`.

    master_verbose : bool or int, default=False
        Controls the verbosity of the `cvxpy` interface.
        Only used when `optimizer` is `ProximalBundle`.

    verbose : bool or int, default=False
        Controls the verbosity of progress messages to stdout. Use a boolean value
        to switch on/off or an int value to show progress each `verbose` time
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
        self.support_ = np.zeros(0)
        self.support_vectors_ = np.zeros(0)
        if not isinstance(optimizer, str):
            self.train_loss_history = []

    def fit(self, X, y):
        raise NotImplementedError

    def _store_train_info(self, opt):
        if opt.is_lagrangian_dual():
            self.train_loss_history.append(opt.primal_f_x)
        else:
            self.train_loss_history.append(opt.f_x)


class PrimalSVM(SVM, ABC):
    """
    Base abstract class for the primal formulation of all (linear) SVM-type estimator.

    To be preferred when n_samples > n_features and the instance
    vector is linearly separable in the given space or, if not, consider
    the possibly to apply a non-linear transformation of the instance vector
    using a low-rank kernel matrix approximation, i.e., Nystrom, before training.
    See more at:
    - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation
    - https://cdn.rawgit.com/mstrazar/mklaren/master/docs/build/html/projection.html
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
                 intercept_scaling=1.,
                 shuffle=True,
                 random_state=None,
                 mu=1,
                 master_solver='ecos',
                 master_verbose=False,
                 verbose=False):
        super(PrimalSVM, self).__init__(loss=loss,
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
        self.intercept_scaling = intercept_scaling
        self.shuffle = shuffle
        self.random_state = random_state
        self.coef_ = np.zeros(0)
        if issubclass(self.optimizer, StochasticOptimizer):
            self.train_loss_history = []
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
            if opt.is_verbose() and opt.epoch != opt.iter:
                print('\tavg_loss: {: 1.4e}'.format(self._avg_epoch_loss), end='')
            self._avg_epoch_loss = 0.
            if self.validation_split:
                val_loss = self.loss(opt.x, X_val, y_val)
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
    Base abstract class for the dual formulation of all SVM-type estimator.

    To be preferred when n_samples < n_features. The training time
    complexity is more than quadratic with the number of samples which
    makes it hard to scale to large datasets. In the latter case consider
    using `PrimalSVM`, possibly after a non-linear transformation
    of the instance vector (if this should not be in the given space) using a
    low-rank kernel matrix approximation, i.e., Nystrom, before training.

    Parameters
    ----------

    kernel : `Kernel` instance like {linear, poly, gaussian, sigmoid}, default=gaussian
        Specifies the kernel type to be used in the algorithm.
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
                 reg_intercept=True,
                 rho=1,
                 mu=1,
                 master_solver='ecos',
                 master_verbose=False,
                 verbose=False):
        super(DualSVM, self).__init__(loss=loss,
                                      C=C,
                                      tol=tol,
                                      optimizer=optimizer,
                                      max_iter=max_iter,
                                      learning_rate=learning_rate,
                                      momentum_type=momentum_type,
                                      momentum=momentum,
                                      max_f_eval=max_f_eval,
                                      fit_intercept=True,
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
        self.reg_intercept = reg_intercept
        self.rho = rho
        if isinstance(self.kernel, LinearKernel):
            self.coef_ = np.zeros(0)
        self.alphas_ = np.zeros(0)
        self.dual_coef_ = np.zeros(0)


class PrimalSVC(LinearClassifierMixin, SparseCoefMixin, PrimalSVM):
    """
    Primal formulation of the (linear) C-Support Vector Classification.

    To be preferred when n_samples > n_features and the instance
    vector is linearly separable in the given space or, if not, consider
    the possibly to apply a non-linear transformation of the instance vector
    using a low-rank kernel matrix approximation, i.e., Nystrom, before training.
    See more at:
    - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation
    - https://cdn.rawgit.com/mstrazar/mklaren/master/docs/build/html/projection.html

    Parameters
    ----------

    loss : `SVMLoss` instance like {hinge, squared_hinge}, default='squared_hinge'
        Specifies the loss function. The epsilon-insensitive loss is the
        L1 loss, while the squared epsilon-insensitive loss is the L2 loss.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.

    tol : float, default=1e-4
        Tolerance for stopping criterion.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e., data is expected to be already centered).

    intercept_scaling : float, default=1.
        When `fit_intercept` is True, instance vector x becomes
        [x, intercept_scaling], i.e., a "synthetic" feature with constant
        value equals to `intercept_scaling` is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note: the synthetic feature weight is subject to L1/L2 regularization
        as all other features. To lessen the effect of regularization on synthetic
        feature weight (and therefore on the intercept) `intercept_scaling` has
        to be increased.

    max_iter : int, default=1000
        The maximum number of iterations to be run.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for train-test split if
        `early_stopping` is True and shuffling the data for batch sampling when
        an instance of `StochasticOptimizer` class is used as `optimizer` value.
        Pass an int for reproducible output across multiple function calls.

    shuffle : bool, default=True
        Whether to shuffle samples for batch sampling in each iteration. Only
        used when the `optimizer` is a subclass of `StochasticOptimizer`.

    verbose : bool or int, default=False
        Enable verbose output.

    Attributes
    ----------

    coef_ : ndarray of shape (n_features,)
        Weights assigned to the features (coefficients in the primal problem).

    intercept_ : float
        Constants in decision function.

    support_ : ndarray of shape (n_SV,)
        Indices of support vectors.

    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors.
    """

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
                 intercept_scaling=1.,
                 shuffle=True,
                 random_state=None,
                 mu=1,
                 master_solver='ecos',
                 master_verbose=False,
                 verbose=False):
        super(PrimalSVC, self).__init__(loss=loss,
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
                                        intercept_scaling=intercept_scaling,
                                        shuffle=shuffle,
                                        random_state=random_state,
                                        mu=mu,
                                        master_solver=master_solver,
                                        master_verbose=master_verbose,
                                        verbose=verbose)
        if not loss._loss_type == 'classifier':
            raise TypeError(f'{loss} is not an allowed SVC loss function')
        self.lb = LabelBinarizer(neg_label=-1)
        self.dual = DualSVC

    def _store_train_val_info(self, opt, X_batch, y_batch, X_val, y_val):
        super(PrimalSVC, self)._store_train_val_info(opt, X_batch, y_batch, X_val, y_val)
        if opt.is_batch_end():
            acc = self.score(X_batch[:, :-1] if self.fit_intercept else X_batch, y_batch)
            self.train_score_history.append(acc)
            if opt.is_verbose():
                print('\tacc: {:1.4f}'.format(acc), end='')
            if self.validation_split:
                val_acc = self.score(X_val[:, :-1] if self.fit_intercept else X_val, y_val)
                self.val_score_history.append(val_acc)
                if opt.is_verbose():
                    print('\tval_acc: {:1.4f}'.format(val_acc), end='')
            self._update_no_improvement_count(opt)

    def fit(self, X, y):
        self.lb.fit(y)
        if len(self.lb.classes_) > 2:
            raise ValueError('use OneVsOneClassifier or OneVsRestClassifier from sklearn.multiclass '
                             'to train a model over more than two labels')
        y = self.lb.transform(y).ravel()

        if issubclass(self.optimizer, LineSearchOptimizer):

            if self.fit_intercept:
                X_biased = np.c_[X, np.full_like(y, self.intercept_scaling)]
            else:
                X_biased = X

            self.loss = self.loss(self, X_biased, y)
            self.optimizer = self.optimizer(f=self.loss,
                                            max_iter=self.max_iter,
                                            max_f_eval=self.max_f_eval,
                                            random_state=self.random_state,
                                            callback=self._store_train_info,
                                            verbose=self.verbose).minimize()

            if self.optimizer.status == 'stopped':
                if self.optimizer.iter >= self.max_iter:
                    warnings.warn('max_iter reached but the optimization has not converged yet', ConvergenceWarning)
                elif self.optimizer.f_eval >= self.max_f_eval:
                    warnings.warn('max_f_eval reached but the optimization has not converged yet', ConvergenceWarning)

            self._unpack(self.optimizer.x)

        elif issubclass(self.optimizer, ProximalBundle):

            if self.fit_intercept:
                X_biased = np.c_[X, np.full_like(y, self.intercept_scaling)]
            else:
                X_biased = X

            self.loss = self.loss(self, X_biased, y)
            self.optimizer = self.optimizer(f=self.loss,
                                            mu=self.mu,
                                            max_iter=self.max_iter,
                                            master_solver=self.master_solver,
                                            master_verbose=self.master_verbose,
                                            random_state=self.random_state,
                                            callback=self._store_train_info,
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
                    X_val_biased = np.c_[X_val, np.full_like(y_val, self.intercept_scaling)]
                else:
                    X_val_biased = X_val

            else:
                X_val_biased = None
                y_val = None

            if self.fit_intercept:
                X_biased = np.c_[X, np.full_like(y, self.intercept_scaling)]
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

        self.support_ = np.argwhere(np.abs(self.decision_function(X)) <= 1).ravel()
        self.support_vectors_ = X[self.support_]

        return self

    def decision_function(self, X):
        return np.dot(X, self.coef_) + self.intercept_

    def predict(self, X):
        return self.lb.inverse_transform(self.decision_function(X))


class DualSVC(ClassifierMixin, DualSVM):
    """
    Dual formulation of the C-Support Vector Classification.

    To be preferred when n_samples < n_features. The training time
    complexity is more than quadratic with the number of samples which
    makes it hard to scale to large datasets. In the latter case consider
    using `PrimalSVR`, possibly after a non-linear transformation
    of the instance vector (if this should not be in the given space) using a
    low-rank kernel matrix approximation, i.e., Nystrom, before training.

    Parameters
    ----------

    loss : `SVMLoss` instance like {hinge, squared_hinge}, default='hinge'
        Specifies the loss function. The hinge loss is the L1 loss,
        while the squared hinge loss is the L2 loss.

    kernel : `Kernel` instance like {linear, poly, gaussian, sigmoid}, default=gaussian
         Specifies the kernel type to be used in the algorithm.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    max_iter : int, default=1000
        The maximum number of iterations to be run.

    verbose : bool or int, default=False
        Enable verbose output.

    Attributes
    ----------

    coef_ : ndarray of shape (n_features,)
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

    dual_coef_ : ndarray of shape (n_SV,)
        Coefficients of the support vector in the decision function.

    intercept_ : float
        Constants in decision function.

    support_ : ndarray of shape (n_SV,)
        Indices of support vectors.

    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors.
    """

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
                 reg_intercept=True,
                 rho=1,
                 mu=1,
                 master_solver='ecos',
                 master_verbose=False,
                 verbose=False):
        super(DualSVC, self).__init__(loss=loss,
                                      kernel=kernel,
                                      C=C,
                                      tol=tol,
                                      optimizer=optimizer,
                                      max_iter=max_iter,
                                      learning_rate=learning_rate,
                                      momentum_type=momentum_type,
                                      momentum=momentum,
                                      max_f_eval=max_f_eval,
                                      reg_intercept=reg_intercept,
                                      rho=rho,
                                      mu=mu,
                                      master_solver=master_solver,
                                      master_verbose=master_verbose,
                                      verbose=verbose)
        if not loss._loss_type == 'classifier':
            raise TypeError(f'{loss} is not an allowed SVC loss function')
        self.lb = LabelBinarizer(neg_label=-1)

    def fit(self, X, y):
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

                if not self.reg_intercept:

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
                                                verbose=False if self.verbose < 0 else True)

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
                                                verbose=False if self.verbose < 0 else True)

                stdout = out.getvalue()
                if stdout:
                    self.iter = int(max(re.findall(r'(\d+):', stdout)))
                    if self.verbose:
                        print(stdout)

            else:

                if issubclass(self.optimizer, BoxConstrainedQuadraticOptimizer):

                    if not self.reg_intercept:

                        # TODO constrained optimizer with A x = 0 and 0 <= x <= ub is not available, so relax
                        #  the equality constraint and solve the lagrangian with the bcqp optimizer
                        self.obj = LagrangianQuadratic(primal=Quadratic(Q, q),
                                                       A=y,
                                                       b=np.zeros(1),
                                                       rho=self.rho)

                    else:

                        Q += np.outer(y, y)
                        self.obj = Quadratic(Q, q)

                    self.optimizer = self.optimizer(quad=self.obj,
                                                    ub=ub,
                                                    max_iter=self.max_iter,
                                                    callback=self._store_train_info,
                                                    verbose=self.verbose).minimize()

                elif issubclass(self.optimizer, Optimizer):

                    lb = np.zeros(n_samples)  # lower bounds

                    if not self.reg_intercept:

                        self.obj = LagrangianQuadratic(primal=Quadratic(Q, q),
                                                       A=y,
                                                       b=np.zeros(1),
                                                       lb=lb,
                                                       ub=ub,
                                                       rho=self.rho)

                    else:

                        Q += np.outer(y, y)
                        self.obj = LagrangianQuadratic(primal=Quadratic(Q, q),
                                                       lb=lb,
                                                       ub=ub,
                                                       rho=self.rho)

                    if issubclass(self.optimizer, LineSearchOptimizer):

                        self.optimizer = self.optimizer(f=self.obj,
                                                        max_iter=self.max_iter,
                                                        max_f_eval=self.max_f_eval,
                                                        callback=self._store_train_info,
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
                                                        callback=self._store_train_info,
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

                        if self.optimizer.status == 'stopped':
                            warnings.warn('max_iter reached but the optimization has not converged yet',
                                          ConvergenceWarning)

                    else:

                        raise TypeError(f'{self.optimizer} is not an allowed optimizer')

                self.alphas_ = self.optimizer.x

        elif self.loss == SquaredHinge:

            D = np.diag(np.ones(n_samples) / (2 * self.C))
            Q += D

            if isinstance(self.optimizer, str):

                lb = np.zeros(n_samples)  # lower bounds

                if not self.reg_intercept:

                    self.obj = Quadratic(Q, q)

                    out = StringIO()
                    with pipes(stdout=out, stderr=STDOUT):
                        self.alphas_ = solve_qp(P=Q,
                                                q=q,
                                                A=y.astype(float),
                                                b=np.zeros(1),
                                                lb=lb,
                                                solver=self.optimizer,
                                                verbose=False if self.verbose < 0 else True)

                else:

                    Q += np.outer(y, y)
                    self.obj = Quadratic(Q, q)

                    out = StringIO()
                    with pipes(stdout=out, stderr=STDOUT):
                        self.alphas_ = solve_qp(P=Q,
                                                q=q,
                                                lb=lb,
                                                solver=self.optimizer,
                                                verbose=False if self.verbose < 0 else True)

                stdout = out.getvalue()
                if stdout:
                    self.iter = int(max(re.findall(r'(\d+):', stdout)))
                    if self.verbose:
                        print(stdout)

            else:

                if issubclass(self.optimizer, Optimizer):

                    lb = np.zeros(n_samples)  # lower bounds

                    if not self.reg_intercept:

                        self.obj = LagrangianQuadratic(primal=Quadratic(Q, q),
                                                       A=y,
                                                       b=np.zeros(1),
                                                       lb=lb,
                                                       rho=self.rho)

                    else:

                        Q += np.outer(y, y)
                        self.obj = LagrangianQuadratic(primal=Quadratic(Q, q),
                                                       lb=lb,
                                                       rho=self.rho)

                    if issubclass(self.optimizer, LineSearchOptimizer):

                        self.optimizer = self.optimizer(f=self.obj,
                                                        max_iter=self.max_iter,
                                                        max_f_eval=self.max_f_eval,
                                                        callback=self._store_train_info,
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
                                                        callback=self._store_train_info,
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

                        if self.optimizer.status == 'stopped':
                            warnings.warn('max_iter reached but the optimization has not converged yet',
                                          ConvergenceWarning)

                    else:

                        raise TypeError(f'{self.optimizer} is not an allowed optimizer')

                self.alphas_ = self.optimizer.x

        else:

            raise TypeError(f'{self.loss} is not an allowed loss')

        sv = self.alphas_ > 1e-6
        self.support_ = np.arange(len(self.alphas_))[sv]
        self.support_vectors_, sv_y, alphas = X[sv], y[sv], self.alphas_[sv]
        self.dual_coef_ = alphas * sv_y

        if self.optimizer != SMOClassifier:

            if isinstance(self.kernel, LinearKernel):
                self.coef_ = np.dot(self.dual_coef_, self.support_vectors_)

            for n in range(len(alphas)):
                self.intercept_ += sv_y[n]
                self.intercept_ -= np.sum(self.dual_coef_ * K[self.support_[n], sv])
            self.intercept_ /= len(alphas)

        return self

    def decision_function(self, X):
        if not isinstance(self.kernel, LinearKernel):
            return np.dot(self.dual_coef_, self.kernel(self.support_vectors_, X)) + self.intercept_
        return np.dot(X, self.coef_) + self.intercept_

    def predict(self, X):
        return self.lb.inverse_transform(self.decision_function(X))


class PrimalSVR(RegressorMixin, LinearModel, PrimalSVM):
    """
    Primal formulation of the (linear) Epsilon-Support Vector Regression.

    To be preferred when n_samples > n_features and the instance
    vector is linearly separable in the given space or, if not, consider
    the possibly to apply a non-linear transformation of the instance vector
    using a low-rank kernel matrix approximation, i.e., Nystrom, before training.
    See more at:
    - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation
    - https://cdn.rawgit.com/mstrazar/mklaren/master/docs/build/html/projection.html

    Parameters
    ----------

    loss : `SVMLoss` instance like {epsilon_insensitive, squared_epsilon_insensitive}, \
                default='squared_epsilon_insensitive'
        Specifies the loss function. The epsilon-insensitive loss is the
        L1 loss, while the squared epsilon-insensitive loss is the L2 loss.

    epsilon : float, default=0.1
        Epsilon parameter in the (squared) epsilon-insensitive loss function.
        It specifies the epsilon-tube within which no penalty is associated
        in the training loss function with points predicted within a distance
        epsilon from the actual value.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.

    tol : float, default=1e-4
        Tolerance for stopping criterion.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e., data is expected to be already centered).

    intercept_scaling : float, default=1.
        When `fit_intercept` is True, instance vector x becomes
        [x, intercept_scaling], i.e., a "synthetic" feature with constant
        value equals to `intercept_scaling` is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note: the synthetic feature weight is subject to L1/L2 regularization
        as all other features. To lessen the effect of regularization on synthetic
        feature weight (and therefore on the intercept) `intercept_scaling` has
        to be increased.

    max_iter : int, default=1000
        The maximum number of iterations to be run.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for train-test split if
        `early_stopping` is True and shuffling the data for batch sampling when
        an instance of `StochasticOptimizer` class is used as `optimizer` value.
        Pass an int for reproducible output across multiple function calls.

    shuffle : bool, default=True
        Whether to shuffle samples for batch sampling in each iteration. Only
        used when the `optimizer` is a subclass of `StochasticOptimizer`.

    verbose : bool or int, default=False
        Enable verbose output.

    Attributes
    ----------

    coef_ : ndarray of shape (n_features,)
        Weights assigned to the features (coefficients in the primal problem).

    intercept_ : float
        Constants in decision function.

    support_ : ndarray of shape (n_SV,)
        Indices of support vectors.

    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors.
    """

    def __init__(self,
                 loss=squared_epsilon_insensitive,
                 epsilon=0.1,
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
                 intercept_scaling=1.,
                 shuffle=True,
                 random_state=None,
                 mu=1,
                 master_solver='ecos',
                 master_verbose=False,
                 verbose=False):
        super(PrimalSVR, self).__init__(loss=loss,
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
                                        intercept_scaling=intercept_scaling,
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
        self.dual = DualSVR

    def _store_train_val_info(self, opt, X_batch, y_batch, X_val, y_val):
        super(PrimalSVR, self)._store_train_val_info(opt, X_batch, y_batch, X_val, y_val)
        if opt.is_batch_end():
            r2 = self.score(X_batch[:, :-1] if self.fit_intercept else X_batch, y_batch)
            self.train_score_history.append(r2)
            if opt.is_verbose():
                print('\tr2: {: 1.4f}'.format(r2), end='')
            if self.validation_split:
                val_r2 = self.score(X_val[:, :-1] if self.fit_intercept else X_val, y_val)
                self.val_score_history.append(val_r2)
                if opt.is_verbose():
                    print('\tval_r2: {: 1.4f}'.format(val_r2), end='')
            self._update_no_improvement_count(opt)

    def fit(self, X, y):
        targets = y.shape[1] if y.ndim > 1 else 1
        if targets > 1:
            raise ValueError('use sklearn.multioutput.MultiOutputRegressor '
                             'to train a model over more than one target')

        if issubclass(self.optimizer, LineSearchOptimizer):

            if self.fit_intercept:
                X_biased = np.c_[X, np.full_like(y, self.intercept_scaling)]
            else:
                X_biased = X

            self.loss = self.loss(self, X_biased, y, self.epsilon)
            self.optimizer = self.optimizer(f=self.loss,
                                            max_iter=self.max_iter,
                                            max_f_eval=self.max_f_eval,
                                            random_state=self.random_state,
                                            callback=self._store_train_info,
                                            verbose=self.verbose).minimize()

            if self.optimizer.status == 'stopped':
                if self.optimizer.iter >= self.max_iter:
                    warnings.warn('max_iter reached but the optimization has not converged yet', ConvergenceWarning)
                elif self.optimizer.f_eval >= self.max_f_eval:
                    warnings.warn('max_f_eval reached but the optimization has not converged yet', ConvergenceWarning)

            self._unpack(self.optimizer.x)

        elif issubclass(self.optimizer, ProximalBundle):

            if self.fit_intercept:
                X_biased = np.c_[X, np.full_like(y, self.intercept_scaling)]
            else:
                X_biased = X

            self.loss = self.loss(self, X_biased, y, self.epsilon)
            self.optimizer = self.optimizer(f=self.loss,
                                            mu=self.mu,
                                            max_iter=self.max_iter,
                                            master_solver=self.master_solver,
                                            master_verbose=self.master_verbose,
                                            random_state=self.random_state,
                                            callback=self._store_train_info,
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
                    X_val_biased = np.c_[X_val, np.full_like(y_val, self.intercept_scaling)]
                else:
                    X_val_biased = X_val

            else:
                X_val_biased = None
                y_val = None

            if self.fit_intercept:
                X_biased = np.c_[X, np.full_like(y, self.intercept_scaling)]
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

        self.support_ = np.argwhere(np.abs(y - self.predict(X)) >= self.epsilon).ravel()
        self.support_vectors_ = X[self.support_]

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_


class DualSVR(RegressorMixin, DualSVM):
    """
    Dual formulation of the Epsilon-Support Vector Regression.

    To be preferred when n_samples < n_features. The training time
    complexity is more than quadratic with the number of samples which
    makes it hard to scale to large datasets. In the latter case consider
    using `PrimalSVR`, possibly after a non-linear transformation
    of the instance vector (if this should not be in the given space) using a
    low-rank kernel matrix approximation, i.e., Nystrom, before training.

    Parameters
    ----------

    loss : `SVMLoss` instance like {epsilon_insensitive, squared_epsilon_insensitive}, default='epsilon_insensitive'
        Specifies the loss function. The epsilon-insensitive loss
        is the L1 loss, while the squared epsilon-insensitive
        loss is the L2 loss.

    epsilon : float, default=0.1
         Epsilon parameter in the (squared) epsilon-insensitive loss function.
         It specifies the epsilon-tube within which no penalty is associated
         in the training loss function with points predicted within a distance
         epsilon from the actual value.

    kernel : `Kernel` instance like {linear, poly, gaussian, sigmoid}, default=gaussian
         Specifies the kernel type to be used in the algorithm.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    max_iter : int, default=1000
        The maximum number of iterations to be run.

    verbose : bool or int, default=False
        Enable verbose output.

    Attributes
    ----------

    coef_ : ndarray of shape (n_features,)
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

    dual_coef_ : ndarray of shape (n_SV,)
        Coefficients of the support vector in the decision function.

    intercept_ : float
        Constants in decision function.

    support_ : ndarray of shape (n_SV,)
        Indices of support vectors.

    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors.
    """

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
                 reg_intercept=True,
                 rho=1,
                 mu=1,
                 master_solver='ecos',
                 master_verbose=False,
                 verbose=False):
        super(DualSVR, self).__init__(loss=loss,
                                      kernel=kernel,
                                      C=C,
                                      tol=tol,
                                      optimizer=optimizer,
                                      max_iter=max_iter,
                                      learning_rate=learning_rate,
                                      momentum_type=momentum_type,
                                      momentum=momentum,
                                      max_f_eval=max_f_eval,
                                      reg_intercept=reg_intercept,
                                      rho=rho,
                                      mu=mu,
                                      master_solver=master_solver,
                                      master_verbose=master_verbose,
                                      verbose=verbose)
        if not loss._loss_type == 'regressor':
            raise TypeError(f'{loss} is not an allowed SVR loss function')
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

        if self.loss == EpsilonInsensitive:

            ub = np.ones(2 * n_samples) * self.C  # upper bounds

            if self.optimizer == 'smo' or self.optimizer == SMO:

                self.obj = Quadratic(Q, q)

                self.optimizer = SMORegression(self.obj, X, y, K, self.kernel, self.C,
                                               self.epsilon, self.tol, self.verbose).minimize()
                alphas_p, alphas_n = self.optimizer.alphas_p, self.optimizer.alphas_n
                self.alphas_ = np.concatenate((alphas_p, alphas_n))
                if isinstance(self.kernel, LinearKernel):
                    self.coef_ = self.optimizer.w
                self.intercept_ = self.optimizer.b

            else:

                e = np.hstack((np.ones(n_samples), -np.ones(n_samples)))  # equality matrix

                if isinstance(self.optimizer, str):

                    lb = np.zeros(2 * n_samples)  # lower bounds

                    if not self.reg_intercept:

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
                                                    verbose=False if self.verbose < 0 else True)

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
                                                    verbose=False if self.verbose < 0 else True)

                    stdout = out.getvalue()
                    if stdout:
                        self.iter = int(max(re.findall(r'(\d+):', stdout)))
                        if self.verbose:
                            print(stdout)

                else:

                    if issubclass(self.optimizer, BoxConstrainedQuadraticOptimizer):

                        if not self.reg_intercept:

                            # TODO constrained optimizer with A x = 0 and 0 <= x <= ub is not available, so relax
                            #  the equality constraint and solve the lagrangian with the bcqp optimizer
                            self.obj = LagrangianQuadratic(primal=Quadratic(Q, q),
                                                           A=e,
                                                           b=np.zeros(1),
                                                           rho=self.rho)

                        else:

                            Q += np.outer(e, e)
                            self.obj = Quadratic(Q, q)

                        self.optimizer = self.optimizer(quad=self.obj,
                                                        ub=ub,
                                                        max_iter=self.max_iter,
                                                        callback=self._store_train_info,
                                                        verbose=self.verbose).minimize()

                    elif issubclass(self.optimizer, Optimizer):

                        lb = np.zeros(2 * n_samples)  # lower bounds

                        if not self.reg_intercept:

                            self.obj = LagrangianQuadratic(primal=Quadratic(Q, q),
                                                           A=e,
                                                           b=np.zeros(1),
                                                           lb=lb,
                                                           ub=ub,
                                                           rho=self.rho)

                        else:

                            Q += np.outer(e, e)
                            self.obj = LagrangianQuadratic(primal=Quadratic(Q, q),
                                                           lb=lb,
                                                           ub=ub,
                                                           rho=self.rho)

                        if issubclass(self.optimizer, LineSearchOptimizer):

                            self.optimizer = self.optimizer(f=self.obj,
                                                            max_iter=self.max_iter,
                                                            max_f_eval=self.max_f_eval,
                                                            callback=self._store_train_info,
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
                                                            callback=self._store_train_info,
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

                            if self.optimizer.status == 'stopped':
                                warnings.warn('max_iter reached but the optimization has not converged yet',
                                              ConvergenceWarning)

                        else:

                            raise TypeError(f'{self.optimizer} is not an allowed optimizer')

                    self.alphas_ = self.optimizer.x

                alphas_p, alphas_n = np.split(self.alphas_, 2)

        elif self.loss == SquaredEpsilonInsensitive:

            D = np.diag(np.ones(2 * n_samples) / (2 * self.C))
            Q += D

            e = np.hstack((np.ones(n_samples), -np.ones(n_samples)))  # equality matrix

            if isinstance(self.optimizer, str):

                lb = np.zeros(2 * n_samples)  # lower bounds

                if not self.reg_intercept:

                    self.obj = Quadratic(Q, q)

                    out = StringIO()
                    with pipes(stdout=out, stderr=STDOUT):
                        self.alphas_ = solve_qp(P=Q,
                                                q=q,
                                                A=e,
                                                b=np.zeros(1),
                                                lb=lb,
                                                solver=self.optimizer,
                                                verbose=False if self.verbose < 0 else True)

                else:

                    Q += np.outer(e, e)
                    self.obj = Quadratic(Q, q)

                    out = StringIO()
                    with pipes(stdout=out, stderr=STDOUT):
                        self.alphas_ = solve_qp(P=Q,
                                                q=q,
                                                lb=lb,
                                                solver=self.optimizer,
                                                verbose=False if self.verbose < 0 else True)

                stdout = out.getvalue()
                if stdout:
                    self.iter = int(max(re.findall(r'(\d+):', stdout)))
                    if self.verbose:
                        print(stdout)

            else:

                if issubclass(self.optimizer, Optimizer):

                    lb = np.zeros(2 * n_samples)  # lower bounds

                    if not self.reg_intercept:

                        self.obj = LagrangianQuadratic(primal=Quadratic(Q, q),
                                                       A=e,
                                                       b=np.zeros(1),
                                                       lb=lb,
                                                       rho=self.rho)

                    else:

                        Q += np.outer(e, e)
                        self.obj = LagrangianQuadratic(primal=Quadratic(Q, q),
                                                       lb=lb,
                                                       rho=self.rho)

                    if issubclass(self.optimizer, LineSearchOptimizer):

                        self.optimizer = self.optimizer(f=self.obj,
                                                        max_iter=self.max_iter,
                                                        max_f_eval=self.max_f_eval,
                                                        callback=self._store_train_info,
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
                                                        callback=self._store_train_info,
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

                        if self.optimizer.status == 'stopped':
                            warnings.warn('max_iter reached but the optimization has not converged yet',
                                          ConvergenceWarning)

                    else:

                        raise TypeError(f'{self.optimizer} is not an allowed optimizer')

                self.alphas_ = self.optimizer.x

            alphas_p, alphas_n = np.split(self.alphas_, 2)

        else:

            raise TypeError(f'{self.loss} is not an allowed loss')

        sv = np.logical_or(alphas_p > 1e-6, alphas_n > 1e-6)
        self.support_ = np.arange(len(alphas_p))[sv]
        self.support_vectors_, sv_y, alphas_p, alphas_n = X[sv], y[sv], alphas_p[sv], alphas_n[sv]
        self.dual_coef_ = alphas_p - alphas_n

        if self.optimizer != SMORegression:

            if isinstance(self.kernel, LinearKernel):
                self.coef_ = np.dot(self.dual_coef_, self.support_vectors_)

            for n in range(len(alphas_p)):
                self.intercept_ += sv_y[n]
                self.intercept_ -= np.sum(self.dual_coef_ * K[self.support_[n], sv])
            self.intercept_ -= self.epsilon
            self.intercept_ /= len(alphas_p)

        return self

    def predict(self, X):
        if not isinstance(self.kernel, LinearKernel):
            return np.dot(self.dual_coef_, self.kernel(self.support_vectors_, X)) + self.intercept_
        return np.dot(X, self.coef_) + self.intercept_
