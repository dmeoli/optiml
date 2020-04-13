import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import qpsolvers
from matplotlib.lines import Line2D
from qpsolvers import solve_qp
from sklearn.base import ClassifierMixin, BaseEstimator, RegressorMixin

from optimization.optimization_function import BoxConstrainedQuadratic, LagrangianBoxConstrained, Quadratic
from optimization.optimizer import BoxConstrainedOptimizer, Optimizer
from utils import scipy_solve_qp, scipy_solve_svm

plt.style.use('ggplot')


class SVM(BaseEstimator):
    def __init__(self, kernel='rbf', degree=3., gamma='scale', coef0=0., C=1.,
                 tol=1e-3, optimizer='SMO', epochs=1000, verbose=False):
        self.kernels = {'linear': self.linear,
                        'poly': self.poly,
                        'rbf': self.rbf,
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
        if (optimizer not in (solve_qp, scipy_solve_qp, scipy_solve_svm, 'SMO')
                and not issubclass(optimizer, Optimizer)):
            raise TypeError('optimizer is not an allowed optimizer')
        self.optimizer = optimizer
        if not np.isscalar(epochs):
            raise ValueError('epochs is not an integer scalar')
        if not epochs > 0:
            raise ValueError('epochs must be > 0')
        self.epochs = epochs
        if not isinstance(verbose, bool) or verbose not in (0, 1):
            raise ValueError('verbose is not a boolean value')
        self.verbose = verbose
        self.intercept_ = 0.

    # kernels

    def linear(self, X, y):
        return np.dot(X, y.T)

    def poly(self, X, y):
        gamma = (1. / (X.shape[1] * X.var()) if self.gamma is 'scale' else  # auto
                 1. / X.shape[1] if isinstance(self.gamma, str) else self.gamma)
        return (gamma * np.dot(X, y.T) + self.coef0) ** self.degree

    def rbf(self, X, y):
        gamma = (1. / (X.shape[1] * X.var()) if self.gamma is 'scale' else  # auto
                 1. / X.shape[1] if isinstance(self.gamma, str) else self.gamma)
        return np.exp(-gamma * np.linalg.norm(X[:, np.newaxis] - y[np.newaxis, :], axis=2) ** 2)

    def sigmoid(self, X, y):
        gamma = (1. / (X.shape[1] * X.var()) if self.gamma is 'scale' else  # auto
                 1. / X.shape[1] if isinstance(self.gamma, str) else self.gamma)
        return np.tanh(gamma * np.dot(X, y.T) + self.coef0)

    # Platt's SMO algorithm
    class SMO:
        def __init__(self, f, X, y, K, kernel='rbf', C=1., tol=1e-3, verbose=False):
            self.f = f
            self.X = X
            self.y = y
            self.K = K
            self.kernel = kernel
            if kernel is 'linear':
                self.w = np.zeros(2)
            self.b = 0.
            self.C = C
            # error cache to speed up computation
            # is initialized to SVM output - y
            self.errors = self._svm_output() - y
            self.tol = tol
            self.verbose = verbose

        def _svm_output(self):
            raise NotImplementedError

        def _take_step(self, i1, i2):
            raise NotImplementedError

        def _examine_example(self, i2):
            raise NotImplementedError

        def smo(self):
            raise NotImplementedError

    @staticmethod
    def plot(svm, X, y):
        ax = plt.axes()

        # axis labels and limits
        if isinstance(svm, ClassifierMixin):
            labels = np.unique(y)
            X1, X2 = X[y == labels[0]], X[y == labels[1]]
            plt.xlabel('$x_1$', fontsize=9)
            plt.ylabel('$x_2$', fontsize=9)
            ax.set(xlim=(X1.min(), X1.max()), ylim=(X2.min(), X2.max()))
        elif isinstance(svm, RegressorMixin):
            plt.xlabel('$X$', fontsize=9)
            plt.ylabel('$y$', fontsize=9)

        plt.title(f'{"custom" if isinstance(svm, SVM) else "sklearn"} '
                  f'{type(svm).__name__} using {svm.kernel} kernel', fontsize=9)

        # set the legend
        if isinstance(svm, ClassifierMixin):
            plt.legend([Line2D([0], [0], linestyle='none', marker='x', color='lightblue',
                               markerfacecolor='lightblue', markersize=9),
                        Line2D([0], [0], linestyle='none', marker='o', color='darkorange',
                               markerfacecolor='darkorange', markersize=9),
                        Line2D([0], [0], linestyle='-', marker='.', color='black',
                               markerfacecolor='darkorange', markersize=0),
                        Line2D([0], [0], linestyle='--', marker='.', color='black',
                               markerfacecolor='darkorange', markersize=0),
                        Line2D([0], [0], linestyle='none', marker='.', color='blue',
                               markerfacecolor='blue', markersize=9)],
                       ['negative -1', 'positive +1', 'decision boundary', 'margin', 'support vectors'],
                       fontsize='7', shadow=True).get_frame().set_facecolor('white')
        elif isinstance(svm, RegressorMixin):
            plt.legend([Line2D([0], [0], linestyle='none', marker='o', color='darkorange',
                               markerfacecolor='darkorange', markersize=9),
                        Line2D([0], [0], linestyle='-', marker='.', color='black',
                               markerfacecolor='darkorange', markersize=0),
                        Line2D([0], [0], linestyle='--', marker='.', color='black',
                               markerfacecolor='darkorange', markersize=0),
                        Line2D([0], [0], linestyle='none', marker='.', color='blue',
                               markerfacecolor='blue', markersize=9)],
                       ['training data', 'decision boundary', '$\epsilon$-insensitive tube', 'support vectors'],
                       fontsize='7', shadow=True).get_frame().set_facecolor('white')

        # training data
        if isinstance(svm, ClassifierMixin):
            plt.plot(X1[:, 0], X1[:, 1], marker='x', markersize=5, color='lightblue', linestyle='none')
            plt.plot(X2[:, 0], X2[:, 1], marker='o', markersize=4, color='darkorange', linestyle='none')
        else:
            plt.plot(X, y, marker='o', markersize=4, color='darkorange', linestyle='none')

        # support vectors
        if isinstance(svm, ClassifierMixin):
            plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=60, color='blue')
        elif isinstance(svm, RegressorMixin):
            plt.scatter(X[svm.support_], y[svm.support_], s=60, color='blue')

        if isinstance(svm, ClassifierMixin):
            _X1, _X2 = np.meshgrid(np.linspace(X1.min(), X1.max(), 50), np.linspace(X1.min(), X1.max(), 50))
            X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(_X1), np.ravel(_X2))])
            Z = svm.decision_function(X).reshape(_X1.shape)
            plt.contour(_X1, _X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
            plt.contour(_X1, _X2, Z + 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
            plt.contour(_X1, _X2, Z - 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
        elif isinstance(svm, RegressorMixin):
            _X = np.linspace(-2 * np.pi, 2 * np.pi, 10000).reshape((-1, 1))
            Z = svm.predict(_X)
            ax.plot(_X, Z, color='k', linewidth=1)
            ax.plot(_X, Z + svm.epsilon, color='grey', linestyle='--', linewidth=1)
            ax.plot(_X, Z - svm.epsilon, color='grey', linestyle='--', linewidth=1)

        plt.show()


class SVC(ClassifierMixin, SVM):
    def __init__(self, kernel='rbf', degree=3., gamma='scale', coef0=0., C=1.,
                 tol=1e-3, optimizer='SMO', epochs=1000, verbose=False):
        super().__init__(kernel, degree, gamma, coef0, C, tol, optimizer, epochs, verbose)
        if kernel is 'linear':
            self.coef_ = np.zeros(2)

    # Platt's SMO algorithm for SVC
    class SMOClassifier(SVM.SMO):

        def __init__(self, f, X, y, K, kernel='rbf', C=1., tol=1e-3, verbose=False):
            self.alphas = np.zeros(len(X))
            super().__init__(f, X, y, K, kernel, C, tol, verbose)

            # initialize variables and structures to implement improvements
            # on the original Platt's SMO algorithm described in Keerthi et
            # al. for better performance ed efficiency

            # set of indices
            self.I0 = set()  # { i : 0 < alphas[i] < C }
            self.I1 = set(range(len(X)))  # { i : y[i] = +1, alphas[i] = 0 }
            self.I2 = set()  # { i : y[i] = -1, alphas[i] = C }
            self.I3 = set()  # { i : y[i] = +1, alphas[i] = C }
            self.I4 = set()  # { i : y[i] = -1, alphas[i] = 0 }

            # multiple thresholds
            self.b_up_idx = 0
            self.b_low_idx = 0
            self.b_up = y[self.b_up_idx]
            self.b_low = y[self.b_low_idx]

        def _svm_output(self, i=None):
            return (self.alphas * self.y if i is None else self.y[i]).dot(self.K if i is None else self.K[i]) + self.b

        def _take_step(self, i1, i2):
            # skip if chosen alphas are the same
            if i1 == i2:
                return False

            alpha1 = self.alphas[i1]
            y1 = self.y[i1]
            E1 = self.errors[i1]

            alpha2 = self.alphas[i2]
            y2 = self.y[i2]
            E2 = self.errors[i2]

            s = y1 * y2

            # gamma = s * alpha1 + alpha2

            # compute L and H, the bounds on new possible alpha values
            # based on equations 13 and 14 in Platt's paper
            if y1 != y2:
                # L = max(0, gamma)
                # H = min(self.C, gamma + self.C)
                L = max(0, alpha2 - alpha1)
                H = min(self.C, self.C + alpha2 - alpha1)
            else:
                # L = max(0, gamma - self.C)
                # H = min(self.C, gamma)
                L = max(0, alpha2 + alpha1 - self.C)
                H = min(self.C, alpha2 + alpha1)

            if L == H:
                return False

            # compute the 2nd derivative of the objective function along
            # the diagonal line based on equation 15 in Platt's paper
            eta = self.K[i1, i1] + self.K[i2, i2] - 2 * self.K[i1, i2]

            # under normal circumstances, the objective function will be positive
            # definite, there will be a minimum along the direction of the linear
            # equality constraint, and eta will be greater than zero compute new
            # alpha2, a2, if eta is positive based on equation 16 in Platt's paper
            if eta > 0:
                # clip a2 based on bounds L and H based
                # on equation 17 in Platt's paper
                # a2 = np.clip(alpha2 + y2 * (E1 - E2) / eta, L, H)
                a2 = alpha2 + y2 * (E1 - E2) / eta
                if a2 < L:
                    a2 = L
                elif a2 > H:
                    a2 = H
            else:
                # under unusual circumstances, eta will not be positive. A negative eta
                # will occur if the kernel K does not obey Mercer’s condition, which can
                # cause the objective function to become indefinite. A zero eta can occur
                # even with a correct kernel, if more than one training example has the
                # same input vector. In any event, SMO will work even when eta is not
                # positive, in which case the objective function should be evaluated at
                # each end of the line segment based on equations 19 in Platt's paper

                f1 = y1 * (E1 + self.b) - alpha1 * self.K[i1, i1] - s * alpha2 * self.K[i1, i2]
                f2 = y2 * (E2 + self.b) - alpha2 * self.K[i2, i2] - s * alpha1 * self.K[i1, i2]
                L1 = alpha1 + s * (alpha2 - L)
                H1 = alpha1 + s * (alpha2 - H)
                Lobj = (L1 * f1 + L * f2 + 0.5 * L1 ** 2 * self.K[i1, i1] +
                        0.5 * L ** 2 * self.K[i2, i2] + s * L * L1 * self.K[i1, i2])
                Hobj = (H1 * f1 + H * f2 + 0.5 * H1 ** 2 * self.K[i1, i1] +
                        0.5 * H ** 2 * self.K[i2, i2] + s * H * H1 * self.K[i1, i2])

                # or, in our case, just by calling the objective function at a2 = L and a2 = H:

                alphas_adj = self.alphas.copy()
                alphas_adj[i2] = L
                Lobj_ = self.f.function(alphas_adj)
                alphas_adj[i2] = H
                Hobj_ = self.f.function(alphas_adj)

                assert Lobj == Lobj_
                assert Hobj == Hobj_

                if Lobj > Hobj + 1e-12:
                    a2 = L
                elif Lobj < Hobj - 1e-12:
                    a2 = H
                else:
                    a2 = alpha2

                warnings.warn('eta is zero')

            # if examples can't be optimized within tol, skip this pair
            if abs(a2 - alpha2) < 1e-12 * (a2 + alpha2 + 1e-12):
                return False

            # calculate new alpha1 based on equation 18 in Platt's paper
            a1 = alpha1 + s * (alpha2 - a2)

            old_b = self.b

            # update threshold b to reflect change in alphas based on
            # equations 20 and 21 in Platt's paper set new threshold
            # based on if a1 or a2 is bound by L and/or H
            if 0 < a1 < self.C:
                self.b += E1 + y1 * (a1 - alpha1) * self.K[i1, i1] + y2 * (a2 - alpha2) * self.K[i1, i2]
            elif 0 < a2 < self.C:
                self.b += E2 + y1 * (a1 - alpha1) * self.K[i1, i2] + y2 * (a2 - alpha2) * self.K[i2, i2]
            else:  # average thresholds if both are bound
                self.b += (E1 + y1 * (a1 - alpha1) * self.K[i1, i1] + y2 * (a2 - alpha2) * self.K[i1, i2] +
                           E2 + y1 * (a1 - alpha1) * self.K[i1, i2] + y2 * (a2 - alpha2) * self.K[i2, i2]) / 2

            # update weight vector to reflect change in a1 and a2, if
            # kernel is linear, based on equation 22 in Platt's paper
            if self.kernel is 'linear':
                self.w += y1 * (a1 - alpha1) * self.X[i1] + y2 * (a2 - alpha2) * self.X[i2]

            # # update error cache using new alphas
            # for i in range(len(alphas)):
            #     if 0 < alphas[i] < self.C:
            #         if i != i1 and i != i2:
            #             errors[i] += y1 * (a1 - alpha1) * K[i1, i] + y2 * (a2 - alpha2) * K[i2, i]
            # # update error cache using new alphas for i1 and i2
            # errors[i1] = y1 * (a1 - alpha1) * K[i1, i1] + y2 * (a2 - alpha2) * K[i1, i2]
            # errors[i2] = y1 * (a1 - alpha1) * K[i1, i2] + y2 * (a2 - alpha2) * K[i2, i2]

            # update error cache using new alphas
            self.errors += y1 * (a1 - alpha1) * self.K[i1] + y2 * (a2 - alpha2) * self.K[i2] + old_b - self.b

            # update model object with new alphas
            self.alphas[i1] = a1
            self.alphas[i2] = a2

            return True

        def _examine_example(self, i2):
            alpha2 = self.alphas[i2]
            r2 = self.errors[i2] * self.y[i2]

            if (r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0):

                # if the number of non-zero and non-C alphas is greater than 1
                if len(self.alphas[np.logical_and(self.alphas != 0, self.alphas != self.C)]) > 1:
                    # 1st heuristic is not implemented since it makes training slower
                    # use 2nd choice heuristic: choose max difference in error (section 2.2 of the Platt's paper)
                    if self.errors[i2] > 0:
                        i1 = np.argmin(self.errors)
                    else:  # elif errors[i2] <= 0:
                        i1 = np.argmax(self.errors)
                    step_result = self._take_step(i1, i2)
                    if step_result:
                        return True

                # loop over all non-zero and non-C alphas, starting at a random point
                for i1 in np.roll(np.where(np.logical_and(self.alphas != 0, self.alphas != self.C))[0],
                                  np.random.choice(np.arange(len(self.alphas)))):
                    step_result = self._take_step(i1, i2)
                    if step_result:
                        return True

                # loop over all possible alphas, starting at a random point
                for i1 in np.roll(np.arange(len(self.alphas)), np.random.choice(np.arange(len(self.alphas)))):
                    step_result = self._take_step(i1, i2)
                    if step_result:
                        return True

            return False

        def smo(self):
            if self.verbose:
                print('iter\tf(x)')

            num_changed = 0
            examine_all = True
            loop_counter = 0
            while num_changed > 0 or examine_all:
                loop_counter += 1
                num_changed = 0
                # loop over all training examples
                if examine_all:
                    for i in range(len(self.X)):
                        num_changed += self._examine_example(i)
                else:
                    # loop over examples where alphas are not already at their limits
                    for i in np.where(np.logical_and(self.alphas != 0, self.alphas != self.C))[0]:
                        num_changed += self._examine_example(i)
                if examine_all:
                    examine_all = False
                elif num_changed == 0:
                    examine_all = True

                if self.verbose:
                    print('{:4d}\t{:1.4e}'.format(loop_counter, self.f.function(self.alphas)))

            return self

    def fit(self, X, y):
        """
        Trains the model by solving a constrained quadratic programming problem.
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples] holding the class labels
        """
        self.labels = np.unique(y)
        if self.labels.size > 2:
            raise ValueError('use OneVsOneClassifier or OneVsRestClassifier from sklearn.multiclass '
                             'to train a model over more than two labels')
        y = np.where(y == self.labels[0], -1., 1.)

        n_samples = len(y)

        # kernel matrix
        K = self.kernels[self.kernel](X, X)

        P = K * np.outer(y, y)
        P = (P + P.T) / 2  # ensure P is symmetric
        q = -np.ones(n_samples)

        A = y.astype(np.float)  # equality matrix
        ub = np.ones(n_samples) * self.C  # upper bounds

        if self.optimizer in ('SMO', scipy_solve_svm):
            obj_fun = Quadratic(P, q)

            if self.optimizer is 'SMO':
                smo = self.SMOClassifier(obj_fun, X, y, K, self.kernel, self.C, self.tol, self.verbose).smo()
                alphas = smo.alphas
                if self.kernel is 'linear':
                    self.coef_ = smo.w
                # Platt’s paper uses the convention that the linear classifier is of the form
                # w.T x − b rather than the convention we use in this repo, that w.T x + b
                self.intercept_ = -smo.b

            else:
                alphas = scipy_solve_svm(obj_fun, A, ub, self.epochs, self.verbose)

        else:
            obj_fun = BoxConstrainedQuadratic(P, q, ub)

            if self.optimizer in (solve_qp, scipy_solve_qp):
                G = np.vstack((-np.identity(n_samples), np.identity(n_samples)))  # inequality matrix
                lb = np.zeros(n_samples)  # lower bounds
                h = np.hstack((lb, ub))  # inequality vector

                b = np.zeros(1)  # equality vector

                if self.optimizer is solve_qp:
                    qpsolvers.cvxopt_.options['show_progress'] = self.verbose
                    alphas = solve_qp(P, q, G, h, A, b, solver='cvxopt')

                else:
                    alphas = scipy_solve_qp(obj_fun, G, h, A, b, self.epochs, self.verbose)

            elif issubclass(self.optimizer, BoxConstrainedOptimizer):
                alphas = self.optimizer(obj_fun, max_iter=self.epochs, verbose=self.verbose).minimize()[0]

            elif issubclass(self.optimizer, Optimizer):
                # dual lagrangian relaxation of the box-constrained problem
                dual = LagrangianBoxConstrained(obj_fun)
                self.optimizer(dual, max_iter=self.epochs, verbose=self.verbose).minimize()
                alphas = dual.primal_solution

        sv = alphas > 1e-5
        self.support_ = np.arange(len(alphas))[sv]
        self.support_vectors_, self.sv_y, self.alphas = X[sv], y[sv], alphas[sv]
        self.dual_coef_ = self.alphas * self.sv_y

        if self.optimizer is not 'SMO':

            if self.kernel is 'linear':
                self.coef_ = np.dot(self.dual_coef_, self.support_vectors_)

            for n in range(len(self.alphas)):
                self.intercept_ += self.sv_y[n]
                self.intercept_ -= np.sum(self.dual_coef_ * K[self.support_[n], sv])
            self.intercept_ /= len(self.alphas)

        return self

    def decision_function(self, X):
        if self.kernel is not 'linear':
            return np.dot(self.dual_coef_, self.kernels[self.kernel](self.support_vectors_, X)) + self.intercept_
        return np.dot(X, self.coef_) + self.intercept_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.labels[1], self.labels[0])


class SVR(RegressorMixin, SVM):
    def __init__(self, kernel='rbf', degree=3., gamma='scale', coef0=0., C=1., tol=1e-3,
                 epsilon=0.1, optimizer='SMO', epochs=1000, verbose=False):
        super().__init__(kernel, degree, gamma, coef0, C, tol, optimizer, epochs, verbose)
        self.epsilon = epsilon  # epsilon insensitive loss value
        if kernel is 'linear':
            self.coef_ = np.zeros(1)

    # Platt's SMO algorithm for SVR
    class SMORegression(SVM.SMO):

        def __init__(self, f, X, y, K, kernel='rbf', C=1., epsilon=0.1, tol=1e-3, verbose=False):
            self.alphas_p = np.zeros(len(X))
            self.alphas_n = np.zeros(len(X))
            super().__init__(f, X, y, K, kernel, C, tol, verbose)
            self.epsilon = epsilon

            # initialize variables and structures to implement improvements
            # on the original Platt's SMO algorithm described in Shevade et
            # al. for better performance ed efficiency

            # set of indices
            self.I0 = set()
            self.I1 = set(range(len(X)))
            self.I2 = set()
            self.I3 = set()

            # multiple thresholds
            self.b_up_idx = 0
            self.b_low_idx = 0
            self.b_up = y[self.b_up_idx] + self.epsilon
            self.b_low = y[self.b_low_idx] - self.epsilon

        def _svm_output(self, i=None):
            return (self.alphas_p - self.alphas_n).dot(self.K if i is None else self.K[i]) + self.b

        def _take_step(self, i1, i2):
            # skip if chosen alphas are the same
            if i1 == i2:
                return False

            alpha1_p, alpha1_n = self.alphas_p[i1], self.alphas_n[i1]
            E1 = self.errors[i1]

            alpha2_p, alpha2_n = self.alphas_p[i2], self.alphas_n[i2]
            E2 = self.errors[i2]

            # compute kernel and 2nd derivative eta
            # based on equation 15 in Platt's paper
            eta = self.K[i1, i1] + self.K[i2, i2] - 2 * self.K[i1, i2]

            if eta < 0:
                eta = 0

            gamma = alpha1_p - alpha1_n + alpha2_p - alpha2_n

            case1 = case2 = case3 = case4 = False
            changed = finished = False

            delta_E = E1 - E2

            while not finished:  # occurs at most three times
                if (not case1 and
                        (alpha1_p > 0 or (alpha1_n == 0 and delta_E > 0)) and
                        (alpha2_p > 0 or (alpha2_n == 0 and delta_E < 0))):
                    # compute L and H wrt alpha1_p, alpha2_p
                    L = max(0, gamma - self.C)
                    H = min(self.C, gamma)
                    if L < H:
                        if eta > 0:
                            # a2 = np.clip(alpha2_p - delta_E / eta, L, H)
                            a2 = alpha2_p - delta_E / eta
                            if a2 < L:
                                a2 = L
                            elif a2 > H:
                                a2 = H
                        else:
                            Lobj = -L * delta_E
                            Hobj = -H * delta_E
                            if Lobj > Hobj:
                                a2 = L
                            else:
                                a2 = H
                            warnings.warn('eta is zero')
                        a1 = alpha1_p - (a2 - alpha2_p)
                        # update alpha1, alpha2_p if change is larger than some eps
                        if abs(a1 - alpha1_p) > 1e-12 or abs(a2 - alpha2_p) > 1e-12:
                            alpha1_p = a1
                            alpha2_p = a2
                            changed = True
                    else:
                        finished = True
                    case1 = True
                elif (not case2 and
                      (alpha1_p > 0 or (alpha1_n == 0 and delta_E > 2 * self.epsilon)) and
                      (alpha2_n > 0 or (alpha2_p == 0 and delta_E > 2 * self.epsilon))):
                    # compute L and H wrt alpha1_p, alpha2_n
                    L = max(0, -gamma)
                    H = min(self.C, -gamma + self.C)
                    if L < H:
                        if eta > 0:
                            # a2 = np.clip(alpha2_n + (delta_E - 2 * self.epsilon) / eta, L, H)
                            a2 = alpha2_n + (delta_E - 2 * self.epsilon) / eta
                            if a2 < L:
                                a2 = L
                            elif a2 > H:
                                a2 = H
                        else:
                            Lobj = L * (-2 * self.epsilon + delta_E)
                            Hobj = H * (-2 * self.epsilon + delta_E)
                            if Lobj > Hobj:
                                a2 = L
                            else:
                                a2 = H
                            warnings.warn('eta is zero')
                        a1 = alpha1_p + (a2 - alpha2_n)
                        # update alpha1, alpha2_n if change is larger than some eps
                        if abs(a1 - alpha1_p) > 1e-12 or abs(a2 - alpha2_n) > 1e-12:
                            alpha1_p = a1
                            alpha2_n = a2
                            changed = True
                    else:
                        finished = True
                    case2 = True
                elif (not case3 and
                      (alpha1_n > 0 or (alpha1_p == 0 and delta_E < -2 * self.epsilon)) and
                      (alpha2_p > 0 or (alpha2_n == 0 and delta_E < -2 * self.epsilon))):
                    # computer L and H wrt alpha1_n, alpha2_p
                    L = max(0, gamma)
                    H = min(self.C, self.C + gamma)
                    if L < H:
                        if eta > 0:
                            # a2 = np.clip(alpha2_p - (delta_E + 2 * self.epsilon) / eta, L, H)
                            a2 = alpha2_p - (delta_E + 2 * self.epsilon) / eta
                            if a2 < L:
                                a2 = L
                            elif a2 > H:
                                a2 = H
                        else:
                            Lobj = -L * (2 * self.epsilon + delta_E)
                            Hobj = -H * (2 * self.epsilon + delta_E)
                            if Lobj > Hobj:
                                a2 = L
                            else:
                                a2 = H
                            warnings.warn('eta is zero')
                        a1 = alpha1_n + (a2 - alpha2_p)
                        # update alpha1_n, alpha2_p if change is larger than some eps
                        if abs(a1 - alpha1_n) > 1e-12 or abs(a2 - alpha2_p) > 1e-12:
                            alpha1_n = a1
                            alpha2_p = a2
                            changed = True
                    else:
                        finished = True
                    case3 = True
                elif (not case4 and
                      (alpha1_n > 0 or (alpha1_p == 0 and delta_E < 0)) and
                      (alpha2_n > 0 or (alpha2_p == 0 and delta_E > 0))):
                    # compute L and H wrt alpha1_n, alpha2_n
                    L = max(0, -gamma - self.C)
                    H = min(self.C, -gamma)
                    if L < H:
                        if eta > 0:
                            # a2 = np.clip(alpha2_n + delta_E / eta, L, H)
                            a2 = alpha2_n + delta_E / eta
                            if a2 < L:
                                a2 = L
                            elif a2 > H:
                                a2 = H
                        else:
                            Lobj = L * delta_E
                            Hobj = H * delta_E
                            if Lobj > Hobj:
                                a2 = L
                            else:
                                a2 = H
                            warnings.warn('eta is zero')
                        a1 = alpha1_n - (a2 - alpha2_n)
                        # update alpha1_n, alpha2_n if change is larger than some eps
                        if abs(a1 - alpha1_n) > 1e-12 or abs(a2 - alpha2_n) > 1e-12:
                            alpha1_n = a1
                            alpha2_n = a2
                            changed = True
                    else:
                        finished = True
                    case4 = True
                else:
                    finished = True

                delta_E += eta * ((alpha2_p - alpha2_n) - (self.alphas_p[i2] - self.alphas_n[i2]))

            if not changed:
                return False

            # update error cache using new alphas
            for i in self.I0:
                if i != i1 and i != i2:
                    self.errors[i] += (
                            ((self.alphas_p[i1] - self.alphas_n[i1]) - (alpha1_p - alpha1_n)) * self.K[i1, i] +
                            ((self.alphas_p[i2] - self.alphas_n[i2]) - (alpha2_p - alpha2_n)) * self.K[i2, i])
            # update error cache using new alphas for i1 and i2
            self.errors[i1] += (((self.alphas_p[i1] - self.alphas_n[i1]) - (alpha1_p - alpha1_n)) * self.K[i1, i1] +
                                ((self.alphas_p[i2] - self.alphas_n[i2]) - (alpha2_p - alpha2_n)) * self.K[i1, i2])
            self.errors[i2] += (((self.alphas_p[i1] - self.alphas_n[i1]) - (alpha1_p - alpha1_n)) * self.K[i1, i2] +
                                ((self.alphas_p[i2] - self.alphas_n[i2]) - (alpha2_p - alpha2_n)) * self.K[i2, i2])

            # to prevent precision problems
            if alpha1_p > self.C - 1e-10 * self.C:
                alpha1_p = self.C
            elif alpha1_p <= 1e-10 * self.C:
                alpha1_p = 0

            if alpha1_n > self.C - 1e-10 * self.C:
                alpha1_n = self.C
            elif alpha1_n <= 1e-10 * self.C:
                alpha1_n = 0

            if alpha2_p > self.C - 1e-10 * self.C:
                alpha2_p = self.C
            elif alpha2_p <= 1e-10 * self.C:
                alpha2_p = 0

            if alpha2_n > self.C - 1e-10 * self.C:
                alpha2_n = self.C
            elif alpha2_n <= 1e-10 * self.C:
                alpha2_n = 0

            # update model object with new alphas
            self.alphas_p[i1], self.alphas_p[i2] = alpha1_p, alpha2_p
            self.alphas_n[i1], self.alphas_n[i2] = alpha1_n, alpha2_n

            # update the sets of indices for i1
            if 0 < alpha1_p < self.C or 0 < alpha1_n < self.C:
                self.I0.add(i1)
            else:
                self.I0.discard(i1)

            if alpha1_p == 0 and alpha1_n == 0:
                self.I1.add(i1)
            else:
                self.I1.discard(i1)

            if alpha1_p == 0 and alpha1_n == self.C:
                self.I2.add(i1)
            else:
                self.I2.discard(i1)

            if alpha1_p == self.C and alpha1_n == 0:
                self.I3.add(i1)
            else:
                self.I3.discard(i1)

            # update the sets of indices for i2
            if 0 < alpha2_p < self.C or 0 < alpha2_n < self.C:
                self.I0.add(i2)
            else:
                self.I0.discard(i2)

            if alpha2_p == 0 and alpha2_n == 0:
                self.I1.add(i1)
            else:
                self.I1.discard(i2)

            if alpha2_p == 0 and alpha2_n == self.C:
                self.I2.add(i2)
            else:
                self.I2.discard(i2)

            if alpha2_p == self.C and alpha2_n == 0:
                self.I3.add(i2)
            else:
                self.I3.discard(i2)

            # compute new thresholds
            self.b_up_idx = -1
            self.b_low_idx = -1
            self.b_up = sys.float_info.max
            self.b_low = -sys.float_info.max

            for i in self.I0:
                if 0 < self.alphas_n[i] < self.C and self.errors[i] + self.epsilon > self.b_low:
                    self.b_low = self.errors[i] + self.epsilon
                    self.b_low_idx = i
                elif 0 < self.alphas_p[i] < self.C and self.errors[i] - self.epsilon > self.b_low:
                    self.b_low = self.errors[i] - self.epsilon
                    self.b_low_idx = i

                if 0 < self.alphas_p[i] < self.C and self.errors[i] - self.epsilon < self.b_up:
                    self.b_up = self.errors[i] - self.epsilon
                    self.b_up_idx = i
                elif 0 < self.alphas_n[i] < self.C and self.errors[i] + self.epsilon < self.b_up:
                    self.b_up = self.errors[i] + self.epsilon
                    self.b_up_idx = i

            if i1 not in self.I0:
                if i1 in self.I2 and self.errors[i1] + self.epsilon > self.b_low:
                    self.b_low = self.errors[i1] + self.epsilon
                    self.b_low_idx = i1
                elif i1 in self.I1 and self.errors[i1] - self.epsilon > self.b_low:
                    self.b_low = self.errors[i1] - self.epsilon
                    self.b_low_idx = i1

                if i1 in self.I3 and self.errors[i1] - self.epsilon < self.b_up:
                    self.b_up = self.errors[i1] - self.epsilon
                    self.b_up_idx = i1
                elif i1 in self.I1 and self.errors[i1] + self.epsilon < self.b_up:
                    self.b_up = self.errors[i1] + self.epsilon
                    self.b_up_idx = i1

            if i2 not in self.I0:
                if i2 in self.I2 and self.errors[i2] + self.epsilon > self.b_low:
                    self.b_low = self.errors[i2] + self.epsilon
                    self.b_low_idx = i2
                elif i2 in self.I1 and self.errors[i2] - self.epsilon > self.b_low:
                    self.b_low = self.errors[i2] - self.epsilon
                    self.b_low_idx = i2

                if i2 in self.I3 and self.errors[i2] - self.epsilon < self.b_up:
                    self.b_up = self.errors[i2] - self.epsilon
                    self.b_up_idx = i2
                elif i2 in self.I1 and self.errors[i2] + self.epsilon < self.b_up:
                    self.b_up = self.errors[i2] + self.epsilon
                    self.b_up_idx = i2

            if self.b_low_idx == -1 or self.b_up_idx == -1:
                raise Exception('unexpected status')

            return True

        def _examine_example(self, i2):
            alpha2_p, alpha2_n = self.alphas_p[i2], self.alphas_n[i2]

            if i2 in self.I0:
                E2 = self.errors[i2]
            else:
                E2 = self.errors[i2] = self.y[i2] - self._svm_output(i2)
                if i2 in self.I1:
                    if E2 + self.epsilon < self.b_up:
                        self.b_up = E2 + self.epsilon
                        self.b_up_idx = i2
                    elif E2 - self.epsilon > self.b_low:
                        self.b_low = E2 - self.epsilon
                        self.b_low_idx = i2
                elif i2 in self.I2 and E2 + self.epsilon > self.b_low:
                    self.b_low = E2 + self.epsilon
                    self.b_low_idx = i2
                elif i2 in self.I3 and E2 - self.epsilon < self.b_up:
                    self.b_up = E2 - self.epsilon
                    self.b_up_idx = i2
            i1 = -1
            optimal = True
            if i2 in self.I0:
                if 0 < alpha2_p < self.C:
                    if self.b_low - (E2 - self.epsilon) > 2. * self.tol:
                        optimal = False
                        i1 = self.b_low_idx
                        if (E2 - self.epsilon) - self.b_up > self.b_low - (E2 - self.epsilon):
                            i1 = self.b_up_idx
                    elif (E2 - self.epsilon) - self.b_up > 2. * self.tol:
                        optimal = False
                        i1 = self.b_up_idx
                        if self.b_low - (E2 - self.epsilon) > (E2 - self.epsilon) - self.b_up:
                            i1 = self.b_low_idx
                elif 0 < alpha2_n < self.C:
                    if self.b_low - (E2 + self.epsilon) > 2. * self.tol:
                        optimal = False
                        i1 = self.b_low_idx
                        if (E2 + self.epsilon) - self.b_up > self.b_low - (E2 + self.epsilon):
                            i1 = self.b_up_idx
                    elif (E2 + self.epsilon) - self.b_up > 2. * self.tol:
                        optimal = False
                        i1 = self.b_up_idx
                        if self.b_low - (E2 + self.epsilon) > (E2 + self.epsilon) - self.b_up:
                            i1 = self.b_low_idx
            elif i2 in self.I1:
                if self.b_low - (E2 + self.epsilon) > 2. * self.tol:
                    optimal = False
                    i1 = self.b_low_idx
                    if (E2 + self.epsilon) - self.b_up > self.b_low - (E2 + self.epsilon):
                        i1 = self.b_up_idx
                elif (E2 - self.epsilon) - self.b_up > 2. * self.tol:
                    optimal = False
                    i1 = self.b_up_idx
                    if self.b_low - (E2 - self.epsilon) > (E2 - self.epsilon) - self.b_up:
                        i1 = self.b_low_idx
            elif i2 in self.I2:
                if (E2 + self.epsilon) - self.b_up > 2 * self.tol:
                    optimal = False
                    i1 = self.b_up_idx
            elif i2 in self.I3:
                if self.b_low - (E2 - self.epsilon) > 2. * self.tol:
                    optimal = False
                    i1 = self.b_low_idx
            else:
                raise Exception('the index could not be found')

            if optimal:
                return False
            return self._take_step(i1, i2)

        def smo(self):
            if self.verbose:
                print('iter\tf(x)')

            num_changed = 0
            examine_all = True
            loop_counter = 0
            while num_changed > 0 or examine_all:
                loop_counter += 1
                num_changed = 0
                # loop over all training examples
                if examine_all:
                    for i in range(len(self.X)):
                        num_changed += self._examine_example(i)
                else:
                    # loop over examples where alphas are not already at their limits
                    for i in range(len(self.X)):
                        if 0 < self.alphas_p[i] < self.C or 0 < self.alphas_n[i] < self.C:
                            # for i in np.where(np.logical_and(alphas_p != 0, alphas_n != self.C))[0]:
                            num_changed += self._examine_example(i)
                            if self.b_up > self.b_low - 2 * self.tol:
                                num_changed = 0
                                break
                if examine_all:
                    examine_all = False
                elif num_changed == 0:
                    examine_all = True

                if self.verbose:
                    print('{:4d}\t{:1.4e}'.format(
                        loop_counter, self.f.function(np.hstack((self.alphas_p, self.alphas_n)))))

            self.b = (self.b_low + self.b_up) / 2

            return self

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
        K = self.kernels[self.kernel](X, X)

        P = np.vstack((np.hstack((K, -K)),  # alphas_p, alphas_n
                       np.hstack((-K, K))))  # alphas_n, alphas_p
        P = (P + P.T) / 2  # ensure P is symmetric
        q = np.hstack((-y, y)) + self.epsilon

        A = np.hstack((np.ones(n_samples), -np.ones(n_samples)))  # equality matrix
        ub = np.ones(2 * n_samples) * self.C  # upper bounds

        if self.optimizer in ('SMO', scipy_solve_svm):
            obj_fun = Quadratic(P, q)

            if self.optimizer is 'SMO':
                smo = self.SMORegression(
                    obj_fun, X, y, K, self.kernel, self.C, self.epsilon, self.tol, self.verbose).smo()
                alphas_p, alphas_n = smo.alphas_p, smo.alphas_n
                if self.kernel is 'linear':
                    self.coef_ = smo.w
                self.intercept_ = smo.b

            else:
                alphas = scipy_solve_svm(obj_fun, A, ub, self.epochs, self.verbose)
                alphas_p = alphas[:n_samples]
                alphas_n = alphas[n_samples:]

        else:
            obj_fun = BoxConstrainedQuadratic(P, q, ub)

            if self.optimizer in (solve_qp, scipy_solve_qp):
                G = np.vstack((-np.identity(2 * n_samples), np.identity(2 * n_samples)))  # inequality matrix
                lb = np.zeros(2 * n_samples)  # lower bounds
                h = np.hstack((lb, ub))  # inequality vector

                b = np.zeros(1)  # equality vector

                if self.optimizer is solve_qp:
                    qpsolvers.cvxopt_.options['show_progress'] = self.verbose
                    alphas = solve_qp(P, q, G, h, A, b, solver='cvxopt')
                else:
                    alphas = scipy_solve_qp(obj_fun, G, h, A, b, self.epochs, self.verbose)

            elif issubclass(self.optimizer, BoxConstrainedOptimizer):
                alphas = self.optimizer(obj_fun, max_iter=self.epochs, verbose=self.verbose).minimize()[0]

            elif issubclass(self.optimizer, Optimizer):
                # dual lagrangian relaxation of the box-constrained problem
                dual = LagrangianBoxConstrained(obj_fun)
                self.optimizer(dual, max_iter=self.epochs, verbose=self.verbose).minimize()
                alphas = dual.primal_solution

            alphas_p = alphas[:n_samples]
            alphas_n = alphas[n_samples:]

        sv = np.logical_or(alphas_p > 1e-5, alphas_n > 1e-5)
        self.support_ = np.arange(len(alphas_p))[sv]
        self.support_vectors_, self.sv_y, self.alphas_p, self.alphas_n = X[sv], y[sv], alphas_p[sv], alphas_n[sv]
        self.dual_coef_ = self.alphas_p - self.alphas_n

        if self.kernel is 'linear':
            self.coef_ = np.dot(self.dual_coef_, self.support_vectors_)

        if self.optimizer is not 'SMO':
            for n in range(len(self.alphas_p)):
                self.intercept_ += self.sv_y[n]
                self.intercept_ -= np.sum(self.dual_coef_ * K[self.support_[n], sv])
            self.intercept_ -= self.epsilon
            self.intercept_ /= len(self.alphas_p)

        return self

    def predict(self, X):
        if self.kernel is not 'linear':
            return np.dot(self.dual_coef_, self.kernels[self.kernel](self.support_vectors_, X)) + self.intercept_
        return np.dot(X, self.coef_) + self.intercept_
