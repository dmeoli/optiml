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
                 eps=1e-3, tol=1e-3, optimizer='SMO', epochs=1000, verbose=False):
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
        if not np.isscalar(eps):
            raise ValueError('eps is not a real scalar')
        if not eps > 0:
            raise ValueError('eps must be > 0')
        self.eps = eps
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

    def _take_step(self):
        raise NotImplementedError

    def _examine_example(self):
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
                 eps=1e-3, tol=1e-3, optimizer='SMO', epochs=1000, verbose=False):
        super().__init__(kernel, degree, gamma, coef0, C, eps, tol, optimizer, epochs, verbose)
        if kernel is 'linear':
            self.coef_ = np.zeros(2)
        self.intercept_ = 0.

    # Platt's SMO algorithm for SVC

    def _take_step(self, f, K, X, y, alphas, errors, i1, i2):
        # skip if chosen alphas are the same
        if i1 == i2:
            return False, alphas, errors

        alpha1 = alphas[i1]
        y1 = y[i1]
        E1 = errors[i1]

        alpha2 = alphas[i2]
        y2 = y[i2]
        E2 = errors[i2]

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
            return False, alphas, errors

        # compute kernel and 2nd derivative eta
        # based on equation 15 in Platt's paper
        eta = K[i1, i1] + K[i2, i2] - 2 * K[i1, i2]

        # compute new alpha2 if eta is positive
        # based on equation 16 in Platt's paper
        if eta > 0:
            # clip a2 based on bounds L and H based
            # on equation 17 in Platt's paper
            # a2 = np.clip(alpha2 + y2 * (E1 - E2) / eta, L, H)
            a2 = alpha2 + y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:  # else move new a2 to bound with greater objective function value
            # under unusual circumstances, eta will not be positive. A negative eta will occur if the kernel K does
            # not obey Mercer’s condition, which can cause the objective function to become indefinite. A zero eta
            # can occur even with a correct kernel, if more than one training example has the same input vector.
            # In any event, SMO will work even when eta is not positive, in which case the objective function should
            # be evaluated at each end of the line segment:

            # f1 = y1 * E1 - alpha1 * K[i1, i1] - s * alpha2 * K[i1, i2]
            # f2 = y2 * E2 - alpha2 * K[i2, i2] - s * alpha1 * K[i1, i2]
            # L1 = alpha1 + s * (alpha2 - L)
            # H1 = alpha1 + s * (alpha2 - H)
            # Lobj = -0.5 * L1 * L1 * K[i1, i1] - 0.5 * L * L * K[i2, i2] - s * L * L1 * K[i1, i2] - L1 * f1 - L * f2
            # Hobj = -0.5 * H1 * H1 * K[i1, i1] - 0.5 * H * H * K[i2, i2] - s * H * H1 * K[i1, i2] - H1 * f1 - H * f2

            Lobj = y2 * (E1 - E2) * L
            Hobj = y2 * (E1 - E2) * H

            alphas_adj = alphas.copy()
            alphas_adj[i2] = L
            # objective function output with a2 = L
            Lobj_ = f.function(alphas_adj)
            alphas_adj[i2] = H
            # objective function output with a2 = H
            Hobj_ = f.function(alphas_adj)

            assert Lobj == Lobj_
            assert Hobj == Hobj_

            if Lobj > Hobj + self.eps:
                a2 = L
            elif Lobj < Hobj - self.eps:
                a2 = H
            else:
                a2 = alpha2

            warnings.warn('eta is zero')

        # if examples can't be optimized within tol, skip this pair
        if abs(a2 - alpha2) < self.eps * (a2 + alpha2 + self.eps):
            return False, alphas, errors

        # calculate new alpha1 based on equation 18 in Platt's paper
        a1 = alpha1 + s * (alpha2 - a2)

        b = self.intercept_

        # update threshold b to reflect change in alphas
        # based on equations 20 and 21 in Platt's paper
        # set new threshold based on if a1 or a2 is bound by L and/or H
        if 0 < a1 < self.C:
            self.intercept_ += E1 + y1 * (a1 - alpha1) * K[i1, i1] + y2 * (a2 - alpha2) * K[i1, i2]
        elif 0 < a2 < self.C:
            self.intercept_ += E2 + y1 * (a1 - alpha1) * K[i1, i2] + y2 * (a2 - alpha2) * K[i2, i2]
        else:  # average thresholds if both are bound
            self.intercept_ += (E1 + y1 * (a1 - alpha1) * K[i1, i1] + y2 * (a2 - alpha2) * K[i1, i2] +
                                E2 + y1 * (a1 - alpha1) * K[i1, i2] + y2 * (a2 - alpha2) * K[i2, i2]) / 2

        # update weight vector to reflect change in a1 and a2, if
        # kernel is linear, based on equation 22 in Platt's paper
        if self.kernel is 'linear':
            self.coef_ += y1 * (a1 - alpha1) * X[i1] + y2 * (a2 - alpha2) * X[i2]

        # # update error cache using new alphas
        # for i in range(len(alphas)):
        #     if 0 < alphas[i] < self.C:
        #         if i != i1 and i != i2:
        #             errors[i] += y1 * (a1 - alpha1) * K[i1, i] + y2 * (a2 - alpha2) * K[i2, i]
        # # update error cache using new alphas for i1 and i2
        # errors[i1] = y1 * (a1 - alpha1) * K[i1, i1] + y2 * (a2 - alpha2) * K[i1, i2]
        # errors[i2] = y1 * (a1 - alpha1) * K[i1, i2] + y2 * (a2 - alpha2) * K[i2, i2]

        # update error cache using new alphas
        errors += y1 * (a1 - alpha1) * K[i1] + y2 * (a2 - alpha2) * K[i2] + b - self.intercept_

        # update model object with new alphas
        alphas[i1] = a1
        alphas[i2] = a2

        return True, alphas, errors

    def _examine_example(self, f, K, X, y, alphas, errors, i2):
        y2 = y[i2]
        alpha2 = alphas[i2]
        E2 = errors[i2]
        r2 = E2 * y2

        # proceed if error is within specified tol
        if (r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0):

            # if the number of non-zero and non-C alphas is greater than 1
            if len(alphas[(alphas != 0) & (alphas != self.C)]) > 1:
                # use 2nd choice heuristic: choose max difference in error (section 2.2 of the Platt's paper)
                if errors[i2] > 0:
                    i1 = np.argmin(errors)
                elif errors[i2] <= 0:
                    i1 = np.argmax(errors)
                step_result, alphas, errors = self._take_step(f, K, X, y, alphas, errors, i1, i2)
                if step_result:
                    return True, alphas, errors

            # loop over all non-zero and non-C alphas, starting at a random point
            for i1 in np.roll(np.where((alphas != 0) & (alphas != self.C))[0],
                              np.random.choice(np.arange(len(alphas)))):
                step_result, alphas, errors = self._take_step(f, K, X, y, alphas, errors, i1, i2)
                if step_result:
                    return True, alphas, errors

            # loop over all possible alphas, starting at a random point
            for i1 in np.roll(np.arange(len(alphas)), np.random.choice(np.arange(len(alphas)))):
                step_result, alphas, errors = self._take_step(f, K, X, y, alphas, errors, i1, i2)
                if step_result:
                    return True, alphas, errors

        return False, alphas, errors

    def smo(self, f, K, X, y, alphas, errors):
        it = 0
        if self.verbose:
            print('iter\tf(x)')

        num_changed = 0
        examine_all = True
        while num_changed > 0 or examine_all:
            num_changed = 0
            # loop over all training examples
            if examine_all:
                for i in range(len(y)):
                    examine_result, alphas, errors = self._examine_example(f, K, X, y, alphas, errors, i)
                    num_changed += examine_result
                    if examine_result and self.verbose:
                        it += 1
                        print('{:4d}\t{:1.4e}'.format(it, f.function(alphas)))
            else:
                # loop over examples where alphas are not already at their limits
                for i in np.where((alphas != 0) & (alphas != self.C))[0]:
                    examine_result, alphas, errors = self._examine_example(f, K, X, y, alphas, errors, i)
                    num_changed += examine_result
                    if examine_result and self.verbose:
                        it += 1
                        print('{:4d}\t{:1.4e}'.format(it, f.function(alphas)))
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

        return alphas

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
                alphas = np.zeros(obj_fun.n)
                # initial error is equal to SVC output (the decision function) - y
                errors = (alphas * y).dot(K) + self.intercept_ - y
                alphas = self.smo(obj_fun, K, X, y, alphas, errors)

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
    def __init__(self, kernel='rbf', degree=3., gamma='scale', coef0=0., C=1., eps=1e-3,
                 tol=1e-3, epsilon=0.1, optimizer='SMO', epochs=1000, verbose=False):
        super().__init__(kernel, degree, gamma, coef0, C, eps, tol, optimizer, epochs, verbose)
        self.epsilon = epsilon  # epsilon insensitive loss value
        if kernel is 'linear':
            self.coef_ = np.zeros(1)
        self.intercept_ = -epsilon

    # Platt's SMO algorithm for SVR

    def _take_step(self, f, K, X, y, alphas, errors, i1, i2):
        # skip if chosen alphas are the same
        if i1 == i2:
            return False, alphas, errors

        alphas_p, alphas_n = np.split(alphas, 2)

        alpha1_p, alpha1_n = alphas_p[i1], alphas_n[i1]
        E1 = errors[i1]

        alpha2_p, alpha2_n = alphas_p[i2], alphas_n[i2]
        E2 = errors[i2]

        # compute kernel and 2nd derivative eta
        # based on equation 15 in Platt's paper
        eta = K[i1, i1] + K[i2, i2] - 2 * K[i1, i2]

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

            delta_E += eta * ((alpha2_p - alpha2_n) - (alphas_p[i2] - alphas_n[i2]))

        if not changed:
            return False, alphas, errors

        # update error cache using new alphas
        for i in range(len(alphas)):
            if i != i1 and i != i2:
                errors[i] += (((alphas_p[i1] - alphas_n[i1]) - (alpha1_p - alpha1_n)) * K[i1, i] +
                              ((alphas_p[i2] - alphas_n[i2]) - (alpha2_p - alpha2_n)) * K[i2, i])
        # update error cache using new alphas for i1 and i2
        errors[i1] += (((alphas_p[i1] - alphas_n[i1]) - (alpha1_p - alpha1_n)) * K[i1, i1] +
                       ((alphas_p[i2] - alphas_n[i2]) - (alpha2_p - alpha2_n)) * K[i1, i2])
        errors[i2] += (((alphas_p[i1] - alphas_n[i1]) - (alpha1_p - alpha1_n)) * K[i1, i2] +
                       ((alphas_p[i2] - alphas_n[i2]) - (alpha2_p - alpha2_n)) * K[i2, i2])

        # update model object with new alphas
        alphas_p[i1], alphas_p[i2] = alpha1_p, alpha2_p
        alphas_n[i1], alphas_n[i2] = alpha1_n, alpha2_n

        return True, np.hstack((alphas_p, alphas_n)), errors

    def _examine_example(self, f, K, X, y, alphas, errors, i2):
        alphas_p, alphas_n = np.split(alphas, 2)

        y2 = y[i2]
        alpha2_p, alpha2_n = alphas_p[i2], alphas_n[i2]
        E2 = errors[i2]
        r2 = E2 * y2

        # proceed if error is within specified tol
        if ((r2 > self.tol and alpha2_n < self.C) or
                (r2 < self.tol and alpha2_n > 0) or
                (-r2 > self.tol and alpha2_p < self.C) or
                (-r2 > self.tol and alpha2_p > 0)):

            # if the number of non-zero and non-C alphas is greater than 1
            if len(alphas[(alphas != 0) & (alphas != self.C)]) > 1:
                # use 2nd choice heuristic: choose max difference in error (section 2.2 of the Platt's paper)
                if errors[i2] > 0:
                    i1 = np.argmin(errors)
                elif errors[i2] <= 0:
                    i1 = np.argmax(errors)
                step_result, alphas, errors = self._take_step(f, K, X, y, alphas, errors, i1, i2)
                if step_result:
                    return True, alphas, errors

            # loop over all non-zero and non-C alphas, starting at a random point
            for i1 in np.roll(np.where((alphas_p != 0) & (alphas_p != self.C))[0],
                              np.random.choice(np.arange(len(alphas_p)))):
                step_result, alphas, errors = self._take_step(f, K, X, y, alphas, errors, i1, i2)
                if step_result:
                    return True, alphas, errors

            # loop over all possible alphas, starting at a random point
            for i1 in np.roll(np.arange(len(alphas_p)), np.random.choice(np.arange(len(alphas_p)))):
                step_result, alphas, errors = self._take_step(f, K, X, y, alphas, errors, i1, i2)
                if step_result:
                    return True, alphas, errors

        return False, alphas, errors

    def smo(self, f, K, X, y, alphas, errors):
        it = 0
        if self.verbose:
            print('iter\tf(x)')

        num_changed = 0
        examine_all = True
        sig_fig = -100
        loop_counter = 0
        while (num_changed > 0 or examine_all) or sig_fig < 3:
            loop_counter += 1
            num_changed = 0
            # loop over all training examples
            if examine_all:
                for i in range(len(y)):
                    examine_result, alphas, errors = self._examine_example(f, K, X, y, alphas, errors, i)
                    num_changed += examine_result
                    if examine_result and self.verbose:
                        it += 1
                        print('{:4d}\t{:1.4e}'.format(it, f.function(alphas)))
            else:
                # loop over examples where alphas are not already at their limits
                for i in np.where((alphas != 0) & (alphas != self.C))[0]:
                    examine_result, alphas, errors = self._examine_example(f, K, X, y, alphas, errors, i)
                    num_changed += examine_result
                    if examine_result and self.verbose:
                        it += 1
                        print('{:4d}\t{:1.4e}'.format(it, f.function(alphas)))
            if loop_counter % 2 == 0:
                min_num_changed = max(1, 0.1 * len(alphas))
            else:
                min_num_changed = 1
            if examine_all:
                examine_all = False
            elif num_changed < min_num_changed:
                examine_all = True

        return alphas

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
                alphas = np.zeros(2 * n_samples)
                # initial error is equal to SVR output (the predict function) - y
                errors = (alphas[:n_samples] - alphas[n_samples:]).dot(K) + self.intercept_ - y
                alphas = self.smo(obj_fun, K, X, y, alphas, errors)

            else:
                alphas = scipy_solve_svm(obj_fun, A, ub, self.epochs, self.verbose)

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

        for n in range(len(self.alphas_p)):
            self.intercept_ += self.sv_y[n]
            self.intercept_ -= np.sum(self.dual_coef_ * K[self.support_[n], sv])
        self.intercept_ /= len(self.alphas_p)

        return self

    def predict(self, X):
        if self.kernel is not 'linear':
            return np.dot(self.dual_coef_, self.kernels[self.kernel](self.support_vectors_, X)) + self.intercept_
        return np.dot(X, self.coef_) + self.intercept_
