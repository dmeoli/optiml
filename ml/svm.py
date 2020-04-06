import matplotlib.pyplot as plt
import numpy as np
import qpsolvers
from matplotlib.lines import Line2D
from qpsolvers import solve_qp
from sklearn.base import ClassifierMixin, BaseEstimator, RegressorMixin

from optimization.optimization_function import BoxConstrainedQuadratic, LagrangianBoxConstrained, Quadratic
from optimization.optimizer import BoxConstrainedOptimizer, Optimizer
from utils import scipy_solve_qp

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
        self.C = C
        if not np.isscalar(tol):
            raise ValueError('tol is not a real scalar')
        if not tol > 0:
            raise ValueError('tol must be > 0')
        self.tol = tol
        if optimizer not in (solve_qp, scipy_solve_qp, 'SMO') and not issubclass(optimizer, Optimizer):
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

    def _take_step(self, f, K, y, alphas, errors, i1, i2):
        # skip if chosen alphas are the same
        if i1 == i2:
            return False, alphas, errors

        alpha1 = alphas[i1]
        y1 = y[i1]
        E1 = errors[i1]

        alpha2 = alphas[i2]
        y2 = y[i2]
        E2 = errors[i2]

        # compute L and H, the bounds on new possible alpha values
        # based on equations 13 and 14 in Platt's paper
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)

        if L == H:
            return False, alphas, errors

        # compute kernel and 2nd derivative eta
        # based on equation 15 in Platt's paper
        eta = K[i1, i1] + K[i2, i2] - 2 * K[i1, i2]

        # compute new alpha2 if eta is positive based on equation 16 in Platt's paper
        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            # clip a2 based on bounds L and H based
            # on equation 17 in Platt's paper
            if a2 <= L:
                a2 = L
            elif a2 >= H:
                a2 = H
        else:  # else move new a2 to bound with greater objective function value
            alphas_adj = alphas.copy()
            alphas_adj[i2] = L
            # objective function output with a2 = L
            Lobj = f.function(alphas_adj)
            alphas_adj[i2] = H
            # objective function output with a2 = H
            Hobj = f.function(alphas_adj)
            if Lobj < (Hobj - self.tol):
                a2 = L
            elif Lobj > (Hobj + self.tol):
                a2 = H
            else:
                a2 = alpha2

        # if examples can't be optimized within tol, skip this pair
        if np.abs(a2 - alpha2) < self.tol * (a2 + alpha2 + self.tol):
            return False, alphas, errors

        # calculate new alpha1
        s = y1 * y2
        a1 = alpha1 + s * (alpha2 - a2)

        # update threshold b to reflect change in alphas
        # based on equations 20 and 21 in Platt's paper
        # calculate both possible thresholds
        b1 = E1 + y1 * (a1 - alpha1) * K[i1, i1] + y2 * (a2 - alpha2) * K[i1, i2] + self.intercept_
        b2 = E2 + y1 * (a1 - alpha1) * K[i1, i2] + y2 * (a2 - alpha2) * K[i2, i2] + self.intercept_

        # set new threshold based on if a1 or a2 is bound by L and/or H
        if 0 < a1 < self.C:
            b_new = b1
        elif 0 < a2 < self.C:
            b_new = b2
        else:  # average thresholds if both are bound
            b_new = (b1 + b2) / 2

        # update error cache: for optimized alphas is set to 0 if they're unbound
        for idx, alpha in zip([i1, i2], [a1, a2]):
            if 0 < alpha < self.C:
                errors[idx] = 0.
        # set non-optimized errors
        non_opt = [n for n in range(len(alphas)) if n != i1 and n != i2]
        errors[non_opt] += (y1 * (a1 - alpha1) * K[i1, non_opt] +
                            y2 * (a2 - alpha2) * K[i2, non_opt] + self.intercept_ - b_new)

        # update model object with new alphas
        alphas[i1] = a1
        alphas[i2] = a2

        # update model threshold
        self.intercept_ = b_new

        return True, alphas, errors

    def _examine_example(self, f, K, y, alphas, errors, i2):
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
                step_result, alphas, errors = self._take_step(f, K, y, alphas, errors, i1, i2)
                if step_result:
                    return True, alphas, errors

            # loop over all non-zero and non-C alphas, starting at a random point
            for i1 in np.roll(np.where((alphas != 0) & (alphas != self.C))[0],
                              np.random.choice(np.arange(len(alphas)))):
                step_result, alphas, errors = self._take_step(f, K, y, alphas, errors, i1, i2)
                if step_result:
                    return True, alphas, errors

            # loop over all possible alphas, starting at a random point
            for i1 in np.roll(np.arange(len(alphas)), np.random.choice(np.arange(len(alphas)))):
                step_result, alphas, errors = self._take_step(f, K, y, alphas, errors, i1, i2)
                if step_result:
                    return True, alphas, errors

        return False, alphas, errors

    def smo(self, f, K, y, alphas, errors):
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
                    examine_result, alphas, errors = self._examine_example(f, K, y, alphas, errors, i)
                    num_changed += examine_result
                    if examine_result and self.verbose:
                        it += 1
                        print('{:4d}\t{:1.4e}'.format(it, f.function(alphas)))
            else:
                # loop over examples where alphas are not already at their limits
                for i in np.where((alphas != 0) & (alphas != self.C))[0]:
                    examine_result, alphas, errors = self._examine_example(f, K, y, alphas, errors, i)
                    num_changed += examine_result
                    if examine_result and self.verbose:
                        it += 1
                        print('{:4d}\t{:1.4e}'.format(it, f.function(alphas)))
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

        return alphas

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
        self.intercept_ = 0.

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
        y = np.where(y == self.labels[0], -1, 1)

        n_samples = len(y)

        # kernel matrix
        K = self.kernels[self.kernel](X, X)

        P = K * np.outer(y, y)
        P = (P + P.T) / 2  # ensure P is symmetric
        q = -np.ones(n_samples)

        if self.optimizer is 'SMO':
            obj_fun = Quadratic(P, q)
            alphas = np.zeros(n_samples)
            errors = (alphas * y).dot(K) - self.intercept_ - y
            alphas = self.smo(obj_fun, K, y, alphas, errors)

        else:
            ub = np.ones(n_samples) * self.C  # upper bounds
            obj_fun = BoxConstrainedQuadratic(P, q, ub)

            if self.optimizer in (solve_qp, scipy_solve_qp):
                G = np.vstack((-np.identity(n_samples), np.identity(n_samples)))  # inequality matrix
                lb = np.zeros(n_samples)  # lower bounds
                h = np.hstack((lb, ub))  # inequality vector

                A = y.astype(np.float)  # equality matrix
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

        sv = alphas > self.tol
        self.support_ = np.arange(len(alphas))[sv]
        self.support_vectors_, self.sv_y, self.alphas = X[sv], y[sv], alphas[sv]
        self.dual_coef_ = self.alphas * self.sv_y

        if self.kernel is 'linear':
            self.coef_ = np.dot(self.dual_coef_, self.support_vectors_)

        if self.optimizer is not 'SMO':
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
        self.epsilon = epsilon
        self.intercept_ = -epsilon

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

        if self.optimizer is 'SMO':
            obj_fun = Quadratic(P, q)
            alphas = np.zeros(2 * n_samples)
            errors = (alphas[:n_samples] - alphas[n_samples:]).dot(K) - self.intercept_ - y
            alphas = self.smo(obj_fun, K, y, alphas, errors)

        else:
            ub = np.ones(2 * n_samples) * self.C  # upper bounds
            obj_fun = BoxConstrainedQuadratic(P, q, ub)

            if self.optimizer in (solve_qp, scipy_solve_qp):
                G = np.vstack((-np.identity(2 * n_samples), np.identity(2 * n_samples)))  # inequality matrix
                lb = np.zeros(2 * n_samples)  # lower bounds
                h = np.hstack((lb, ub))  # inequality vector

                A = np.hstack((np.ones(n_samples), -np.ones(n_samples)))  # equality matrix
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

        sv = np.logical_or(alphas_p > self.tol, alphas_n > self.tol)
        self.support_ = np.arange(len(alphas_p))[sv]
        self.support_vectors_, self.sv_y, self.alphas_p, self.alphas_n = X[sv], y[sv], alphas_p[sv], alphas_n[sv]
        self.dual_coef = self.alphas_p - self.alphas_n

        if self.kernel is 'linear':
            self.coef_ = np.dot(self.dual_coef, self.support_vectors_)

        if self.optimizer is not 'SMO':
            for n in range(len(self.alphas_p)):
                self.intercept_ += self.sv_y[n]
                self.intercept_ -= np.sum(self.dual_coef * K[self.support_[n], sv])
            self.intercept_ /= len(self.alphas_p)

        return self

    def predict(self, X):
        if self.kernel is not 'linear':
            return np.dot(self.dual_coef, self.kernels[self.kernel](self.support_vectors_, X)) + self.intercept_
        return np.dot(X, self.coef_) + self.intercept_
