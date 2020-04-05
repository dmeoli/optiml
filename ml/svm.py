import matplotlib.pyplot as plt
import numpy as np
import qpsolvers
from matplotlib.lines import Line2D
from qpsolvers import solve_qp
from scipy.optimize import minimize
from sklearn.base import ClassifierMixin, BaseEstimator, RegressorMixin

from optimization.optimization_function import BoxConstrainedQuadratic, LagrangianBoxConstrained
from optimization.optimizer import BoxConstrainedOptimizer, Optimizer

plt.style.use('ggplot')


class SVM(BaseEstimator):
    def __init__(self, kernel='rbf', degree=3., gamma='scale', coef0=0., C=1.,
                 tol=1e-3, optimizer=None, epochs=1000, verbose=False):
        self.kernels = {'linear': self.linear,
                        'poly': self.poly,
                        'rbf': self.rbf,
                        'sigmoid': self.sigmoid}
        if kernel not in self.kernels.keys():
            raise ValueError(f'unknown kernel function {kernel}')
        self.kernel = kernel
        self.degree = degree
        if gamma not in ('scale', 'auto'):
            raise ValueError(f'unknown gamma type {gamma}')
        self.gamma = gamma
        self.coef0 = coef0
        self.C = C
        self.tol = tol
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose

    def linear(self, X, y):
        return np.dot(X, y.T)

    def poly(self, X, y):
        """A non-stationary kernel well suited for problems
        where all the training data is normalized"""
        gamma = 1. / (X.shape[1] * X.var()) if self.gamma is 'scale' else 1. / X.shape[1]  # auto
        return (gamma * np.dot(X, y.T) + self.coef0) ** self.degree

    def rbf(self, X, y):
        """Radial-basis function kernel (aka squared-exponential kernel)."""
        gamma = 1. / (X.shape[1] * X.var()) if self.gamma is 'scale' else 1. / X.shape[1]  # auto
        # according to: https://stats.stackexchange.com/questions/239008/rbf-kernel-algorithm-python
        return np.exp(-gamma * (np.dot(X ** 2, np.ones((X.shape[1], y.shape[0]))) +
                                np.dot(np.ones((X.shape[0], X.shape[1])), y.T ** 2) - 2. * np.dot(X, y.T)))

    def sigmoid(self, X, y):
        gamma = 1. / (X.shape[1] * X.var()) if self.gamma is 'scale' else 1. / X.shape[1]  # auto
        return np.tanh(gamma * np.dot(X, y.T) + self.coef0)

    def take_step(self, X, y, K, i1, i2):
        n_samples = len(y)

        alpha1 = self.alphas[i1]
        y1 = y[i1]
        if 0 < alpha1 < self.C:
            E1 = self.e_cache[i1]
        else:
            E1 = X[i1] * self.coef_ + self.intercept_ - y[i1]
        alpha2 = self.alphas[i2]
        y2 = y[i2]
        E2 = self.e_cache[i2]
        s = y1 * y2
        if y1 == y2:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)
        else:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        if L == H:
            return False
        eta = K[i1, i1] + K[i2, i2] - 2 * K[i1, i2]
        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            c1 = eta / 2.
            c2 = y2 * (E1 - E2) - eta * alpha2
            Lobj = c1 * L * L + c2 * L
            Hobj = c1 * H * H + c2 * H
            if Lobj > Hobj + self.tol:
                a2 = L
            elif Lobj < Hobj - self.tol:
                a2 = H
            else:
                a2 = alpha2
        if abs(a2 - alpha2) < self.tol:
            return False
        a1 = alpha1 - s * (a2 - alpha2)
        if 0 < a1 < self.C:
            bnew = self.intercept_ - E1 - y1 * (a1 - alpha1) * K[i1, i1] - y2 * (a2 - alpha2) * K[i1, i2]
        elif 0 < a2 < self.C:
            bnew = self.intercept_ - E2 - y1 * (a1 - alpha1) * K[i1, i2] - y2 * (a2 - alpha2) * K[i2, i2]
        else:
            b1 = self.intercept_ - E1 - y1 * (a1 - alpha1) * K[i1, i1] - y2 * (a2 - alpha2) * K[i1, i2]
            b2 = self.intercept_ - E2 - y1 * (a1 - alpha1) * K[i1, i2] - y2 * (a2 - alpha2) * K[i2, i2]
            bnew = (b1 + b2) / 2.0
        self.intercept_ = bnew
        self.alphas[i1] = a1
        self.alphas[i2] = a2
        self.coef_ = X.T * np.multiply(self.alphas, y)
        for i in range(n_samples):
            if 0 < self.alphas[i] < self.C:
                self.e_cache[i] = X[i] * self.coef_ + self.intercept_ - y[i]
        return True

    def examine_example(self, X, y, K, i2):
        n_samples = len(y)

        y2 = y[i2]
        alpha2 = self.alphas[i2]
        if 0 < alpha2 < self.C:
            E2 = self.e_cache[i2]
        else:
            E2 = X[i2] * self.coef_ + self.intercept_ - y[i2]
            self.e_cache[i2] = E2
        r2 = E2 * y2
        if (r2 < -self.tol and self.alphas[i2] < self.C) or (r2 > self.tol and self.alphas[i2] > 0):
            # heuristic 1: find the max deltaE
            max_delta_E = 0
            i1 = -1
            for i in range(n_samples):
                if 0 < self.alphas[i] < self.C:
                    if i == i2:
                        continue
                    E1 = self.e_cache[i]
                    delta_E = abs(E1 - E2)
                    if delta_E > max_delta_E:
                        max_delta_E = delta_E
                        i1 = i
            if i1 >= 0:
                if self.take_step(X, y, K, i1, i2):
                    return True
            # heuristic 2: find the suitable i1 on border at random
            random_index = np.random.permutation(n_samples)
            for i in random_index:
                if 0 < self.alphas[i] < self.C:
                    if i == i2:
                        continue
                    if self.take_step(X, y, K, i, i2):
                        return True
            # heuristic 3: find the suitable i1 at random on all alphas
            random_index = np.random.permutation(n_samples)
            for i in random_index:
                if i == i2:
                    continue
                if self.take_step(X, y, K, i1, i2):
                    return True
        return False

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


def scipy_solve_qp(f, G, h, A, b, max_iter, verbose):
    return minimize(fun=f.function, jac=f.jacobian, method='slsqp', x0=f.ub / 2,  # start from the middle of the box
                    constraints=({'type': 'ineq',
                                  'fun': lambda x: h - np.dot(G, x),
                                  'jac': lambda x: -G},
                                 {'type': 'eq',
                                  'fun': lambda x: np.dot(A, x) - b,
                                  'jac': lambda x: A}),
                    options={'maxiter': max_iter,
                             'disp': verbose}).x


class SVC(ClassifierMixin, SVM):
    def __init__(self, kernel='rbf', degree=3., gamma='scale', coef0=0., C=1.,
                 tol=1e-3, optimizer=None, epochs=1000, verbose=False):
        super().__init__(kernel, degree, gamma, coef0, C, tol, optimizer, epochs, verbose)

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

        if self.optimizer is None:  # default Platt's SMO algorithm
            num_changed = 0
            examine_all = True
            while num_changed > 0 or examine_all:
                num_changed = 0
                if examine_all:
                    for i in range(n_samples):
                        num_changed += self.examine_example(X, y, K, i)
                else:
                    for i in range(n_samples):
                        if 0 < self.alphas[i] < self.C:
                            num_changed += self.examine_example(X, y, K, i)
                if examine_all:
                    examine_all = False
                elif num_changed is 0:
                    examine_all = True
        else:
            P = K * np.outer(y, y)
            P = (P + P.T) / 2  # ensure P is symmetric
            q = -np.ones(n_samples)

            G = np.vstack((-np.identity(n_samples), np.identity(n_samples)))  # inequality matrix
            lb = np.zeros(n_samples)  # lower bounds
            ub = np.ones(n_samples) * self.C  # upper bounds
            h = np.hstack((lb, ub))  # inequality vector

            A = y.astype(np.float)  # equality matrix
            b = np.zeros(1)  # equality vector

            if self.optimizer is solve_qp:
                qpsolvers.cvxopt_.options['show_progress'] = self.verbose
                alphas = solve_qp(P, q, G, h, A, b, solver='cvxopt')
            else:
                obj_fun = BoxConstrainedQuadratic(P, q, ub)
                if self.optimizer is scipy_solve_qp:
                    alphas = scipy_solve_qp(obj_fun, G, h, A, b, self.epochs, self.verbose)
                elif issubclass(self.optimizer, BoxConstrainedOptimizer):
                    alphas = self.optimizer(obj_fun, max_iter=self.epochs, verbose=self.verbose).minimize()[0]
                elif issubclass(self.optimizer, Optimizer):
                    # dual lagrangian relaxation of the box-constrained problem
                    dual = LagrangianBoxConstrained(obj_fun)
                    self.optimizer(dual, max_iter=self.epochs, verbose=self.verbose).minimize()
                    alphas = dual.primal_solution
                else:
                    raise TypeError(f'unknown optimizer type {self.optimizer}')

        sv = alphas > self.tol
        self.support_ = np.arange(len(alphas))[sv]
        self.support_vectors_, self.sv_y, self.alphas = X[sv], y[sv], alphas[sv]
        self.dual_coef_ = self.alphas * self.sv_y

        if self.kernel is 'linear':
            self.coef_ = np.dot(self.dual_coef_, self.support_vectors_)

        self.intercept_ = 0
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
                 epsilon=0.1, optimizer=None, epochs=1000, verbose=False):
        super().__init__(kernel, degree, gamma, coef0, C, tol, optimizer, epochs, verbose)
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
        K = self.kernels[self.kernel](X, X)

        if self.optimizer is None:  # default Platt's SMO algorithm
            num_changed = 0
            examine_all = True
            while num_changed > 0 or examine_all:
                num_changed = 0
                if examine_all:
                    for i in range(n_samples):
                        num_changed += self.examine_example(X, y, K, i)
                else:
                    for i in range(n_samples):
                        if 0 < self.alphas[i] < self.C:
                            num_changed += self.examine_example(X, y, K, i)
                if examine_all:
                    examine_all = False
                elif num_changed is 0:
                    examine_all = True
        else:
            P = np.vstack((np.hstack((K, -K)),  # alphas_p, alphas_n
                           np.hstack((-K, K))))  # alphas_n, alphas_p
            P = (P + P.T) / 2  # ensure P is symmetric
            q = np.hstack((-y, y)) + self.epsilon

            G = np.vstack((-np.identity(2 * n_samples), np.identity(2 * n_samples)))  # inequality matrix
            lb = np.zeros(2 * n_samples)  # lower bounds
            ub = np.ones(2 * n_samples) * self.C  # upper bounds
            h = np.hstack((lb, ub))  # inequality vector

            A = np.hstack((np.ones(n_samples), -np.ones(n_samples)))  # equality matrix
            b = np.zeros(1)  # equality vector

            if self.optimizer is solve_qp:
                qpsolvers.cvxopt_.options['show_progress'] = self.verbose
                alphas = solve_qp(P, q, G, h, A, b, solver='cvxopt')
            else:
                obj_fun = BoxConstrainedQuadratic(P, q, ub)
                if self.optimizer is scipy_solve_qp:
                    alphas = scipy_solve_qp(obj_fun, G, h, A, b, self.epochs, self.verbose)
                elif issubclass(self.optimizer, BoxConstrainedOptimizer):
                    alphas = self.optimizer(obj_fun, max_iter=self.epochs, verbose=self.verbose).minimize()[0]
                elif issubclass(self.optimizer, Optimizer):
                    # dual lagrangian relaxation of the box-constrained problem
                    dual = LagrangianBoxConstrained(obj_fun)
                    self.optimizer(dual, max_iter=self.epochs, verbose=self.verbose).minimize()
                    alphas = dual.primal_solution
                else:
                    raise TypeError(f'unknown optimizer type {self.optimizer}')

        alphas_p = alphas[:n_samples]
        alphas_n = alphas[n_samples:]

        sv = np.logical_or(alphas_p > self.tol, alphas_n > self.tol)
        self.support_ = np.arange(len(alphas_p))[sv]
        self.support_vectors_, self.sv_y, self.alphas_p, self.alphas_n = X[sv], y[sv], alphas_p[sv], alphas_n[sv]
        self.dual_coef = self.alphas_p - self.alphas_n

        if self.kernel is 'linear':
            self.coef_ = np.dot(self.dual_coef, self.support_vectors_)

        self.intercept_ = 0
        for n in range(len(self.alphas_p)):
            self.intercept_ += self.sv_y[n]
            self.intercept_ -= np.sum(self.dual_coef * K[self.support_[n], sv])
        self.intercept_ -= self.epsilon
        self.intercept_ /= len(self.alphas_p)

        return self

    def predict(self, X):
        if self.kernel is not 'linear':
            return np.dot(self.dual_coef, self.kernels[self.kernel](self.support_vectors_, X)) + self.intercept_
        return np.dot(X, self.coef_) + self.intercept_
