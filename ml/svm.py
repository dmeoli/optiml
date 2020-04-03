import matplotlib.pyplot as plt
import numpy as np
import qpsolvers
from matplotlib.lines import Line2D
from qpsolvers import solve_qp
from scipy.optimize import minimize
from sklearn.base import ClassifierMixin, BaseEstimator, RegressorMixin

from ml.kernels import rbf_kernel, linear_kernel, polynomial_kernel, sigmoid_kernel
from optimization.optimization_function import BoxConstrainedQuadratic, LagrangianBoxConstrained
from optimization.optimizer import BoxConstrainedOptimizer, Optimizer

plt.style.use('ggplot')


class SVM(BaseEstimator):
    def __init__(self, kernel=rbf_kernel, degree=3., gamma='scale', C=1., coef0=0.,
                 optimizer=solve_qp, epochs=1000, verbose=False):
        if kernel not in (linear_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel):
            raise ValueError('unknown kernel function {}'.format(kernel))
        self.kernel = kernel
        self.degree = degree
        if gamma not in ('scale', 'auto'):
            raise ValueError('unknown gamma type {}'.format(gamma))
        self.gamma = gamma
        self.C = C
        # polynomial, rbf and sigmoid coef0 for sklearn compatibility
        self.coef0 = coef0
        # support vectors for sklearn compatibility
        self.support_vectors_ = np.zeros(0)
        self.sv_y = np.zeros(0)
        # support vectors indices for sklearn compatibility
        self.support_ = np.zeros(0)
        # w vector for sklearn compatibility
        self.coef_ = None
        # b value for sklearn compatibility
        self.intercept_ = 0.
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose

    @staticmethod
    def plot(svm, X, y):

        ax = plt.axes()

        # axis labels and limits
        if isinstance(svm, ClassifierMixin):
            X1, X2 = X[y == 1], X[y == -1]
            plt.xlabel('$x_1$', fontsize=9)
            plt.ylabel('$x_2$', fontsize=9)
            ax.set(xlim=(X1.min(), X1.max()), ylim=(X2.min(), X2.max()))
        elif isinstance(svm, RegressorMixin):
            plt.xlabel('X', fontsize=9)
            plt.ylabel('y', fontsize=9)

        plt.title('{} {} using {}'.format('custom' if isinstance(svm, SVM) else 'sklearn', type(svm).__name__,
                                          svm.kernel.__name__.replace('_', ' ') if callable(svm.kernel)
                                          else svm.kernel + ' kernel'), fontsize=9)

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
    def __init__(self, kernel=rbf_kernel, degree=3., gamma='scale', C=1., coef0=0.,
                 optimizer=solve_qp, epochs=1000, verbose=False):
        super().__init__(kernel, degree, gamma, C, coef0, optimizer, epochs, verbose)
        self.alphas = np.zeros(0)

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
        # gram matrix
        K = (self.kernel(X, X, self.coef0, self.degree, self.gamma)
             if self.kernel is polynomial_kernel else
             self.kernel(X, X, self.gamma)
             if self.kernel is rbf_kernel else
             self.kernel(X, X, self.coef0, self.gamma)
             if self.kernel is sigmoid_kernel else
             self.kernel(X, X))  # linear kernel
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
                raise TypeError('unknown optimizer type {}'.format(self.optimizer))

        sv = alphas > 1e-5
        self.support_ = np.arange(len(alphas))[sv]
        self.support_vectors_, self.sv_y, self.alphas = X[sv], y[sv], alphas[sv]

        if self.kernel is linear_kernel:
            self.coef_ = np.dot(self.alphas * self.sv_y, self.support_vectors_)

        for n in range(len(self.alphas)):
            self.intercept_ += self.sv_y[n]
            self.intercept_ -= np.sum(self.alphas * self.sv_y * K[self.support_[n], sv])
        self.intercept_ /= len(self.alphas)

        return self

    def decision_function(self, X):
        if self.kernel is not linear_kernel:
            return np.dot(self.alphas * self.sv_y,
                          self.kernel(self.support_vectors_, X, self.coef0, self.degree, self.gamma)
                          if self.kernel is polynomial_kernel else
                          self.kernel(self.support_vectors_, X, self.gamma)
                          if self.kernel is rbf_kernel else  # sigmoid kernel
                          self.kernel(self.support_vectors_, X, self.coef0, self.gamma)) + self.intercept_
        return np.dot(X, self.coef_) + self.intercept_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.labels[1], self.labels[0])


class SVR(RegressorMixin, SVM):
    def __init__(self, kernel=rbf_kernel, degree=3., gamma='scale', C=1., epsilon=0.1, coef0=0.,
                 optimizer=solve_qp, epochs=1000, verbose=False):
        super().__init__(kernel, degree, gamma, C, coef0, optimizer, epochs, verbose)
        self.epsilon = epsilon
        self.alphas_p = np.zeros(0)
        self.alphas_n = np.zeros(0)

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
        # gram matrix
        K = (self.kernel(X, X, self.coef0, self.degree, self.gamma)
             if self.kernel is polynomial_kernel else
             self.kernel(X, X, self.gamma)
             if self.kernel is rbf_kernel else
             self.kernel(X, X, self.coef0, self.gamma)
             if self.kernel is sigmoid_kernel else
             self.kernel(X, X))  # linear kernel
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
                raise TypeError('unknown optimizer type {}'.format(self.optimizer))
        alphas_p = alphas[:n_samples]
        alphas_n = alphas[n_samples:]

        sv = np.logical_or(alphas_p > 1e-5, alphas_n > 1e-5)
        self.support_ = np.arange(len(alphas_p))[sv]
        self.support_vectors_, self.sv_y, self.alphas_p, self.alphas_n = X[sv], y[sv], alphas_p[sv], alphas_n[sv]

        if self.kernel is linear_kernel:
            self.coef_ = np.dot(self.alphas_p - self.alphas_n, self.support_vectors_)

        for n in range(len(self.alphas_p)):
            self.intercept_ += self.sv_y[n]
            self.intercept_ -= np.sum((self.alphas_p - self.alphas_n) * K[self.support_[n], sv])
        self.intercept_ -= self.epsilon
        self.intercept_ /= len(self.alphas_p)

        return self

    def predict(self, X):
        if self.kernel is not linear_kernel:
            return np.dot(self.alphas_p - self.alphas_n,
                          self.kernel(self.support_vectors_, X, self.coef0, self.degree, self.gamma)
                          if self.kernel is polynomial_kernel else
                          self.kernel(self.support_vectors_, X, self.gamma)
                          if self.kernel is rbf_kernel else  # sigmoid kernel
                          self.kernel(self.support_vectors_, X, self.coef0, self.gamma)) + self.intercept_
        return np.dot(X, self.coef_) + self.intercept_
