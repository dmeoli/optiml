import matplotlib.pyplot as plt
import numpy as np
import qpsolvers
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from qpsolvers import solve_qp
from scipy.optimize import minimize
from sklearn.base import ClassifierMixin, BaseEstimator, RegressorMixin

from ml.kernels import rbf_kernel, linear_kernel, polynomial_kernel, sigmoid_kernel
from optimization.optimization_function import BoxConstrainedQuadratic, LagrangianBoxConstrained
from optimization.optimizer import BoxConstrainedOptimizer, Optimizer

plt.style.use('ggplot')


class SVM(BaseEstimator):
    def __init__(self, kernel=rbf_kernel, degree=3., gamma='scale', C=1., r=0.,
                 optimizer=solve_qp, epochs=1000, verbose=False):
        if kernel not in (linear_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel):
            raise ValueError('unknown kernel function {}'.format(kernel))
        self.kernel = kernel
        self.degree = degree
        if gamma not in ('scale', 'auto'):
            raise ValueError('unknown gamma type {}'.format(gamma))
        self.gamma = gamma
        self.C = C
        self.r = r
        self.n_sv = -1
        self.sv_idx = np.zeros(0)
        self.w = None
        self.b = 0.
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose

    def fit(self, X, y):
        raise NotImplementedError

    @staticmethod
    def plot(svm, X, y):

        ax = plt.axes()
        ax.set_axisbelow(True)
        plt.grid(color='w', linestyle='solid')

        # hide axis spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # hide top and right ticks
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()

        # lighten ticks and labels
        ax.tick_params(colors='gray', direction='out')
        for tick in ax.get_xticklabels():
            tick.set_color('gray')
        for tick in ax.get_yticklabels():
            tick.set_color('gray')

        # format axis
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))  # separate 000 with ,
        ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))  # separate 000 with ,
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 2dp for x axis.
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 2dp for y axis.
        ax.xaxis.set_tick_params(labelsize=8)  # tick label size
        ax.yaxis.set_tick_params(labelsize=8)  # tick label size

        if isinstance(svm, ClassifierMixin):
            X1, X2 = X[y == 1], X[y == -1]
        elif isinstance(svm, RegressorMixin):
            X1, X2 = X, y

        # axis limits
        x1_min, x1_max = X1.min(), X1.max()
        x2_min, x2_max = X2.min(), X2.max()
        ax.set(xlim=(x1_min, x1_max), ylim=(x2_min, x2_max))

        # axis labels
        plt.xlabel('$x_1$', fontsize=9)
        plt.ylabel('$x_2$', fontsize=9)
        plt.title('{0} SVM using {1}'.format('custom' if isinstance(svm, SVM) else 'sklearn',
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

        # decision boundary
        if isinstance(svm, ClassifierMixin):
            plt.plot(X1[:, 0], X1[:, 1], marker='x', markersize=5, color='lightblue', linestyle='none')
            plt.plot(X2[:, 0], X2[:, 1], marker='o', markersize=4, color='darkorange', linestyle='none')
        else:
            plt.plot(X1, svm.fit(X1, X2).predict(X1), label='decision boundary')

        # support vectors
        if isinstance(svm, ClassifierMixin):
            plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=60, color='blue')
        elif isinstance(svm, RegressorMixin):
            plt.scatter(X1[svm.support_], X2[svm.support_], s=60, color='blue')

        if isinstance(svm, ClassifierMixin):  # margin
            _X1, _X2 = np.meshgrid(np.linspace(x1_min, x1_max, 50), np.linspace(x1_min, x1_max, 50))
            X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(_X1), np.ravel(_X2))])
            Z = svm.decision_function(X).reshape(_X1.shape)
            plt.contour(_X1, _X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
            plt.contour(_X1, _X2, Z + 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
            plt.contour(_X1, _X2, Z - 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
        elif isinstance(svm, RegressorMixin):  # epsilon-insensitive tube
            pass

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
    def __init__(self, kernel=rbf_kernel, degree=3., gamma='scale', C=1., r=0.,
                 optimizer=solve_qp, epochs=1000, verbose=False):
        super().__init__(kernel, degree, gamma, C, r, optimizer, epochs, verbose)
        self.support_vectors_ = np.zeros(0)
        self.sv_y = np.zeros(0)
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

        m = len(y)  # m = n_samples
        K = (self.kernel(X, X, self.r, self.degree)
             if self.kernel is polynomial_kernel else
             self.kernel(X, X, self.gamma)
             if self.kernel is rbf_kernel else
             self.kernel(X, X, self.r, self.gamma)
             if self.kernel is sigmoid_kernel else
             self.kernel(X, X))  # linear kernel
        P = K * np.outer(y, y)
        P = (P + P.T) / 2  # ensure P is symmetric
        q = -np.ones(m)

        G = np.vstack((-np.identity(m), np.identity(m)))  # inequality matrix
        lb = np.zeros(m)  # lower bounds
        ub = np.ones(m) * self.C  # upper bounds
        h = np.hstack((lb, ub))  # inequality vector

        A = y.astype(np.float)  # equality matrix
        b = np.zeros(1)  # equality vector

        if self.optimizer is solve_qp:
            qpsolvers.cvxopt_.options['show_progress'] = self.verbose
            self.alphas = solve_qp(P, q, G, h, A, b, solver='cvxopt')
        else:
            obj_fun = BoxConstrainedQuadratic(P, q, ub)
            if self.optimizer is scipy_solve_qp:
                self.alphas = scipy_solve_qp(obj_fun, G, h, A, b, self.epochs, self.verbose)
            elif issubclass(self.optimizer, BoxConstrainedOptimizer):
                self.alphas = self.optimizer(obj_fun, max_iter=self.epochs, verbose=self.verbose).minimize()[0]
            elif issubclass(self.optimizer, Optimizer):
                # dual lagrangian relaxation of the box-constrained problem
                dual = LagrangianBoxConstrained(obj_fun)
                self.optimizer(dual, max_iter=self.epochs, verbose=self.verbose).minimize()
                self.alphas = dual.primal_solution
            else:
                raise TypeError('unknown optimizer type {}'.format(self.optimizer))

        self.sv_idx = np.argwhere(self.alphas > 1e-5).ravel()
        self.support_vectors_, self.sv_y, self.alphas = X[self.sv_idx], y[self.sv_idx], self.alphas[self.sv_idx]
        self.n_sv = len(self.alphas)

        if self.kernel is linear_kernel:
            self.w = np.dot(self.alphas * self.sv_y, self.support_vectors_)

        self.b = np.mean(self.sv_y - np.dot(self.alphas * self.sv_y,
                                            self.kernel(self.support_vectors_, self.support_vectors_, self.r,
                                                        self.degree)
                                            if self.kernel is polynomial_kernel else
                                            self.kernel(self.support_vectors_, self.support_vectors_, self.gamma)
                                            if self.kernel is rbf_kernel else
                                            self.kernel(self.support_vectors_, self.support_vectors_, self.r,
                                                        self.gamma)
                                            if self.kernel is sigmoid_kernel else
                                            self.kernel(self.support_vectors_, self.support_vectors_)))  # linear kernel
        return self

    def decision_function(self, X):
        """
        Predicts the score for a given example.
        """
        if self.kernel is not linear_kernel:
            return np.dot(self.alphas * self.sv_y,
                          self.kernel(self.support_vectors_, X, self.r, self.degree)
                          if self.kernel is polynomial_kernel else
                          self.kernel(self.support_vectors_, X, self.gamma)
                          if self.kernel is rbf_kernel else
                          self.kernel(self.support_vectors_, X, self.r, self.gamma)) + self.b  # sigmoid kernel
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        """
        Predicts the class of a given example.
        """
        return np.where(self.decision_function(X) >= 0, self.labels[1], self.labels[0])


class SVR(RegressorMixin, SVM):
    def __init__(self, kernel=rbf_kernel, degree=3., gamma='scale', C=1., eps=0.1, r=0.,
                 optimizer=solve_qp, epochs=1000, verbose=False):
        super().__init__(kernel, degree, gamma, C, r, optimizer, epochs, verbose)
        self.support_ = np.zeros(0)
        self.eps = eps
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

        m = len(y)  # m = n_samples
        K = (self.kernel(X, X, self.r, self.degree)
             if self.kernel is polynomial_kernel else
             self.kernel(X, X, self.gamma)
             if self.kernel is rbf_kernel else
             self.kernel(X, X, self.r, self.gamma)
             if self.kernel is sigmoid_kernel else
             self.kernel(X, X))  # linear kernel
        P = np.vstack((np.hstack((K, -K)),  # alphas_p, alphas_n
                       np.hstack((-K, K))))  # alphas_n, alphas_p
        P = (P + P.T) / 2  # ensure P is symmetric
        q = np.hstack((-y, y)) + self.eps

        G = np.vstack((-np.identity(2 * m), np.identity(2 * m)))  # inequality matrix
        lb = np.zeros(2 * m)  # lower bounds
        ub = np.ones(2 * m) * self.C  # upper bounds
        h = np.hstack((lb, ub))  # inequality vector

        A = np.hstack((np.ones(m), -np.ones(m)))  # equality matrix
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
        self.alphas_p = alphas[:m]
        self.alphas_n = alphas[m:]

        self.sv_idx = np.argwhere(alphas > 1e-5).ravel()
        self.support_, alphas = X, alphas[self.sv_idx]
        self.n_sv = len(alphas)

        if self.kernel is linear_kernel:
            self.w = np.dot(self.alphas_p - self.alphas_n, X)

        self.b = np.mean(y - self.eps - np.dot(self.alphas_p - self.alphas_n,
                                               self.kernel(self.support_, X, self.r, self.degree)
                                               if self.kernel is polynomial_kernel else
                                               self.kernel(self.support_, X, self.gamma)
                                               if self.kernel is rbf_kernel else
                                               self.kernel(self.support_, X, self.r, self.gamma)
                                               if self.kernel is sigmoid_kernel else
                                               self.kernel(self.support_, X)))  # linear kernel
        return self

    def predict(self, X):
        """
        Predicts the score of a given example.
        """
        if self.kernel is not linear_kernel:
            return np.dot(self.alphas_p - self.alphas_n,
                          self.kernel(self.support_, X, self.r, self.degree)
                          if self.kernel is polynomial_kernel else
                          self.kernel(self.support_, X, self.gamma)
                          if self.kernel is rbf_kernel else
                          self.kernel(self.support_, X, self.r, self.gamma)) + self.b  # sigmoid kernel

        return np.dot(X, self.w) + self.b
