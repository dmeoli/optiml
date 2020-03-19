import matplotlib.pyplot as plt
import numpy as np
import qpsolvers
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from qpsolvers import solve_qp
from scipy.optimize import minimize

from ml.learning import Learner
from ml.svm.kernels import rbf_kernel, linear_kernel, polynomial_kernel, sigmoid_kernel
from optimization.optimization_function import BoxConstrainedQuadratic

plt.style.use('ggplot')


class SVM(Learner):
    def __init__(self, kernel=rbf_kernel, degree=3., gamma='scale', C=1., r=0.):
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
        self.sv = np.zeros(0)
        self.w = None
        self.b = 0.

    def fit(self, X, y, optimizer=solve_qp, max_iter=1000):
        raise NotImplementedError

    @staticmethod
    def plot(svm, X1, X2):
        from sklearn.svm import SVC as SKLSVC
        from sklearn.svm import SVR as SKLSVR

        def f(x, w, b, c=0):
            return (-w[0] * x - b + c) / w[1]

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
        # axis limits
        x1_min, x1_max = X1.min(), X1.max()
        x2_min, x2_max = X2.min(), X2.max()
        ax.set(xlim=(x1_min, x1_max), ylim=(x2_min, x2_max))
        # axis labels
        plt.xlabel('$x_1$', fontsize=9)
        plt.ylabel('$x_2$', fontsize=9)
        plt.title('{0} SVM using {1}'.format('sklearn' if isinstance(svm, SKLSVC) or isinstance(svm, SKLSVR)
                                             else 'custom', svm.kernel.__name__ if callable(svm.kernel)
                                             else svm.kernel + ' kernel'), fontsize=9)
        # set the legend
        legend = [
            Line2D([0], [0], linestyle='none', marker='x', color='lightblue', markerfacecolor='lightblue',
                   markersize=9),
            Line2D([0], [0], linestyle='none', marker='o', color='darkorange', markerfacecolor='darkorange',
                   markersize=9),
            Line2D([0], [0], linestyle='-', marker='.', color='black', markerfacecolor='darkorange', markersize=0),
            Line2D([0], [0], linestyle='--', marker='.', color='black', markerfacecolor='darkorange', markersize=0),
            Line2D([0], [0], linestyle='none', marker='.', color='blue', markerfacecolor='blue', markersize=9)]

        if 'linear' not in str(svm.kernel):
            # place the legend in a nicer position
            legend = plt.legend(legend,
                                ['negative -1', 'positive +1', 'decision boundary', 'margin', 'support vectors'],
                                fontsize='7', shadow=True, loc='lower left', bbox_to_anchor=(0.03, 0.03))
        else:
            legend = plt.legend(legend,
                                ['negative -1', 'positive +1', 'decision boundary', 'margin', 'support vectors'],
                                fontsize='7', shadow=True, bbox_to_anchor=(0.3, 0.98))
        legend.get_frame().set_linewidth(0.3)

        # add the plots
        if isinstance(svm, SVC) or isinstance(svm, SKLSVC):
            plt.plot(X1[:, 0], X1[:, 1], marker='x', markersize=5, color='lightblue', linestyle='none')
            plt.plot(X2[:, 0], X2[:, 1], marker='o', markersize=4, color='darkorange', linestyle='none')
        else:
            plt.plot(X1, svm.fit(X1, X2).predict(X1), label='decision boundary')
        # the points designating the support vectors
        if isinstance(svm, SKLSVC):
            plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=60, color='blue')
        elif isinstance(svm, SKLSVR):
            plt.scatter(X1[svm.support_], X2[svm.support_], s=60, color='blue')
        else:
            plt.scatter(svm.sv[:, 0], svm.sv[:, 1], s=60, color='blue')

        if (isinstance(svm, SKLSVC) or isinstance(svm, SKLSVR)
                or 'polynomial' in str(svm.kernel) or 'rbf' in str(svm.kernel)):
            # non-linear margin line needs to be generated
            _X1, _X2 = np.meshgrid(np.linspace(x1_min, x1_max, 50), np.linspace(x1_min, x1_max, 50))
            X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(_X1), np.ravel(_X2))])

            if isinstance(svm, SKLSVC):
                Z = svm.decision_function(X).reshape(_X1.shape)
            elif isinstance(svm, SKLSVR):
                Z = svm.predict(X)
            else:
                Z = svm.predict_score(X).reshape(_X1.shape)

            plt.contour(_X1, _X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
            plt.contour(_X1, _X2, Z + 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
            plt.contour(_X1, _X2, Z - 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
        else:
            # linear margin line needs to be generated
            # this can be done by the above code and use plt.contour
            # decision Boundary:  w.x + b = 0
            _y1 = f(x1_min, svm.w, svm.b)
            _y2 = f(x1_max, svm.w, svm.b)
            plt.plot([x1_min, x1_max], [_y1, _y2], 'k')

            # margin Upper: w.x + b = 1
            _y3 = f(x1_min, svm.w, svm.b, 1)
            _y4 = f(x1_max, svm.w, svm.b, 1)
            plt.plot([x1_min, x1_max], [_y3, _y4], 'k--')

            # margin Lower: w.x + b = -1
            _y5 = f(x1_min, svm.w, svm.b, -1)
            _y6 = f(x1_max, svm.w, svm.b, -1)
            plt.plot([x1_min, x1_max], [_y5, _y6], 'k--')

        plt.show()


def scipy_solve_qp(f, G, h, max_iter, verbose):
    return minimize(fun=f.function, jac=f.jacobian, x0=np.random.rand(f.n),
                    constraints=({'type': 'ineq',
                                  'fun': lambda x: h - np.dot(G, x),
                                  'jac': lambda x: -G}),
                    options={'maxiter': max_iter,
                             'disp': verbose}).x


class SVMLagrangianRelaxation(BoxConstrainedQuadratic):
    def __init__(self, Q, q, ub, A, b):
        """
        Construct the Lagrangian relaxation of the SVC learning problem with equality constraints A x = b
        :param Q: ([n x n] real symmetric matrix, not necessarily positive semidefinite):
                           the Hessian (i.e. the quadratic part) of f. If it is not
                           positive semidefinite, f(x) will be unbounded below.
        :param q: ([n x 1] real column vector): the linear part of f.
        :param A: equality constraints matrix
        :param b: equality constraints vector
        """
        super().__init__(Q, q, ub)
        self.A = A
        self.b = b

    def function(self, x, Q=None, q=None):
        return super().function(x, Q, q) + self.A.dot(x) - self.b

    def jacobian(self, x, Q=None, q=None):
        return super().jacobian(x, Q, q) + self.A


class SVC(SVM):
    def __init__(self, kernel=rbf_kernel, degree=3., gamma='scale', C=1., r=0.):
        super().__init__(kernel, degree, gamma, C, r)
        self.sv_y = np.zeros(0)
        self.alphas = np.zeros(0)

    def fit(self, X, y, optimizer=solve_qp, max_iter=1000, verbose=False):
        """
        Trains the model by solving a constrained quadratic programming problem.
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples] holding the class labels
        :param optimizer:
        :param max_iter:
        :param verbose:
        """
        self.labels = np.unique(y)
        if self.labels.size > 2:
            raise ValueError('use MultiClassClassifier to train a model over more than two labels')
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

        # we'd like to minimize the negative of the Lagrangian dual function subject to linear constraints:
        # inequalities Gx <= h (A is m x n, where m = 2n is the number of inequalities
        # (n box constraints, 2 inequalities each)
        # equalities Ax = b (these's only one equality constraint, i.e. y.T.dot(x) = 0)
        obj = SVMLagrangianRelaxation(P, np.ones_like(q), ub, A, b.item())
        if optimizer is solve_qp:
            qpsolvers.cvxopt_.options['show_progress'] = verbose
        self.alphas = (scipy_solve_qp(obj, G, h, max_iter, verbose) if optimizer is scipy_solve_qp else
                       solve_qp(P, q, G, h, A, b, solver='cvxopt') if optimizer is solve_qp else
                       optimizer(obj, max_iter=max_iter, verbose=verbose).minimize()[0])

        self.sv_idx = np.argwhere(self.alphas > 1e-5).ravel()
        self.sv, self.sv_y, self.alphas = X[self.sv_idx], y[self.sv_idx], self.alphas[self.sv_idx]
        self.n_sv = len(self.alphas)

        if self.kernel is linear_kernel:
            self.w = np.dot(self.alphas * self.sv_y, self.sv)

        self.b = np.mean(self.sv_y - np.dot(self.alphas * self.sv_y,
                                            self.kernel(self.sv, self.sv, self.r, self.degree)
                                            if self.kernel is polynomial_kernel else
                                            self.kernel(self.sv, self.sv, self.gamma)
                                            if self.kernel is rbf_kernel else
                                            self.kernel(self.sv, self.sv, self.r, self.gamma)
                                            if self.kernel is sigmoid_kernel else
                                            self.kernel(self.sv, self.sv)))  # linear kernel
        return self

    def predict_score(self, X):
        """
        Predicts the score for a given example.
        """
        if self.kernel is not linear_kernel:
            return np.dot(self.alphas * self.sv_y,
                          self.kernel(self.sv, X, self.r, self.degree)
                          if self.kernel is polynomial_kernel else
                          self.kernel(self.sv, X, self.gamma)
                          if self.kernel is rbf_kernel else
                          self.kernel(self.sv, X, self.r, self.gamma)) + self.b  # sigmoid kernel
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        """
        Predicts the class of a given example.
        """
        return np.where(self.predict_score(X) >= 0, self.labels[1], self.labels[0])


class SVR(SVM):
    def __init__(self, kernel=rbf_kernel, degree=3., gamma='scale', C=1., eps=0.1, r=0.):
        super().__init__(kernel, degree, gamma, C, r)
        self.eps = eps
        self.alphas_p = np.zeros(0)
        self.alphas_n = np.zeros(0)

    def fit(self, X, y, optimizer=solve_qp, max_iter=1000, verbose=False):
        """
        Trains the model by solving a constrained quadratic programming problem.
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples] holding the class labels
        :param optimizer:
        :param max_iter:
        :param verbose:
        """
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

        # we'd like to minimize the negative of the Lagrangian dual function subject to linear constraints:
        # inequalities Gx <= h (A is m x n, where m = 2n is the number of inequalities
        # (n box constraints, 2 inequalities each)
        # equalities Ax = b (these's only one equality constraint, i.e. x.T.dot(x) = 0)
        obj = SVMLagrangianRelaxation(P, np.ones_like(q), ub, A, b.item())
        if optimizer is solve_qp:
            qpsolvers.cvxopt_.options['show_progress'] = verbose
        alphas = (scipy_solve_qp(obj, G, h, max_iter, verbose) if optimizer is scipy_solve_qp else
                  solve_qp(P, q, G, h, A, b, solver='cvxopt') if optimizer is solve_qp else
                  optimizer(obj, max_iter=max_iter, verbose=verbose).minimize()[0])
        self.alphas_p = alphas[:m]
        self.alphas_n = alphas[m:]

        self.sv_idx = np.argwhere(alphas > 1e-5).ravel()
        self.sv, alphas = X, alphas[self.sv_idx]
        self.n_sv = len(alphas)

        if self.kernel is linear_kernel:
            self.w = np.dot(self.alphas_p - self.alphas_n, X)

        self.b = np.mean(y - self.eps - np.dot(self.alphas_p - self.alphas_n,
                                               self.kernel(self.sv, X, self.r, self.degree)
                                               if self.kernel is polynomial_kernel else
                                               self.kernel(self.sv, X, self.gamma)
                                               if self.kernel is rbf_kernel else
                                               self.kernel(self.sv, X, self.r, self.gamma)
                                               if self.kernel is sigmoid_kernel else
                                               self.kernel(self.sv, X)))  # linear kernel
        return self

    def predict(self, X):
        """
        Predicts the score of a given example.
        """
        if self.kernel is not linear_kernel:
            return np.dot(self.alphas_p - self.alphas_n,
                          self.kernel(self.sv, X, self.r, self.degree)
                          if self.kernel is polynomial_kernel else
                          self.kernel(self.sv, X, self.gamma)
                          if self.kernel is rbf_kernel else
                          self.kernel(self.sv, X, self.r, self.gamma)) + self.b  # sigmoid kernel

        return np.dot(X, self.w) + self.b
