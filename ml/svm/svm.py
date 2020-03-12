import numpy as np
from qpsolvers import solve_qp
from scipy.optimize import minimize, lsq_linear

from ml.learning import Learner
from ml.svm.kernels import rbf_kernel, linear_kernel, polynomial_kernel
from optimization.constrained.projected_gradient import ConstrainedOptimizer
from optimization.optimization_function import Quadratic


def scipy_solve_qp(f, y, G, h):
    return minimize(fun=f.function, jac=f.jacobian, x0=np.random.rand(f.n),
                    constraints=({'type': 'ineq', 'fun': lambda x: h - np.dot(G, x), 'jac': lambda x: -G},
                                 {'type': 'eq', 'fun': lambda x: np.dot(x, y), 'jac': lambda x: y})).x


class SVM(Learner):
    def __init__(self, kernel=rbf_kernel, degree=3., gamma='scale', C=1., eps=0.1):
        self.kernel = kernel
        self.degree = degree
        if gamma not in ('scale', 'auto'):
            raise ValueError('unknown gamma type {}'.format(gamma))
        self.gamma = gamma
        self.C = C
        self.eps = eps
        self.n_sv = -1
        self.sv = np.zeros(0)
        self.w = None
        self.b = 0.

    def fit(self, X, y, optimizer=None, max_iter=1000):
        raise NotImplementedError


class SVC(SVM):
    def __init__(self, kernel=rbf_kernel, degree=3., gamma='scale', C=1., eps=0.1):
        super().__init__(kernel, degree, gamma, C, eps)
        self.sv_y = np.zeros(0)
        self.alphas = np.zeros(0)

    def fit(self, X, y, optimizer, max_iter=1000):
        """
        Trains the model by solving a constrained quadratic programming problem.
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples] holding the class labels
        """
        m = len(y)  # m = n_samples
        K = (self.kernel(X, X, self.degree)
             if self.kernel is polynomial_kernel else
             self.kernel(X, X, self.gamma)
             if self.kernel is rbf_kernel else
             self.kernel(X, X))  # linear kernel
        P = K * np.outer(y, y)  # quadratic part
        q = -np.ones(m)  # linear part
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
        lagrangian = Quadratic(P, np.ones_like(y))

        self.alphas = (scipy_solve_qp(lagrangian, y, G, h) if optimizer is scipy_solve_qp else
                       solve_qp(P, q, G, h, A, b, solver='cvxopt', sym_proj=True) if optimizer is solve_qp else
                       optimizer(lagrangian, max_iter).minimize(ub)[0])

        sv_idx = np.arange(len(self.alphas))[self.alphas > self.eps]
        self.sv, self.sv_y, self.alphas = X[sv_idx], y[sv_idx], self.alphas[sv_idx]
        self.n_sv = len(self.alphas)
        if self.kernel == linear_kernel:
            self.w = np.dot(self.alphas * self.sv_y, self.sv)

        self.b = np.mean(self.sv_y - np.dot(self.alphas * self.sv_y,
                                            self.kernel(self.sv, self.sv, self.degree)
                                            if self.kernel is polynomial_kernel else
                                            self.kernel(self.sv, self.sv, self.gamma)
                                            if self.kernel is rbf_kernel else  # linear kernel
                                            self.kernel(self.sv, self.sv)))
        return self

    def predict_score(self, X):
        """
        Predicts the score for a given example.
        """
        if self.w is None:
            return np.dot(self.alphas * self.sv_y,
                          self.kernel(self.sv, X, self.degree)
                          if self.kernel is polynomial_kernel else
                          self.kernel(self.sv, X, self.gamma)
                          if self.kernel is rbf_kernel else
                          self.kernel(self.sv, X)) + self.b  # linear kernel
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        """
        Predicts the class of a given example.
        """
        return np.sign(self.predict_score(X))


class SVR(SVM):
    def __init__(self, kernel=rbf_kernel, degree=3., gamma='scale', C=1., eps=0.1):
        super().__init__(kernel, degree, gamma, C, eps)
        self.alphas_p = np.zeros(0)
        self.alphas_n = np.zeros(0)

    def fit(self, X, y, optimizer=None, max_iter=1000):
        """
        Trains the model by solving a constrained quadratic programming problem.
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples] holding the class labels
        """
        m = len(y)  # m = n_samples
        K = (self.kernel(X, X, self.degree)
             if self.kernel is polynomial_kernel else
             self.kernel(X, X, self.gamma)
             if self.kernel is rbf_kernel else
             self.kernel(X, X))  # linear kernel
        # quadratic part
        P = np.vstack((np.hstack((K, -K)),  # alphas_p, alphas_n
                       np.hstack((-K, K))))  # alphas_n, alphas_p
        q = np.hstack((-y, y)) + self.eps  # linear part
        G = np.vstack((-np.identity(2 * m), np.identity(2 * m)))  # inequality matrix
        lb = np.zeros(2 * m)  # lower bounds
        ub = np.ones(2 * m) * self.C  # upper bounds
        h = np.hstack((lb, ub))  # inequality vector
        A = np.hstack((np.ones(m), -np.ones(m)))  # equality matrix
        b = np.zeros(1)  # equality vector

        # we'd like to minimize the negative of the Lagrangian dual function subject to linear constraints:
        # inequalities Gx <= h (A is m x n, where m = 2n is the number of inequalities
        # (n box constraints, 2 inequalities each)
        # equalities Ax = b (these's only one equality constraint, i.e. y.T.dot(x) = 0)
        lagrangian = Quadratic(P, np.ones_like(y))

        alphas = (scipy_solve_qp(lagrangian, y, G, h) if optimizer is scipy_solve_qp else
                  solve_qp(P, q, G, h, A, b, solver='cvxopt', sym_proj=True) if optimizer is solve_qp else
                  optimizer(lagrangian, max_iter).minimize(ub)[0])
        self.alphas_p = alphas[:m]
        self.alphas_n = alphas[m:]

        self.sv = X
        self.n_sv = len(alphas)
        if self.kernel == linear_kernel:
            self.w = np.dot(self.alphas_p - self.alphas_n, X)

        self.b = np.mean(y - self.eps - np.dot(self.alphas_p - self.alphas_n,
                                               self.kernel(X, X, self.degree)
                                               if self.kernel is polynomial_kernel else
                                               self.kernel(X, X, self.gamma)
                                               if self.kernel is rbf_kernel else  # linear kernel
                                               self.kernel(X, X)))
        return self

    def predict(self, X):
        """
        Predicts the score of a given example.
        """
        if self.w is None:
            return np.dot(self.alphas_p - self.alphas_n,
                          self.kernel(self.sv, X, self.degree)
                          if self.kernel is polynomial_kernel else
                          self.kernel(self.sv, X, self.gamma)
                          if self.kernel is rbf_kernel else
                          self.kernel(self.sv, X)) + self.b  # linear kernel
        return np.dot(X, self.w) + self.b
