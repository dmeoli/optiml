import numpy as np
from qpsolvers import solve_qp

from ml.svm.kernels import rbf_kernel, linear_kernel, polynomial_kernel
from ml.learning import Learner
from optimization.constrained.projected_gradient import ProjectedGradient
from optimization.optimization_function import Quadratic


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

    def fit(self, X, y, optimizer=None, max_iter=1000):
        """
        Trains the model by solving a constrained quadratic programming problem.
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples] holding the class labels
        """
        m = len(y)  # m = n_samples
        H = (self.kernel(X, X, self.degree)
             if self.kernel is polynomial_kernel else
             self.kernel(X, X, self.gamma)
             if self.kernel is rbf_kernel else
             self.kernel(X, X))  # linear kernel
        Q = H * np.outer(y, y)  # quadratic part
        q = -np.ones(m)  # linear part
        lb = np.vstack((-np.identity(m), np.identity(m)))  # lower bounds
        ub = np.hstack((np.zeros(m), np.zeros(m) + self.C))  # upper bounds
        Aeq = y.astype(np.float).reshape((1, -1))
        beq = np.zeros(1)
        self.alphas = (optimizer(Quadratic(Q, q), max_iter).minimize(ub) if optimizer else  # box constrained quadratic
                       solve_qp(Q, q, lb, ub, Aeq, beq, solver='cvxopt', sym_proj=True))  # Lagrange multipliers

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
        H = (self.kernel(X, X, self.degree)
             if self.kernel is polynomial_kernel else
             self.kernel(X, X, self.gamma)
             if self.kernel is rbf_kernel else
             self.kernel(X, X))  # linear kernel
        # quadratic part
        Q = np.vstack((np.hstack((H, -H)),  # alphas_p, alphas_n
                       np.hstack((-H, H))))  # alphas_n, alphas_p
        q = np.hstack((-y, y)) + self.eps  # linear part
        lb = np.vstack((-np.identity(2 * m), np.identity(2 * m)))  # lower bounds
        ub = np.hstack((np.zeros(2 * m), np.zeros(2 * m) + self.C))  # upper bounds
        Aeq = np.hstack((np.ones(m), -np.ones(m))).reshape((1, -1))
        beq = np.zeros(1)
        alphas = (optimizer(Quadratic(Q, q), max_iter).minimize(ub) if optimizer else  # box constrained quadratic
                  solve_qp(Q, q, lb, ub, Aeq, beq, solver='cvxopt', sym_proj=True))  # Lagrange multipliers
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
