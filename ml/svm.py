import numpy as np
from qpsolvers import solve_qp

from ml.kernels import rbf_kernel, linear_kernel, polynomial_kernel
from ml.learning import Learner


class SVM(Learner):
    def __init__(self, kernel=rbf_kernel, degree=3., gamma='scale', C=1., eps=0.01):
        self.kernel = kernel
        self.degree = degree
        if gamma not in ('scale', 'auto'):
            raise ValueError('unknown gamma type {}'.format(gamma))
        self.gamma = gamma
        self.C = C
        self.eps = eps
        self.n_sv = -1
        self.sv_X = np.zeros(0)
        self.w = None
        self.b = 0.


class SVC(SVM):
    def __init__(self, kernel=rbf_kernel, degree=3., gamma='scale', C=1., eps=0.01):
        super().__init__(kernel, degree, gamma, C, eps)
        self.sv_y = np.zeros(0)
        self.alphas = np.zeros(0)

    def fit(self, X, y):
        """
        Trains the model by solving a quadratic programming problem.
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
        self.alphas = solve_qp(Q, q, lb, ub, Aeq, beq, solver='cvxopt', sym_proj=True)  # Lagrange multipliers

        sv_idx = list(filter(lambda i: self.alphas[i] > self.eps, range(len(y))))
        self.sv_X, self.sv_y, self.alphas = X[sv_idx], y[sv_idx], self.alphas[sv_idx]
        self.n_sv = len(sv_idx)
        if self.kernel == linear_kernel:
            self.w = np.dot(self.alphas * self.sv_y, self.sv_X)

        sv_boundary = self.alphas < self.C - self.eps
        self.b = np.mean(self.sv_y[sv_boundary] - np.dot(self.alphas * self.sv_y,
                                                         self.kernel(self.sv_X, self.sv_X[sv_boundary], self.degree)
                                                         if self.kernel is polynomial_kernel else
                                                         self.kernel(self.sv_X, self.sv_X[sv_boundary], self.gamma)
                                                         if self.kernel is rbf_kernel else  # linear kernel
                                                         self.kernel(self.sv_X, self.sv_X[sv_boundary])))
        return self

    def predict_score(self, X):
        """
        Predicts the score for a given example.
        """
        if self.w is None:
            return np.dot(self.alphas * self.sv_y,
                          self.kernel(self.sv_X, X, self.degree)
                          if self.kernel is polynomial_kernel else
                          self.kernel(self.sv_X, X, self.gamma)
                          if self.kernel is rbf_kernel else
                          self.kernel(self.sv_X, X)) + self.b  # linear kernel
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        """
        Predicts the class of a given example.
        """
        return np.sign(self.predict_score(X))


class SVR(SVM):
    def __init__(self, kernel=rbf_kernel, degree=3., gamma='scale', C=1., eps=0.01):
        super().__init__(kernel, degree, gamma, C, eps)
        self.alphas_p = np.zeros(0)
        self.alphas_n = np.zeros(0)

    def fit(self, X, y):
        """
        Trains the model by solving a quadratic programming problem.
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
        Q = np.vstack((np.hstack((H, -H)),  # a_p, a_n
                       np.hstack((-H, H))))  # a_n, a_p
        q = np.hstack((-y, y)) + self.eps  # linear part
        lb = np.vstack((-np.identity(2 * m), np.identity(2 * m)))  # lower bounds
        ub = np.hstack((np.zeros(2 * m), np.zeros(2 * m) + self.C))  # upper bounds
        Aeq = np.hstack((np.ones(m), -np.ones(m))).reshape((1, -1))
        beq = np.zeros(1)
        alphas = solve_qp(Q, q, lb, ub, Aeq, beq, solver='cvxopt', sym_proj=True)  # Lagrange multipliers
        self.alphas_p = alphas[:m]
        self.alphas_n = alphas[m:]

        sv_idx = list(filter(lambda i: alphas[i] > self.eps, range(len(y))))
        self.sv_X = X
        self.n_sv = len(sv_idx)
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
                          self.kernel(self.sv_X, X, self.degree)
                          if self.kernel is polynomial_kernel else
                          self.kernel(self.sv_X, X, self.gamma)
                          if self.kernel is rbf_kernel else
                          self.kernel(self.sv_X, X)) + self.b  # linear kernel
        return np.dot(X, self.w) + self.b
