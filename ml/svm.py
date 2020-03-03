import numpy as np
from qpsolvers import solve_qp

from ml.kernels import rbf_kernel, linear_kernel, polynomial_kernel
from ml.learning import Learner


class SVM(Learner):
    def __init__(self, kernel=rbf_kernel, degree=3., gamma='scale', C=1.):
        self.kernel = kernel
        self.degree = degree
        if gamma not in ('scale', 'auto'):
            raise ValueError('unknown gamma type {}'.format(gamma))
        self.gamma = gamma
        self.C = C  # hyper-parameter
        self.eps = 1e-6
        self.n_sv = -1
        self.sv_x, self.sv_y, = np.zeros(0), np.zeros(0)
        self.alphas = np.zeros(0)
        self.w = None
        self.b = 0.  # intercept

    def fit(self, X, y):
        """
        Trains the model by solving a quadratic programming problem.
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples] holding the class labels
        """
        # In QP formulation (dual): m variables, 2m+1 constraints (1 equation, 2m inequations)
        self.QP(X, y)
        sv_indices = list(filter(lambda i: self.alphas[i] > self.eps, range(len(y))))
        self.sv_x, self.sv_y, self.alphas = X[sv_indices], y[sv_indices], self.alphas[sv_indices]
        self.n_sv = len(sv_indices)
        if self.kernel == linear_kernel:
            self.w = np.dot(self.alphas * self.sv_y, self.sv_x)
        # calculate b: average over all support vectors
        sv_boundary = self.alphas < self.C - self.eps
        self.b = np.mean(self.sv_y[sv_boundary] - np.dot(self.alphas * self.sv_y,
                                                         self.kernel(self.sv_x, self.sv_x[sv_boundary], self.degree)
                                                         if self.kernel is polynomial_kernel else
                                                         self.kernel(self.sv_x, self.sv_x[sv_boundary], self.gamma)
                                                         if self.kernel is rbf_kernel else  # linear kernel
                                                         self.kernel(self.sv_x, self.sv_x[sv_boundary])))
        return self

    def QP(self, X, y):
        """
        Solves a quadratic programming problem. In QP formulation (dual):
        m variables, 2m+1 constraints (1 equation, 2m inequations).
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples] holding the class labels
        """
        #
        m = len(y)  # m = n_samples
        K = (self.kernel(X, degree=self.degree)
             if self.kernel is polynomial_kernel else
             self.kernel(X, gamma=self.gamma)
             if self.kernel is rbf_kernel else
             self.kernel(X))  # linear kernel
        P = K * np.outer(y, y)
        q = -np.ones(m)
        G = np.vstack((-np.identity(m), np.identity(m)))
        h = np.hstack((np.zeros(m), np.ones(m) * self.C))
        A = y.reshape((1, -1))
        b = np.zeros(1)
        # make sure P is positive definite
        P += np.eye(P.shape[0]).__mul__(1e-3)
        self.alphas = solve_qp(P, q, G, h, A, b, sym_proj=True)

    def predict_score(self, X):
        """
        Predicts the score for a given example.
        """
        if self.w is None:
            return np.dot(self.alphas * self.sv_y,
                          self.kernel(self.sv_x, X, self.degree)
                          if self.kernel is polynomial_kernel else
                          self.kernel(self.sv_x, X, self.gamma)
                          if self.kernel is rbf_kernel else
                          self.kernel(self.sv_x, X, self.degree)) + self.b  # linear kernel
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        """
        Predicts the class of a given example.
        """
        return np.sign(self.predict_score(X))
