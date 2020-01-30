import copy

from qpsolvers import solve_qp

import numpy as np

from ml.learning import Learner


def linear_kernel(x, y=None):
    if y is None:
        y = x
    return np.dot(x, y.T)


def polynomial_kernel(x, y=None, degree=2.0):
    if y is None:
        y = x
    return (1.0 + np.dot(x, y.T)) ** degree


def rbf_kernel(x, y=None, gamma=None):
    """Radial-basis function kernel (aka squared-exponential kernel)."""
    if y is None:
        y = x
    if gamma is None:
        gamma = 1.0 / x.shape[1]  # 1.0 / n_features
    return np.exp(-gamma * (-2.0 * np.dot(x, y.T) +
                            np.sum(x * x, axis=1).reshape((-1, 1)) + np.sum(y * y, axis=1).reshape((1, -1))))


class BinarySVM(Learner):
    def __init__(self, kernel=linear_kernel, C=1.0):
        self.kernel = kernel
        self.C = C  # hyper-parameter
        self.eps = 1e-6
        self.n_sv = -1
        self.sv_x, self.sv_y, = np.zeros(0), np.zeros(0)
        self.alphas = np.zeros(0)
        self.w = None
        self.b = 0.0  # intercept

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
        K = self.kernel(X)  # gram matrix
        P = K * np.outer(y, y)
        q = -np.ones(m)
        G = np.vstack((-np.identity(m), np.identity(m)))
        h = np.hstack((np.zeros(m), np.ones(m) * self.C))
        A = y.reshape((1, -1))
        b = np.zeros(1)
        # make sure P is positive definite
        P += np.eye(P.shape[0]).__mul__(1e-3)
        self.alphas = solve_qp(P, q, G, h, A, b, sym_proj=True)

    def predict_score(self, x):
        """
        Predicts the score for a given example.
        """
        if self.w is None:
            return np.dot(self.alphas * self.sv_y, self.kernel(self.sv_x, x)) + self.b
        return np.dot(x, self.w) + self.b

    def predict(self, x):
        """
        Predicts the class of a given example.
        """
        return np.sign(self.predict_score(x))


class MultiSVM(Learner):
    def __init__(self, kernel=linear_kernel, decision_function='ovr', C=1.0):
        self.kernel = kernel
        if decision_function not in ('ovr', 'ovo'):
            raise ValueError("decision function must be either 'ovr' or 'ovo'")
        self.decision_function = decision_function
        self.C = C  # hyper-parameter
        self.n_class, self.classifiers = 0, []

    def fit(self, X, y):
        """
        Trains n_class or n_class * (n_class - 1) / 2 classifiers
        according to the training method, ovr or ovo respectively.
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples] holding the class labels
        :return: array of classifiers
        """
        labels = np.unique(y)
        self.n_class = len(labels)
        if self.decision_function == 'ovr':  # one-vs-rest method
            for label in labels:
                y1 = np.array(y)
                y1[y1 != label] = -1.0
                y1[y1 == label] = 1.0
                clf = BinarySVM(self.kernel, self.C)
                clf.fit(X, y1)
                self.classifiers.append(copy.deepcopy(clf))
        elif self.decision_function == 'ovo':  # use one-vs-one method
            n_labels = len(labels)
            for i in range(n_labels):
                for j in range(i + 1, n_labels):
                    neg_id, pos_id = y == labels[i], y == labels[j]
                    x1, y1 = np.r_[X[neg_id], X[pos_id]], np.r_[y[neg_id], y[pos_id]]
                    y1[y1 == labels[i]] = -1.0
                    y1[y1 == labels[j]] = 1.0
                    clf = BinarySVM(self.kernel, self.C)
                    clf.fit(x1, y1)
                    self.classifiers.append(copy.deepcopy(clf))
        return self

    def predict(self, x):
        """
        Predicts the class of a given example according to the training method.
        """
        n_samples = len(x)
        if self.decision_function == 'ovr':  # one-vs-rest method
            assert len(self.classifiers) == self.n_class
            score = np.zeros((n_samples, self.n_class))
            for i in range(self.n_class):
                clf = self.classifiers[i]
                score[:, i] = clf.predict_score(x)
            return np.argmax(score, axis=1)
        elif self.decision_function == 'ovo':  # use one-vs-one method
            assert len(self.classifiers) == self.n_class * (self.n_class - 1) / 2
            vote = np.zeros((n_samples, self.n_class))
            clf_id = 0
            for i in range(self.n_class):
                for j in range(i + 1, self.n_class):
                    res = self.classifiers[clf_id].predict(x)
                    vote[res < 0, i] += 1.0  # negative sample: class i
                    vote[res > 0, j] += 1.0  # positive sample: class j
                    clf_id += 1
            return np.argmax(vote, axis=1)
