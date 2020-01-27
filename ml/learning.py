import copy

import numpy as np

from ml.losses import MeanSquaredError, CrossEntropy
from ml.neural_network.initializers import zeros
from ml.validation import err_ratio
from optimization.optimizer import LineSearchOptimizer
from optimization.unconstrained.quasi_newton import BFGS


class Learner:
    def fit(self, X, y):
        return NotImplementedError

    def predict(self, x):
        return NotImplementedError


class LinearRegressionLearner(Learner):
    """
    Linear classifier with hard threshold.
    """

    def __init__(self, l_rate=0.01, epochs=1000, optimizer=BFGS):
        self.l_rate = l_rate
        self.epochs = epochs
        self.optimizer = optimizer

    def fit(self, X, y):
        if issubclass(self.optimizer, LineSearchOptimizer):
            self.w = self.optimizer(MeanSquaredError(X, y), zeros((X.shape[1], 1)),
                                    max_f_eval=self.epochs).minimize()[0]
        else:
            self.w = self.optimizer(MeanSquaredError(X, y), zeros((X.shape[1], 1)),
                                    step_rate=self.l_rate, max_iter=self.epochs).minimize()[0]
        return self

    def predict(self, x):
        return np.dot(x, self.w)[:, 0]


class BinaryLogisticRegressionLearner(Learner):
    """
    Linear classifier with logistic regression.
    """

    def __init__(self, l_rate=0.01, epochs=1000, optimizer=BFGS):
        self.l_rate = l_rate
        self.epochs = epochs
        self.optimizer = optimizer

    def fit(self, X, y):
        if issubclass(self.optimizer, LineSearchOptimizer):
            self.w = self.optimizer(CrossEntropy(X, y), zeros((X.shape[1], 1)),
                                    max_f_eval=self.epochs).minimize()[0]
        else:
            self.w = self.optimizer(CrossEntropy(X, y), zeros((X.shape[1], 1)),
                                    step_rate=self.l_rate, max_iter=self.epochs).minimize()[0]
        return self

    def predict_score(self, x):
        return CrossEntropy.predict(x, self.w)[:, 0]

    def predict(self, x, tol=0.5):
        return (self.predict_score(x) >= tol).astype(int)


class MultiLogisticRegressionLearner(Learner):
    def __init__(self, l_rate=0.01, epochs=1000, optimizer=BFGS, decision_function='ovr'):
        self.l_rate = l_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.decision_function = decision_function
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
                clf = BinaryLogisticRegressionLearner(self.l_rate, self.epochs, self.optimizer)
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
                    clf = BinaryLogisticRegressionLearner(self.l_rate, self.epochs, self.optimizer)
                    clf.fit(x1, y1)
                    self.classifiers.append(copy.deepcopy(clf))
        else:
            return ValueError("Decision function must be either 'ovr' or 'ovo'.")
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
        else:
            return ValueError("Decision function must be either 'ovr' or 'ovo'.")
