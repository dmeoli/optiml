import copy
import itertools

import numpy as np

import utils
from ml.losses import MeanSquaredError, CrossEntropy
from ml.neural_network.initializers import zeros
from optimization.optimizer import LineSearchOptimizer
from optimization.unconstrained.gradient_descent import GD


class Learner:
    def fit(self, X, y):
        return NotImplementedError

    def predict(self, x):
        return NotImplementedError


class LinearRegressionLearner(Learner):

    def __init__(self, l_rate=0.01, epochs=1000, batch_size=None, optimizer=GD,
                 regularization_type='l1', lmbda=0.1, alpha=0.2):
        self.l_rate = l_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.regularization_type = regularization_type
        self.lmbda = lmbda
        self.alpha = alpha

    def fit(self, X, y):
        loss_function = MeanSquaredError(self.regularization_type, self.lmbda, self.alpha)
        args = itertools.repeat(([X, y[:, np.newaxis]], {})) if self.batch_size is None \
            else ((i, {}) for i in utils.iter_mini_batches([X, y[:, np.newaxis]], self.batch_size))
        if issubclass(self.optimizer, LineSearchOptimizer):
            self.w = self.optimizer(loss_function, zeros((X.shape[1], 1)), max_f_eval=self.epochs,
                                    args=args).minimize()[0][:, 0]
        else:
            self.w = self.optimizer(loss_function, zeros((X.shape[1], 1)), step_rate=self.l_rate,
                                    max_iter=self.epochs, args=args).minimize()[0][:, 0]
        return self

    def predict(self, x):
        return MeanSquaredError.predict(x, self.w)


class BinaryLogisticRegressionLearner(Learner):

    def __init__(self, l_rate=0.01, epochs=1000, batch_size=None, optimizer=GD,
                 regularization_type='l2', lmbda=0.1, alpha=0.2):
        self.l_rate = l_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.regularization_type = regularization_type
        self.lmbda = lmbda
        self.alpha = alpha

    def fit(self, X, y):
        self.labels = np.unique(y)
        y = np.where(y == self.labels[0], 0, 1)
        loss_function = CrossEntropy(self.regularization_type, self.lmbda, self.alpha)
        args = itertools.repeat(([X, y[:, np.newaxis]], {})) if self.batch_size is None \
            else ((i, {}) for i in utils.iter_mini_batches([X, y[:, np.newaxis]], self.batch_size))
        if issubclass(self.optimizer, LineSearchOptimizer):
            self.w = self.optimizer(loss_function, zeros((X.shape[1], 1)), max_f_eval=self.epochs,
                                    args=args).minimize()[0][:, 0]
        else:
            self.w = self.optimizer(loss_function, zeros((X.shape[1], 1)), step_rate=self.l_rate,
                                    max_iter=self.epochs, args=args).minimize()[0][:, 0]
        return self

    def predict_score(self, x):
        return CrossEntropy.predict(x, self.w)

    def predict(self, x):
        return np.where(self.predict_score(x) >= 0.5, self.labels[1], self.labels[0]).astype(int)


class MultiLogisticRegressionLearner(Learner):
    def __init__(self, l_rate=0.01, epochs=1000, batch_size=None, optimizer=GD,
                 regularization_type='l2', lmbda=0.1, alpha=0.2, decision_function='ovr'):
        self.l_rate = l_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.regularization_type = regularization_type
        self.lmbda = lmbda
        self.alpha = alpha
        if decision_function not in ('ovr', 'ovo'):
            raise ValueError("decision function must be either 'ovr' or 'ovo'")
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
                clf = BinaryLogisticRegressionLearner(self.l_rate, self.epochs, self.batch_size, self.optimizer,
                                                      self.regularization_type, self.lmbda, self.alpha)
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
                    clf = BinaryLogisticRegressionLearner(self.l_rate, self.epochs, self.batch_size, self.optimizer,
                                                          self.regularization_type, self.lmbda, self.alpha)
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
