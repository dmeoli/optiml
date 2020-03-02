import copy

import numpy as np

from ml.losses import mean_squared_error, cross_entropy
from ml.neural_network.activations import Sigmoid
from optimization.optimization_function import OptimizationFunction
from optimization.optimizer import LineSearchOptimizer


class Learner:
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class LinearModelLossFunction(OptimizationFunction):

    def __init__(self, X, y, linear_model, loss):
        super().__init__(X.shape[1])
        self.X = X
        self.y = y
        self.linear_model = linear_model
        self.loss = loss

    def x_star(self):
        if self.loss is mean_squared_error:
            if not hasattr(self, 'x_opt'):
                # or np.linalg.lstsq(self.X, self.y)[0]
                self.x_opt = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
            return self.x_opt

    def f_star(self):
        if self.x_star() is not None:
            return self.function(self.x_star(), self.X, self.y)
        return super().f_star()

    def args(self):
        return self.X, self.y

    def function(self, theta, X, y):
        return self.loss(self.linear_model._predict(X, theta), y)

    def jacobian(self, theta, X, y):
        return np.dot(X.T, self.linear_model._predict(X, theta) - y) / X.shape[0]


class LinearRegressionLearner(Learner):

    def __init__(self, optimizer, learning_rate=0.01, epochs=1000, batch_size=None, max_f_eval=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.max_f_eval = max_f_eval

    def fit(self, X, y, verbose=False):
        loss = LinearModelLossFunction(X, y, self, mean_squared_error)
        if issubclass(self.optimizer, LineSearchOptimizer):
            self.w = self.optimizer(f=loss, batch_size=self.batch_size, max_iter=self.epochs,
                                    max_f_eval=self.max_f_eval, verbose=verbose).minimize()[0]
        else:
            self.w = self.optimizer(f=loss, batch_size=self.batch_size, step_rate=self.learning_rate,
                                    max_iter=self.epochs, verbose=verbose).minimize()[0]
        return self

    def _predict(self, X, theta):
        return np.dot(X, theta)

    def predict(self, X):
        return self._predict(X, self.w)


class BinaryLogisticRegressionLearner(Learner):

    def __init__(self, optimizer, learning_rate=0.01, epochs=1000, batch_size=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

    def fit(self, X, y):
        self.labels = np.unique(y)
        y = np.where(y == self.labels[0], 0, 1)
        loss = LinearModelLossFunction(X, y, self, cross_entropy)
        if issubclass(self.optimizer, LineSearchOptimizer):
            self.w = self.optimizer(f=loss, batch_size=self.batch_size, max_iter=self.epochs).minimize()[0]
        else:
            self.w = self.optimizer(f=loss, batch_size=self.batch_size, step_rate=self.learning_rate,
                                    max_iter=self.epochs).minimize()[0]
        return self

    def _predict(self, X, theta):
        return Sigmoid().function(np.dot(X, theta))

    def predict_score(self, X):
        return self._predict(X, self.w)

    def predict(self, X):
        return np.where(self.predict_score(X) >= 0.5, self.labels[1], self.labels[0]).astype(int)


class MultiLogisticRegressionLearner(Learner):
    def __init__(self, optimizer, learning_rate=0.01, epochs=1000, batch_size=None, decision_function='ovr'):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
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
                clf = BinaryLogisticRegressionLearner(self.optimizer, self.learning_rate, self.epochs, self.batch_size)
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
                    clf = BinaryLogisticRegressionLearner(self.optimizer, self.learning_rate,
                                                          self.epochs, self.batch_size)
                    clf.fit(x1, y1)
                    self.classifiers.append(copy.deepcopy(clf))
        return self

    def predict(self, X):
        """
        Predicts the class of a given example according to the training method.
        """
        n_samples = len(X)
        if self.decision_function == 'ovr':  # one-vs-rest method
            assert len(self.classifiers) == self.n_class
            score = np.zeros((n_samples, self.n_class))
            for i in range(self.n_class):
                clf = self.classifiers[i]
                score[:, i] = clf.predict_score(X)
            return np.argmax(score, axis=1)
        elif self.decision_function == 'ovo':  # use one-vs-one method
            assert len(self.classifiers) == self.n_class * (self.n_class - 1) / 2
            vote = np.zeros((n_samples, self.n_class))
            clf_id = 0
            for i in range(self.n_class):
                for j in range(i + 1, self.n_class):
                    res = self.classifiers[clf_id].predict(X)
                    vote[res < 0, i] += 1.0  # negative sample: class i
                    vote[res > 0, j] += 1.0  # positive sample: class j
                    clf_id += 1
            return np.argmax(vote, axis=1)
