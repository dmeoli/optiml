import copy

import numpy as np

from ml.losses import mean_squared_error, cross_entropy
from ml.neural_network.activations import Sigmoid
from ml.regularizers import l1, l2
from optimization.optimization_function import OptimizationFunction
from optimization.optimizer import LineSearchOptimizer


class Learner:
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class MultiClassClassifier(Learner):
    def __init__(self, learner, decision_function='ovr'):
        self.learner = learner
        if decision_function not in ('ovr', 'ovo'):
            raise ValueError('unknown decision function type {}'.format(decision_function))
        self.decision_function = decision_function
        self.n_class, self.classifiers = 0, []

    def fit(self, X, y, **kwargs):
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
                y1[y1 != label] = -1.
                y1[y1 == label] = 1.
                clf = copy.deepcopy(self.learner)
                clf.fit(X, y1, **kwargs)
                self.classifiers.append(clf)
        else:  # use one-vs-one method
            n_labels = len(labels)
            for i in range(n_labels):
                for j in range(i + 1, n_labels):
                    neg_id, pos_id = y == labels[i], y == labels[j]
                    x1, y1 = np.r_[X[neg_id], X[pos_id]], np.r_[y[neg_id], y[pos_id]]
                    y1[y1 == labels[i]] = -1.
                    y1[y1 == labels[j]] = 1.
                    clf = copy.deepcopy(self.learner)
                    clf.fit(x1, y1, **kwargs)
                    self.classifiers.append(clf)
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
                    vote[res < 0, i] += 1.  # negative sample: class i
                    vote[res > 0, j] += 1.  # positive sample: class j
                    clf_id += 1
            return np.argmax(vote, axis=1)


class MultiTargetRegressor(Learner):
    """This strategy consists of fitting one regressor per target. This is a
    simple strategy for extending regressors that do not natively support
    multi-target regression.
    """

    def __init__(self, learner):
        self.learner = learner
        self.learners = []

    def fit(self, X, y, **kwargs):
        """
        Trains n_output learner
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples, n_target] holding the class labels
        :return: array of classifiers
        """
        self.n_output = y.shape[1]
        for target in range(self.n_output):
            clf = copy.deepcopy(self.learner)
            clf.fit(X, y[:, target].ravel(), **kwargs)
            self.learners.append(clf)
        return self

    def predict(self, X):
        y_pred = np.zeros((X.shape[0], self.n_output))
        for target in range(self.n_output):
            y_pred[:, target] = self.learners[target].predict(X)
        return y_pred


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
                # or np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
                self.x_opt = np.linalg.lstsq(self.X, self.y)[0]
            return self.x_opt

    def f_star(self):
        if self.x_star() is not None:
            return self.loss(self.linear_model._predict(self.X, self.x_star()), self.y)
        return super().f_star()

    def args(self):
        return self.X, self.y

    def function(self, theta, X, y):
        return self.loss(self.linear_model._predict(X, theta), y) + self.linear_model.regularization(theta)

    def jacobian(self, theta, X, y):
        return (np.dot(X.T, self.linear_model._predict(X, theta) - y) +
                self.linear_model.regularization.jacobian(theta) / X.shape[0])

    def plot(self):
        surface_plot, surface_axes, contour_plot, contour_axes = super().plot()
        # TODO add loss and accuracy plot over iterations


class LinearRegressionLearner(Learner):

    def __init__(self, optimizer, learning_rate=0.01, epochs=1000, batch_size=None,
                 max_f_eval=1000, regularization=l1, lmbda=0.01, verbose=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.max_f_eval = max_f_eval
        self.regularization = regularization
        self.lmbda = lmbda
        self.verbose = verbose

    def fit(self, X, y):
        self.targets = y.shape[1] if y.ndim > 1 else 1
        if self.targets > 1:
            raise ValueError('use MultiTargetRegressor to train a model over more than one target')

        loss = LinearModelLossFunction(X, y, self, mean_squared_error)
        if issubclass(self.optimizer, LineSearchOptimizer):
            self.w = self.optimizer(f=loss, batch_size=self.batch_size, max_iter=self.epochs,
                                    max_f_eval=self.max_f_eval, verbose=self.verbose).minimize()[0]
        else:
            self.w = self.optimizer(f=loss, batch_size=self.batch_size, step_rate=self.learning_rate,
                                    max_iter=self.epochs, verbose=self.verbose).minimize()[0]
        return self

    def _predict(self, X, theta):
        return np.dot(X, theta)

    def predict(self, X):
        return self._predict(X, self.w)


class LogisticRegressionLearner(Learner):

    def __init__(self, optimizer, learning_rate=0.01, epochs=1000, batch_size=None,
                 max_f_eval=1000, regularization=l2, lmbda=0.01, verbose=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.max_f_eval = max_f_eval
        self.regularization = regularization
        self.lmbda = lmbda
        self.verbose = verbose

    def fit(self, X, y):
        self.labels = np.unique(y)
        if self.labels.size > 2:
            raise ValueError('use MultiClassClassifier to train a model over more than two labels')
        y = np.where(y == self.labels[0], 0, 1)

        loss = LinearModelLossFunction(X, y, self, cross_entropy)
        if issubclass(self.optimizer, LineSearchOptimizer):
            self.w = self.optimizer(f=loss, batch_size=self.batch_size, max_iter=self.epochs,
                                    max_f_eval=self.max_f_eval, verbose=self.verbose).minimize()[0]
        else:
            self.w = self.optimizer(f=loss, batch_size=self.batch_size, step_rate=self.learning_rate,
                                    max_iter=self.epochs, verbose=self.verbose).minimize()[0]
        return self

    def _predict(self, X, theta):
        return Sigmoid().function(np.dot(X, theta))

    def predict_score(self, X):
        return self._predict(X, self.w)

    def predict(self, X):
        return np.where(self.predict_score(X) >= 0.5, self.labels[1], self.labels[0])
