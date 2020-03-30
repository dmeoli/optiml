import numpy as np
from sklearn.base import MultiOutputMixin, RegressorMixin, BaseEstimator
from sklearn.linear_model._base import LinearClassifierMixin

from ml.losses import mean_squared_error, cross_entropy
from ml.neural_network.activations import Sigmoid
from ml.regularizers import l1, l2
from optimization.optimization_function import OptimizationFunction
from optimization.optimizer import LineSearchOptimizer


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
            return self.loss(self.linear_model.predict(self.X, self.x_star()), self.y)
        return super().f_star()

    def args(self):
        return self.X, self.y

    def function(self, theta, X, y):
        return self.loss(self.linear_model.predict(X, theta), y) + self.linear_model.regularization(theta)

    def jacobian(self, theta, X, y):
        return (np.dot(X.T, self.linear_model.predict(X, theta) - y) +
                self.linear_model.regularization.jacobian(theta) / X.shape[0])

    def plot(self):
        surface_plot, surface_axes, contour_plot, contour_axes = super().plot()
        # TODO add loss and accuracy plot over iterations


class LinearRegression(BaseEstimator, MultiOutputMixin, RegressorMixin):

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
            raise ValueError('use sklearn.multioutput.MultiOutputRegressor to train a model over more than one target')

        loss = LinearModelLossFunction(X, y, self, mean_squared_error)
        if issubclass(self.optimizer, LineSearchOptimizer):
            self.w = self.optimizer(f=loss, batch_size=self.batch_size, max_iter=self.epochs,
                                    max_f_eval=self.max_f_eval, verbose=self.verbose).minimize()[0]
        else:
            self.w = self.optimizer(f=loss, batch_size=self.batch_size, step_rate=self.learning_rate,
                                    max_iter=self.epochs, verbose=self.verbose).minimize()[0]
        return self

    def predict(self, X, theta=None):
        return np.dot(X, theta if theta is not None else self.w)


class LogisticRegression(BaseEstimator, LinearClassifierMixin):

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
            raise ValueError('use OneVsOneClassifier or OneVsRestClassifier from sklearn.multiclass '
                             'to train a model over more than two labels')
        y = np.where(y == self.labels[0], 0, 1)

        loss = LinearModelLossFunction(X, y, self, cross_entropy)
        if issubclass(self.optimizer, LineSearchOptimizer):
            self.w = self.optimizer(f=loss, batch_size=self.batch_size, max_iter=self.epochs,
                                    max_f_eval=self.max_f_eval, verbose=self.verbose).minimize()[0]
        else:
            self.w = self.optimizer(f=loss, batch_size=self.batch_size, step_rate=self.learning_rate,
                                    max_iter=self.epochs, verbose=self.verbose).minimize()[0]
        return self

    def decision_function(self, X, theta=None):
        return Sigmoid().function(np.dot(X, theta if theta is not None else self.w))

    def predict(self, X, theta=None):
        return np.where(self.decision_function(X, theta) >= 0.5, self.labels[1], self.labels[0])
