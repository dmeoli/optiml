import warnings

import autograd.numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ml.neural_network.activations import sigmoid, linear, softmax
from ml.neural_network.initializers import compute_fans
from ml.neural_network.layers import Layer, ParamLayer
from ml.neural_network.losses import (mean_squared_error, categorical_cross_entropy, binary_cross_entropy,
                                      sparse_categorical_cross_entropy)
from optimization.optimization_function import OptimizationFunction
from optimization.unconstrained.line_search.line_search_optimizer import LineSearchOptimizer
from optimization.unconstrained.proximal_bundle import ProximalBundle
from optimization.unconstrained.stochastic.stochastic_gradient_descent import StochasticGradientDescent
from optimization.unconstrained.stochastic.stochastic_optimizer import StochasticOptimizer

plt.style.use('ggplot')


class NeuralNetworkLossFunction(OptimizationFunction):

    def __init__(self, neural_net, X_train, X_test, y_train, y_test, x_min=-10, x_max=10, y_min=-10, y_max=10):
        super().__init__(X_train.shape[1], x_min, x_max, y_min, y_max)
        self.neural_net = neural_net
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def args(self):
        return self.X_train, self.y_train

    def function(self, packed_coef_inter, X=None, y=None):
        if X is None:
            X = self.X_train
        if y is None:
            y = self.y_train

        self.neural_net._unpack(packed_coef_inter)

        coef_regs = np.sum(layer.w_reg(layer.W) for layer in self.neural_net.layers
                           if isinstance(layer, ParamLayer)) / X.shape[0]
        inter_regs = np.sum(layer.b_reg(layer.b) for layer in self.neural_net.layers
                            if isinstance(layer, ParamLayer) and layer.use_bias) / X.shape[0]
        return self.neural_net.loss(self.neural_net.forward(X), y) + coef_regs + inter_regs

    def jacobian(self, packed_coef_inter, X=None, y=None):
        if X is None:
            X = self.X_train
        if y is None:
            y = self.y_train

        delta = self.neural_net.loss.jacobian(self.neural_net.forward(X), y)
        return self.neural_net._pack(*self.neural_net.backward(delta))

    def store_plot_data(self, packed_coef_inter, X=None, y=None):
        if X is None:
            X = self.X_train
        if y is None:
            y = self.y_train

        self.neural_net.loss_history['training_loss'].append(self.function(packed_coef_inter, X, y))
        self.neural_net.loss_history['validation_loss'].append(
            self.function(packed_coef_inter, self.X_test, self.y_test))
        if isinstance(self.neural_net, NeuralNetworkClassifier):
            self.neural_net.accuracy_history['training_accuracy'].append(
                self.neural_net.score(X, y))
            self.neural_net.accuracy_history['validation_accuracy'].append(
                self.neural_net.score(self.X_test, self.y_test))


class NeuralNetwork(BaseEstimator, Layer):

    def __init__(self, layers=(), loss=mean_squared_error, optimizer=StochasticGradientDescent, learning_rate=0.01,
                 max_iter=1000, momentum_type='none', momentum=0.9, validation_split=0.2, batch_size=None,
                 max_f_eval=1000, master_solver='ECOS', verbose=False, plot=False):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum_type = momentum_type
        self.momentum = momentum
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.max_f_eval = max_f_eval
        self.master_solver = master_solver
        self.verbose = verbose
        self.plot = plot
        self.loss_history = {'training_loss': [],
                             'validation_loss': []}

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, delta):
        coef_grads = []
        inter_grads = []
        # back propagate
        for layer in self.layers[::-1]:
            if isinstance(layer, ParamLayer):
                delta, grads = layer.backward(delta)
                coef_grads.append(grads['dW'] + layer.w_reg.jacobian(layer.W))
                if layer.use_bias:
                    inter_grads.append(grads['db'] + layer.b_reg.jacobian(layer.b))
            else:
                delta = layer.backward(delta)
        return coef_grads[::-1], inter_grads[::-1]

    @property
    def params(self):
        return ([layer.W for layer in self.layers if isinstance(layer, ParamLayer)],
                [layer.b for layer in self.layers if isinstance(layer, ParamLayer) and layer.use_bias])

    def _pack(self, coefs, intercepts):
        return np.hstack([w.ravel() for w in coefs + intercepts])

    def _unpack(self, packed_coef_inter):
        coef_idx = 0
        inter_idx = 0
        for layer in self.layers:
            if isinstance(layer, ParamLayer):
                start, end, shape = self.coef_idx[coef_idx]
                layer.W = np.reshape(packed_coef_inter[start:end], shape)
                if layer.use_bias:
                    start, end = self.inter_idx[inter_idx]
                    layer.b = packed_coef_inter[start:end]
                    inter_idx += 1
                coef_idx += 1

    def _store_meta_info(self):
        # store meta information for the parameters
        self.coef_idx = []
        self.inter_idx = []
        start = 0
        # save sizes and indices of coefs for faster unpacking
        for layer in self.layers:
            if isinstance(layer, ParamLayer):
                end = start + (np.prod(layer.W.shape))
                self.coef_idx.append((start, end, layer.W.shape))
                start = end
        # save sizes and indices of intercepts for faster unpacking
        for layer in self.layers:
            if isinstance(layer, ParamLayer) and layer.use_bias:
                fan_in, fan_out = compute_fans(layer.b.shape)
                end = start + fan_out
                self.inter_idx.append((start, end))
                start = end

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.validation_split)

        self._store_meta_info()

        packed_coef_inter = self._pack(*self.params)

        loss = NeuralNetworkLossFunction(self, X_train, X_test, y_train, y_test)

        if isinstance(self.optimizer, str):  # scipy optimization
            res = minimize(fun=loss.function, jac=loss.jacobian, args=loss.args(),
                           x0=packed_coef_inter, method=self.optimizer,
                           callback=loss.store_plot_data,
                           options={'disp': self.verbose,
                                    'maxiter': self.max_iter,
                                    'maxfun': self.max_f_eval})
            if res.status != 0:
                warnings.warn('max_iter reached but the optimization has not converged yet')
            self._unpack(res.x)
        else:
            if issubclass(self.optimizer, LineSearchOptimizer):
                res = self.optimizer(f=loss, x=packed_coef_inter, max_iter=self.max_iter,
                                     max_f_eval=self.max_f_eval, callback=loss.store_plot_data,
                                     verbose=self.verbose, plot=self.plot).minimize()
                if res[2] != 'optimal':
                    warnings.warn('max_iter reached but the optimization has not converged yet')
            elif issubclass(self.optimizer, StochasticOptimizer):
                res = self.optimizer(f=loss, x=packed_coef_inter, batch_size=self.batch_size, epochs=self.max_iter,
                                     step_size=self.learning_rate, momentum_type=self.momentum_type,
                                     momentum=self.momentum,
                                     callback=loss.store_plot_data,
                                     callback_args=(X_train, y_train),  # mini batches
                                     verbose=self.verbose, plot=self.plot).minimize()
            elif issubclass(self.optimizer, ProximalBundle):
                res = self.optimizer(f=loss, x=packed_coef_inter, max_iter=self.max_iter,
                                     master_solver=self.master_solver, momentum_type=self.momentum_type,
                                     momentum=self.momentum, callback=loss.store_plot_data,
                                     verbose=self.verbose, plot=self.plot).minimize()
            else:
                raise ValueError(f'unknown optimizer {type(self.optimizer).__name__}')

            self._unpack(res[0])

        return self


class NeuralNetworkClassifier(ClassifierMixin, NeuralNetwork):

    def __init__(self, layers=(), loss=mean_squared_error, optimizer=StochasticGradientDescent, learning_rate=0.01,
                 max_iter=1000, momentum_type='none', momentum=0.9, validation_split=0.2, batch_size=None,
                 max_f_eval=1000, master_solver='ECOS', verbose=False, plot=False):
        super().__init__(layers, loss, optimizer, learning_rate, max_iter, momentum_type, momentum,
                         validation_split, batch_size, max_f_eval, master_solver, verbose, plot)
        self.accuracy_history = {'training_accuracy': [],
                                 'validation_accuracy': []}

    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_classes = np.unique(y).size
        if self.loss in (sparse_categorical_cross_entropy, categorical_cross_entropy):
            if self.layers[-1].activation != softmax:
                raise ValueError(f'NeuralNetworkClassifier with {type(self.loss).__name__} loss '
                                 'function only works with softmax output layer')
            if self.layers[-1].fan_out != n_classes:
                raise ValueError('the number of neurons in the output layer must '
                                 f'be equal to the number of classes, i.e. {n_classes}')
        elif self.loss in (mean_squared_error, binary_cross_entropy):
            if n_classes > 2:
                raise ValueError(f'NeuralNetworkClassifier with {type(self.loss).__name__} '
                                 'loss function only works for binary classification')
            if self.layers[-1].activation != sigmoid:
                raise ValueError(f'NeuralNetworkClassifier with {type(self.loss).__name__} '
                                 'loss function only works with sigmoid output layer')
            if self.layers[-1].fan_out != 1:
                raise ValueError(f'NeuralNetworkClassifier with {type(self.loss).__name__} loss '
                                 'function only works with one neuron in the output layer')

        return super().fit(X, y)

    def predict(self, X):
        if self.layers[-1].activation == sigmoid:
            return self.forward(X) >= 0.5
        elif self.layers[-1].activation == softmax:
            return np.argmax(self.forward(X), axis=1)
        else:
            return self.forward(X)

    def score(self, X, y, sample_weight=None):
        y = np.argmax(y, axis=1) if self.loss == categorical_cross_entropy else y
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class NeuralNetworkRegressor(RegressorMixin, NeuralNetwork):

    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if self.layers[-1].activation not in (linear, sigmoid):
            raise ValueError('NeuralNetworkRegressor only works with linear or '
                             'sigmoid (for regression between 0 and 1) output layer')
        if self.loss == binary_cross_entropy:
            if self.layers[-1].activation != sigmoid:
                raise ValueError('NeuralNetworkRegressor with binary_cross_entropy loss function only '
                                 'works with sigmoid output layer for regression between 0 and 1')
            if not (0 <= y <= 1).all():
                raise ValueError('NeuralNetworkRegressor with binary_cross_entropy loss '
                                 'function only works for regression between 0 and 1')
        n_targets = y.shape[1]
        if self.layers[-1].fan_out != n_targets:
            raise ValueError(f'the number of neurons in the output layer must be '
                             f'equal to the number of targets, i.e. {n_targets}')

        return super().fit(X, y)

    def predict(self, X):
        if self.layers[-1].fan_out == 1:  # one target
            return self.forward(X).ravel()
        else:  # multi target
            return self.forward(X)
