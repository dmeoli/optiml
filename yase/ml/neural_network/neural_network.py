import warnings

import autograd.numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .activations import sigmoid, linear, softmax
from .initializers import compute_fans
from .layers import Layer, ParamLayer
from .losses import (CategoricalCrossEntropy, SparseCategoricalCrossEntropy,
                     MeanSquaredError, BinaryCrossEntropy, mean_squared_error)
from ...optimization.unconstrained import ProximalBundle
from ...optimization.unconstrained.line_search import LineSearchOptimizer
from ...optimization.unconstrained.stochastic import StochasticOptimizer, StochasticGradientDescent


class NeuralNetwork(BaseEstimator, Layer):

    def __init__(self, layers=(), loss=mean_squared_error, optimizer=StochasticGradientDescent, learning_rate=0.01,
                 max_iter=1000, momentum_type='none', momentum=0.9, validation_split=0.1, batch_size=None,
                 max_f_eval=1000, master_solver='ECOS', shuffle=True, random_state=None, verbose=False):
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
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.loss_history = {'train_loss': [],
                             'val_loss': []}

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
                coef_grads.append((grads['dW'] + layer.coef_reg.jacobian(layer.coef_)) / layer._X.shape[0])
                if layer.fit_intercept:
                    inter_grads.append((grads['db'] + layer.inter_reg.jacobian(layer.inter_)) / layer._X.shape[0])
            else:
                delta = layer.backward(delta)
        return coef_grads[::-1], inter_grads[::-1]

    @property
    def coefs_(self):
        return [layer.coef_ for layer in self.layers if isinstance(layer, ParamLayer)]

    @property
    def intercepts_(self):
        return [layer.inter_ for layer in self.layers if isinstance(layer, ParamLayer) and layer.fit_intercept]

    def _pack(self, coefs, intercepts):
        return np.hstack([w.ravel() for w in coefs + intercepts])

    def _unpack(self, packed_coef_inter):
        coef_idx = 0
        inter_idx = 0
        for layer in self.layers:
            if isinstance(layer, ParamLayer):
                start, end, shape = self.coef_idx[coef_idx]
                layer.coef_ = np.reshape(packed_coef_inter[start:end], shape)
                if layer.fit_intercept:
                    start, end = self.inter_idx[inter_idx]
                    layer.inter_ = packed_coef_inter[start:end]
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
                end = start + (np.prod(layer.coef_.shape))
                self.coef_idx.append((start, end, layer.coef_.shape))
                start = end
        # save sizes and indices of intercepts for faster unpacking
        for layer in self.layers:
            if isinstance(layer, ParamLayer) and layer.fit_intercept:
                fan_in, fan_out = compute_fans(layer.inter_.shape)
                end = start + fan_out
                self.inter_idx.append((start, end))
                start = end

    def _store_print_train_val_info(self, opt, X_batch, y_batch, X_val, y_val):
        assert opt.f_x == self.loss.function(opt.x, X_batch, y_batch)  # TODO remove this at the end
        self.loss_history['train_loss'].append(opt.f_x)
        val_loss = self.loss.function(opt.x, X_val, y_val)
        self.loss_history['val_loss'].append(val_loss)
        if self.verbose and not opt.epoch % self.verbose:
            print('\tval_loss: {:1.4e}'.format(val_loss), end='')

    def fit(self, X, y):

        self._store_meta_info()

        packed_coef_inter = self._pack(self.coefs_, self.intercepts_)

        if isinstance(self.optimizer, str):  # scipy optimization
            self.loss = self.loss(self, X, y)

            method = self.optimizer
            if self.loss.ndim == 2:
                self.optimizer = {'x0_history': [],
                                  'x1_history': [],
                                  'f_x_history': []}

            def _save_opt_steps(x):
                if self.loss.ndim == 2:
                    self.optimizer['x0_history'].append(x[0])
                    self.optimizer['x1_history'].append(x[1])
                    self.optimizer['f_x_history'].append(self.loss.function(x))

            res = minimize(fun=self.loss.function, jac=self.loss.jacobian,
                           x0=packed_coef_inter, method=method,
                           callback=_save_opt_steps,
                           options={'disp': self.verbose,
                                    'maxiter': self.max_iter,
                                    'maxfun': self.max_f_eval})

            if res.status != 0:
                warnings.warn('max_iter reached but the optimization has not converged yet')

            self._unpack(res.x)
        else:
            if issubclass(self.optimizer, LineSearchOptimizer):

                self.loss = self.loss(self, X, y)
                self.optimizer = self.optimizer(f=self.loss, x=packed_coef_inter, max_iter=self.max_iter,
                                                max_f_eval=self.max_f_eval, verbose=self.verbose)
                res = self.optimizer.minimize()

                if res[2] != 'optimal':
                    warnings.warn('max_iter reached but the optimization has not converged yet')

            elif issubclass(self.optimizer, StochasticOptimizer):

                # don't stratify in multi-label classification # TODO fix multi-label case
                should_stratify = isinstance(self, NeuralNetworkClassifier) and self.n_classes == 2
                stratify = y if should_stratify else None
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_split,
                                                                  stratify=stratify, random_state=self.random_state)
                self.loss = self.loss(self, X_train, y_train)
                self.optimizer = self.optimizer(f=self.loss, x=packed_coef_inter, step_size=self.learning_rate,
                                                epochs=self.max_iter, batch_size=self.batch_size,
                                                momentum_type=self.momentum_type, momentum=self.momentum,
                                                callback=self._store_print_train_val_info, callback_args=(X_val, y_val),
                                                shuffle=self.shuffle, random_state=self.random_state,
                                                verbose=self.verbose)
                res = self.optimizer.minimize()

            elif issubclass(self.optimizer, ProximalBundle):

                self.loss = self.loss(self, X, y)
                self.optimizer = self.optimizer(f=self.loss, x=packed_coef_inter, max_iter=self.max_iter,
                                                master_solver=self.master_solver, momentum_type=self.momentum_type,
                                                momentum=self.momentum, verbose=self.verbose)
                res = self.optimizer.minimize()

            else:
                raise ValueError(f'unknown optimizer {self.optimizer}')

            self._unpack(res[0])

        return self


class NeuralNetworkClassifier(ClassifierMixin, NeuralNetwork):

    def __init__(self, layers=(), loss=mean_squared_error, optimizer=StochasticGradientDescent, learning_rate=0.01,
                 max_iter=1000, momentum_type='none', momentum=0.9, validation_split=0.1, batch_size=None,
                 max_f_eval=1000, master_solver='ECOS', shuffle=True, random_state=None, verbose=False):
        super().__init__(layers, loss, optimizer, learning_rate, max_iter, momentum_type, momentum, validation_split,
                         batch_size, max_f_eval, master_solver, shuffle, random_state, verbose)
        self.n_classes = 0
        self.accuracy_history = {'train_acc': [],
                                 'val_acc': []}

    def _store_print_train_val_info(self, opt, X_batch, y_batch, X_val, y_val):
        super()._store_print_train_val_info(opt, X_batch, y_batch, X_val, y_val)
        acc = self.score(X_batch, y_batch)
        self.accuracy_history['train_acc'].append(acc)
        val_acc = self.score(X_val, y_val)
        self.accuracy_history['val_acc'].append(val_acc)
        if self.verbose and not opt.epoch % self.verbose:
            print('\tacc: {:1.4f}'.format(acc), end='')
            print('\tval_acc: {:1.4f}'.format(val_acc), end='')

    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # TODO fix multi-label case
        self.n_classes = y.shape[1] if self.loss == CategoricalCrossEntropy else np.unique(y).size
        if self.loss in (SparseCategoricalCrossEntropy, CategoricalCrossEntropy):
            if self.layers[-1].activation != softmax:
                raise ValueError(f'NeuralNetworkClassifier with {type(self.loss).__name__} loss '
                                 'function only works with softmax output layer')
            if self.layers[-1].fan_out != self.n_classes:
                raise ValueError('the number of neurons in the output layer must '
                                 f'be equal to the number of classes, i.e. {self.n_classes}')
        elif self.loss in (MeanSquaredError, BinaryCrossEntropy):
            if self.n_classes > 2:
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
        y = np.argmax(y, axis=1) if isinstance(self.loss, CategoricalCrossEntropy) else y
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class NeuralNetworkRegressor(RegressorMixin, NeuralNetwork):

    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if self.layers[-1].activation not in (linear, sigmoid):
            raise ValueError('NeuralNetworkRegressor only works with linear or '
                             'sigmoid (for regression between 0 and 1) output layer')
        if self.loss == BinaryCrossEntropy:
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
