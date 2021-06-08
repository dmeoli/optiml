import warnings
from abc import ABC

import autograd.numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .activations import sigmoid, linear, softmax
from .layers import Layer, ParamLayer
from .losses import (CategoricalCrossEntropy, SparseCategoricalCrossEntropy,
                     MeanSquaredError, BinaryCrossEntropy, mean_squared_error, NeuralNetworkLoss)
from ...opti import Optimizer
from ...opti.unconstrained import ProximalBundle
from ...opti.unconstrained.line_search import LineSearchOptimizer
from ...opti.unconstrained.stochastic import StochasticOptimizer, StochasticGradientDescent, StochasticMomentumOptimizer


class NeuralNetwork(BaseEstimator, Layer, ABC):

    def __init__(self,
                 layers=(),
                 loss=mean_squared_error,
                 optimizer=StochasticGradientDescent,
                 learning_rate=0.01,
                 max_iter=1000,
                 momentum_type='none',
                 momentum=0.9,
                 tol=1e-4,
                 validation_split=0.,
                 batch_size=None,
                 max_f_eval=15000,
                 early_stopping=False,
                 patience=5,
                 shuffle=True,
                 random_state=None,
                 mu=1,
                 master_solver='ecos',
                 master_verbose=False,
                 verbose=False):
        self.layers = layers
        if not issubclass(loss, NeuralNetworkLoss):
            raise TypeError(f'{loss} is not an allowed neural network loss function')
        self.loss = loss
        if not issubclass(optimizer, Optimizer):
            raise TypeError(f'{optimizer} is not an allowed optimization method')
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum_type = momentum_type
        self.momentum = momentum
        self.tol = tol
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.max_f_eval = max_f_eval
        self.early_stopping = early_stopping
        self.patience = patience
        self.shuffle = shuffle
        self.random_state = random_state
        self.mu = mu
        self.master_solver = master_solver
        self.master_verbose = master_verbose
        self.verbose = verbose
        if issubclass(self.optimizer, StochasticOptimizer):
            self.train_loss_history = []
            self.train_score_history = []
            self._no_improvement_count = 0
            self._avg_epoch_loss = 0
            if self.validation_split:
                self.val_loss_history = []
                self.val_score_history = []
                self.best_val_score = -np.inf
            else:
                self.best_loss = np.inf

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, delta):
        coef_grads = []
        inter_grads = []
        # backpropagate
        for layer in self.layers[::-1]:
            if isinstance(layer, ParamLayer):
                delta, grads = layer.backward(delta)
                coef_grads.append(grads['dW'] + layer.coef_reg.jacobian(layer.coef_) / layer._X.shape[0])
                if layer.fit_intercept:
                    inter_grads.append(grads['db'] + layer.inter_reg.jacobian(layer.inter_) / layer._X.shape[0])
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
                fan_in, fan_out = layer.inter_.shape[0], layer.inter_.shape[1]
                end = start + fan_out
                self.inter_idx.append((start, end))
                start = end

    def _store_train_val_info(self, opt, X_batch, y_batch, X_val, y_val):
        self._avg_epoch_loss += opt.f_x * X_batch.shape[0]
        if opt.is_batch_end():
            self._avg_epoch_loss /= opt.f.X.shape[0]  # n_samples
            self.train_loss_history.append(self._avg_epoch_loss)
            if opt.is_verbose() and opt.epoch != opt.iter:
                print('\tavg_loss: {: 1.4e}'.format(self._avg_epoch_loss), end='')
            self._avg_epoch_loss = 0.
            if self.validation_split:
                val_loss = self.loss(opt.x, X_val, y_val)
                self.val_loss_history.append(val_loss)
                if opt.is_verbose():
                    print('\tval_loss: {: 1.4e}'.format(val_loss), end='')

    def _update_no_improvement_count(self, opt):
        if self.early_stopping:

            if self.validation_split:  # monitor val_score

                if self.val_score_history[-1] < self.best_val_score + self.tol:
                    self._no_improvement_count += 1
                else:
                    self._no_improvement_count = 0
                if self.val_score_history[-1] > self.best_val_score:
                    self.best_val_score = self.val_score_history[-1]
                    self._best_coefs = [coef.copy() for coef in self.coefs_]
                    self._best_intercepts = [inter.copy() for inter in self.intercepts_]

            else:  # monitor train_loss

                if self.train_loss_history[-1] > self.best_loss - self.tol:
                    self._no_improvement_count += 1
                else:
                    self._no_improvement_count = 0
                if self.train_loss_history[-1] < self.best_loss:
                    self.best_loss = self.train_loss_history[-1]

            if self._no_improvement_count >= self.patience:

                if self.validation_split:
                    opt.x = self._pack(self._best_coefs, self._best_intercepts)

                if self.verbose:
                    if self.validation_split:
                        print(f'\ntraining stopped since validation score did not improve more than '
                              f'tol={self.tol} for {self.patience} consecutive epochs')
                    else:
                        print('\ntraining stopped since training loss did not improve more than '
                              f'tol={self.tol} for {self.patience} consecutive epochs')

                raise StopIteration

    def fit(self, X, y):

        self._store_meta_info()

        packed_coef_inter = self._pack(self.coefs_, self.intercepts_)

        if issubclass(self.optimizer, LineSearchOptimizer):

            self.loss = self.loss(self, X, y)
            self.optimizer = self.optimizer(f=self.loss,
                                            x=packed_coef_inter,
                                            max_iter=self.max_iter,
                                            max_f_eval=self.max_f_eval,
                                            verbose=self.verbose).minimize()

            if self.optimizer.status == 'stopped':
                if self.optimizer.iter >= self.max_iter:
                    warnings.warn('max_iter reached but the optimization has not converged yet', ConvergenceWarning)
                elif self.optimizer.f_eval >= self.max_f_eval:
                    warnings.warn('max_f_eval reached but the optimization has not converged yet', ConvergenceWarning)

        elif issubclass(self.optimizer, ProximalBundle):

            self.loss = self.loss(self, X, y)
            self.optimizer = self.optimizer(f=self.loss,
                                            x=packed_coef_inter,
                                            mu=self.mu,
                                            max_iter=self.max_iter,
                                            master_solver=self.master_solver,
                                            master_verbose=self.master_verbose,
                                            verbose=self.verbose).minimize()

            if self.optimizer.status == 'error':
                warnings.warn('failure while computing direction for the master problem', ConvergenceWarning)

        elif issubclass(self.optimizer, StochasticOptimizer):

            if self.validation_split:
                # don't stratify in multi-label classification
                should_stratify = isinstance(self, NeuralNetworkClassifier) and self.layers[-1].fan_out == 1
                stratify = y if should_stratify else None
                X, X_val, y, y_val = train_test_split(X, y,
                                                      stratify=stratify,
                                                      test_size=self.validation_split,
                                                      random_state=self.random_state)
            else:
                X_val = None
                y_val = None

            self.loss = self.loss(self, X, y)

            if issubclass(self.optimizer, StochasticMomentumOptimizer):

                self.optimizer = self.optimizer(f=self.loss,
                                                x=packed_coef_inter,
                                                step_size=self.learning_rate,
                                                epochs=self.max_iter,
                                                batch_size=self.batch_size,
                                                momentum_type=self.momentum_type,
                                                momentum=self.momentum,
                                                callback=self._store_train_val_info,
                                                callback_args=(X_val, y_val),
                                                shuffle=self.shuffle,
                                                random_state=self.random_state,
                                                verbose=self.verbose).minimize()

            else:

                self.optimizer = self.optimizer(f=self.loss,
                                                x=packed_coef_inter,
                                                step_size=self.learning_rate,
                                                epochs=self.max_iter,
                                                batch_size=self.batch_size,
                                                callback=self._store_train_val_info,
                                                callback_args=(X_val, y_val),
                                                shuffle=self.shuffle,
                                                random_state=self.random_state,
                                                verbose=self.verbose).minimize()

        else:

            raise TypeError(f'{self.optimizer} is not an allowed optimizer')

        self._unpack(self.optimizer.x)

        return self


class NeuralNetworkClassifier(ClassifierMixin, NeuralNetwork):

    def _store_train_val_info(self, opt, X_batch, y_batch, X_val, y_val):
        super(NeuralNetworkClassifier, self)._store_train_val_info(opt, X_batch, y_batch, X_val, y_val)
        if opt.is_batch_end():
            acc = self.score(X_batch, y_batch)
            self.train_score_history.append(acc)
            if opt.is_verbose():
                print('\tacc: {:1.4f}'.format(acc), end='')
            if self.validation_split:
                val_acc = self.score(X_val, y_val)
                self.val_score_history.append(val_acc)
                if opt.is_verbose():
                    print('\tval_acc: {:1.4f}'.format(val_acc), end='')
            self._update_no_improvement_count(opt)

    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_classes = y.shape[1] if self.loss == CategoricalCrossEntropy else np.unique(y).size
        if self.loss in (SparseCategoricalCrossEntropy, CategoricalCrossEntropy):
            if self.layers[-1].activation != softmax:
                raise ValueError(f'NeuralNetworkClassifier with {type(self.loss).__name__} loss '
                                 'function only works with softmax output layer')
            if self.layers[-1].fan_out != n_classes:
                raise ValueError('the number of neurons in the output layer must '
                                 f'be equal to the number of classes, i.e., {n_classes}')
        elif self.loss in (MeanSquaredError, BinaryCrossEntropy):
            if n_classes > 2:
                raise ValueError(f'NeuralNetworkClassifier with {type(self.loss).__name__} '
                                 'loss function only works for binary classification')
            if self.layers[-1].activation != sigmoid:
                raise ValueError(f'NeuralNetworkClassifier with {type(self.loss).__name__} '
                                 'loss function only works with sigmoid output layer')
            if self.layers[-1].fan_out != 1:
                raise ValueError(f'NeuralNetworkClassifier with {type(self.loss).__name__} loss '
                                 'function only works with one neuron in the output layer')

        return super(NeuralNetworkClassifier, self).fit(X, y)

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

    def _store_train_val_info(self, opt, X_batch, y_batch, X_val, y_val):
        super(NeuralNetworkRegressor, self)._store_train_val_info(opt, X_batch, y_batch, X_val, y_val)
        if opt.is_batch_end():
            r2 = self.score(X_batch, y_batch)
            self.train_score_history.append(r2)
            if opt.is_verbose():
                print('\tr2: {: 1.4f}'.format(r2), end='')
            if self.early_stopping:
                val_r2 = self.score(X_val, y_val)
                self.val_score_history.append(val_r2)
                if opt.is_verbose():
                    print('\tval_r2: {: 1.4f}'.format(val_r2), end='')
            self._update_no_improvement_count(opt)

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
                             f'equal to the number of targets, i.e., {n_targets}')

        return super(NeuralNetworkRegressor, self).fit(X, y)

    def predict(self, X):
        if self.layers[-1].fan_out == 1:  # one target
            return self.forward(X).ravel()
        else:  # multi target
            return self.forward(X)
