import autograd.numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ml.neural_network.initializers import compute_fans
from ml.neural_network.layers import Layer, ParamLayer
from ml.neural_network.losses import mean_squared_error, cross_entropy
from optimization.optimization_function import OptimizationFunction
from optimization.unconstrained.line_search.line_search_optimizer import LineSearchOptimizer
from optimization.unconstrained.stochastic.stochastic_optimizer import StochasticOptimizer
from optimization.unconstrained.stochastic.stochastic_gradient_descent import StochasticGradientDescent

plt.style.use('ggplot')


class NeuralNetworkLossFunction(OptimizationFunction):

    def __init__(self, neural_net, X_train, X_test, y_train, y_test):
        super().__init__(X_train.shape[1])
        self.neural_net = neural_net
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def args(self):
        return self.X_train, self.y_train

    def function(self, packed_weights_biases, X=None, y=None):
        if X is None:
            X = self.X_train
        if y is None:
            y = self.y_train

        self.neural_net._unpack(packed_weights_biases)

        def _function(X, y):
            weights_regs = np.sum(layer.w_reg(layer.W) for layer in self.neural_net.layers
                                  if isinstance(layer, ParamLayer))
            biases_regs = np.sum(layer.b_reg(layer.b) for layer in self.neural_net.layers
                                 if isinstance(layer, ParamLayer) and layer.use_bias)
            return self.neural_net.loss(self.neural_net.forward(X), y) + (weights_regs + biases_regs) / X.shape[0]

        loss = _function(X, y)
        self.neural_net.loss_history['training_loss'].append(_function(X, y))
        # self.neural_net.loss_history['validation_loss'].append(_function(self.X_test, self.y_test))
        return loss

    def jacobian(self, packed_weights_biases, X=None, y=None):
        """
        The calculation of delta here works with following
        combinations of loss function and output activation:
        mean squared error + identity
        cross entropy + softmax
        binary cross entropy + sigmoid
        """
        if X is None:
            X = self.X_train
        if y is None:
            y = self.y_train

        y_pred = self.neural_net.forward(X)
        assert y_pred.shape == y.shape
        return self.neural_net._pack(*self.neural_net.backward(y_pred - y))


class NeuralNetwork(BaseEstimator, Layer):

    def __init__(self, layers=(), loss=mean_squared_error, optimizer=StochasticGradientDescent, learning_rate=0.01,
                 epochs=100, momentum_type='none', momentum=0.9, validation_split=0.2,
                 batch_size=None, max_f_eval=1000, verbose=False):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum_type = momentum_type
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.max_f_eval = max_f_eval
        self.verbose = verbose
        self.loss_history = {'training_loss': [],
                             'validation_loss': []}

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, delta):
        weights_grads = []
        biases_grads = []
        # back propagate
        for layer in self.layers[::-1]:
            if isinstance(layer, ParamLayer):
                delta, grads = layer.backward(delta)
                weights_grads.append(grads['dW'] + layer.w_reg.jacobian(layer.W))
                if layer.use_bias:
                    biases_grads.append(grads['db'] + layer.b_reg.jacobian(layer.b))
            else:
                delta = layer.backward(delta)
        return weights_grads[::-1], biases_grads[::-1]

    @property
    def params(self):
        return ([layer.W for layer in self.layers if isinstance(layer, ParamLayer)],
                [layer.b for layer in self.layers if isinstance(layer, ParamLayer) and layer.use_bias])

    def _pack(self, weights, biases):
        return np.hstack([w.ravel() for w in weights + biases])

    def _unpack(self, packed_weights_biases):
        weight_idx = 0
        bias_idx = 0
        for layer in self.layers:
            if isinstance(layer, ParamLayer):
                start, end, shape = self.weights_idx[weight_idx]
                layer.W = np.reshape(packed_weights_biases[start:end], shape)
                if layer.use_bias:
                    start, end = self.biases_idx[bias_idx]
                    layer.b = packed_weights_biases[start:end]
                    bias_idx += 1
                weight_idx += 1

    def _store_meta_info(self):
        # store meta information for the parameters
        self.weights_idx = []
        self.biases_idx = []
        start = 0
        # save sizes and indices of weights for faster unpacking
        for layer in self.layers:
            if isinstance(layer, ParamLayer):
                end = start + (np.prod(layer.W.shape))
                self.weights_idx.append((start, end, layer.W.shape))
                start = end
        # save sizes and indices of biases for faster unpacking
        for layer in self.layers:
            if isinstance(layer, ParamLayer) and layer.use_bias:
                fan_in, fan_out = compute_fans(layer.b.shape)
                end = start + fan_out
                self.biases_idx.append((start, end))
                start = end

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.validation_split)

        self._store_meta_info()

        packed_weights_biases = self._pack(*self.params)

        loss = NeuralNetworkLossFunction(self, X_train, X_test, y_train, y_test)

        def store_val_loss(loss, X_test, y_test):
            self.loss_history['validation_loss'].append(loss.function(X_test, y_test))

        if issubclass(self.optimizer, LineSearchOptimizer):
            opt = self.optimizer(f=loss, wrt=packed_weights_biases, max_iter=self.epochs, max_f_eval=self.max_f_eval,
                                 verbose=self.verbose).minimize()
            # if opt[2] != 'optimal':
            #     warnings.warn("max_iter or max_f_eval reached and the optimization hasn't converged yet")
        elif issubclass(self.optimizer, StochasticOptimizer):
            opt = self.optimizer(f=loss, wrt=packed_weights_biases, batch_size=self.batch_size, max_iter=self.epochs,
                                 step_rate=self.learning_rate, momentum_type=self.momentum_type, momentum=self.momentum,
                                 verbose=self.verbose).minimize()
            # if opt[2] != 'optimal':
            #     warnings.warn("max_iter reached and the optimization hasn't converged yet")
        self._unpack(opt[0])

        return self


class NeuralNetworkClassifier(ClassifierMixin, NeuralNetwork):

    def __init__(self, layers=(), loss=cross_entropy, optimizer=StochasticGradientDescent, learning_rate=0.01,
                 epochs=100, momentum_type='none', momentum=0.9, validation_split=0.2,
                 batch_size=None, max_f_eval=1000, verbose=False):
        super().__init__(layers, loss, optimizer, learning_rate, epochs, momentum_type, momentum,
                         validation_split, batch_size, max_f_eval, verbose)
        self.accuracy_history = {'training_accuracy': [],
                                 'validation_accuracy': []}

    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        ohe = OneHotEncoder().fit(y)
        y = ohe.transform(y).toarray()
        return super().fit(X, y)

        # self.accuracy_history['training_accuracy'].append(
        #     self.score(X_train, self.ohe.inverse_transform(y_train)))
        #
        # validation_accuracy = self.score(X_test, self.ohe.inverse_transform(y_test))
        #
        # if self.verbose and not self.optimizer.iter % self.neural_net.verbose:
        #     print('\t accuracy: {:4f}'.format(validation_accuracy), end='')
        #
        # self.accuracy_history['validation_accuracy'].append(validation_accuracy)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


class NeuralNetworkRegressor(RegressorMixin, NeuralNetwork):

    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        super().fit(X, y)

    def predict(self, X):
        if self.layers[-1].fan_out == 1:  # one target
            return self.forward(X).ravel()
        else:  # multi target
            return self.forward(X)
