import inspect

import autograd.numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder

from ml.initializers import compute_fans
from ml.layers import Layer, ParamLayer
from ml.losses import mean_squared_error
from optimization.optimization_function import OptimizationFunction
from optimization.optimizer import LineSearchOptimizer
from optimization.unconstrained.gradient_descent import GradientDescent

plt.style.use('ggplot')


class NeuralNetworkLossFunction(OptimizationFunction):

    def __init__(self, neural_net, X_train, y_train, X_test=None, y_test=None):
        super().__init__(X_train.shape[1])
        self.neural_net = neural_net
        self.X_train = X_train
        self.X_test = X_test
        self.loss_history = {'training_loss': [],  # training loss history
                             'test_loss': []}  # test loss history
        if isinstance(self.neural_net, NeuralNetworkClassifier):
            self.train_ohe = OneHotEncoder().fit(y_train)
            self.y_train = self.train_ohe.transform(y_train).toarray()
            if y_test is not None:
                self.test_ohe = OneHotEncoder().fit(y_test)
                self.y_test = self.test_ohe.transform(y_test).toarray()
            self.accuracy_history = {'training_accuracy': [],  # training accuracy history
                                     'test_accuracy': []}  # test accuracy history
        else:
            self.y_train = y_train
            self.y_test = y_test

    def args(self):
        return self.X_train, self.y_train

    def function(self, packed_weights_biases, X, y):
        self.neural_net._unpack(packed_weights_biases)

        def _function(X, y):
            return (self.neural_net.loss(self.neural_net.forward(X), y) +
                    np.sum(np.sum(layer.w_reg(layer.W) for layer in self.neural_net.layers
                                  if isinstance(layer, ParamLayer)) +
                           np.sum(layer.b_reg(layer.b) for layer in self.neural_net.layers
                                  if isinstance(layer, ParamLayer) and layer.use_bias)) / X.shape[0])

        training_loss = _function(X, y)
        if inspect.stack()[1].function == 'minimize':  # ignore calls form line search

            if isinstance(self.neural_net, NeuralNetworkClassifier):

                self.accuracy_history['training_accuracy'].append(
                    self.neural_net.score(X, self.train_ohe.inverse_transform(y)))

                if self.X_test is not None and self.y_test is not None:
                    test_accuracy = self.neural_net.score(self.X_test, self.test_ohe.inverse_transform(self.y_test))

                    if self.neural_net.verbose and self.accuracy_history['test_accuracy']:
                        print('\t accuracy: {:4f}'.format(test_accuracy), end='')

                    self.accuracy_history['test_accuracy'].append(test_accuracy)

            self.loss_history['training_loss'].append(training_loss)
            if self.X_test is not None and self.y_test is not None:
                self.loss_history['test_loss'].append(_function(self.X_test, self.y_test))

        return training_loss

    def jacobian(self, packed_weights_biases, X, y):
        """
        The calculation of delta here works with following
        combinations of loss function and output activation:
        mean squared error + identity
        cross entropy + softmax
        binary cross entropy + sigmoid
        """
        y_pred = self.neural_net.forward(X)
        assert y_pred.shape == y.shape
        return self.neural_net._pack(*self.neural_net.backward(y_pred - y))

    def plot(self):
        # TODO add accuracy plot over iterations
        fig, loss = plt.subplots()
        loss.plot(self.loss_history['training_loss'], color='blue', marker='.')
        loss.plot(self.loss_history['test_loss'], color='green', marker='.')
        loss.set_title('model loss')
        loss.set_xlabel('epoch')
        loss.set_ylabel('loss')
        loss.legend(['training', 'test'])
        plt.show()
        if isinstance(self.neural_net, NeuralNetworkClassifier):
            fig, accuracy = plt.subplots()
            accuracy.plot(self.accuracy_history['training_accuracy'], color='blue', marker='.')
            accuracy.plot(self.accuracy_history['test_accuracy'], color='green', marker='.')
            accuracy.set_title('model accuracy')
            accuracy.set_xlabel('epoch')
            accuracy.set_ylabel('accuracy')
            accuracy.legend(['training', 'test'])
            plt.show()


class NeuralNetwork(BaseEstimator, Layer):

    def __init__(self, layers=(), loss=mean_squared_error, optimizer=GradientDescent, learning_rate=0.01, epochs=100,
                 momentum_type='none', momentum=0.9, batch_size=None, max_f_eval=1000, verbose=False, plot=False):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum_type = momentum_type
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_f_eval = max_f_eval
        self.verbose = verbose
        self.plot = plot

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

    def fit(self, X_train, y_train, X_test=None, y_test=None):

        if y_train.ndim == 1:
            y_train = y_train.reshape((-1, 1))
        if y_test is not None and y_test.ndim == 1:
            y_test = y_test.reshape((-1, 1))

        self._store_meta_info()

        packed_weights_biases = self._pack(*self.params)

        loss = NeuralNetworkLossFunction(self, X_train, y_train, X_test, y_test)
        if issubclass(self.optimizer, LineSearchOptimizer):
            opt = self.optimizer(f=loss, wrt=packed_weights_biases, batch_size=self.batch_size, max_iter=self.epochs,
                                 max_f_eval=self.max_f_eval, verbose=self.verbose).minimize()
            # if opt[2] != 'optimal':
            #     warnings.warn("max_iter or max_f_eval reached and the optimization hasn't converged yet")
        else:
            opt = self.optimizer(f=loss, wrt=packed_weights_biases, batch_size=self.batch_size,
                                 step_rate=self.learning_rate, momentum_type=self.momentum_type,
                                 momentum=self.momentum, max_iter=self.epochs, verbose=self.verbose).minimize()
            # if opt[2] != 'optimal':
            #     warnings.warn("max_iter reached and the optimization hasn't converged yet")
        self._unpack(opt[0])

        if self.plot:
            loss.plot()

        return self


class NeuralNetworkClassifier(ClassifierMixin, NeuralNetwork):

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


class NeuralNetworkRegressor(RegressorMixin, NeuralNetwork):

    def predict(self, X):
        if self.layers[-1].fan_out == 1:  # one target
            return self.forward(X).ravel()
        else:  # multi target
            return self.forward(X)
