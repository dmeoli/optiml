import warnings

import autograd.numpy as np
from matplotlib import pyplot as plt

from ml.initializers import compute_fans
from ml.learning import Learner
from ml.neural_network.activations import Linear
from ml.neural_network.layers import Layer, ParamLayer
from optimization.optimization_function import OptimizationFunction
from optimization.optimizer import LineSearchOptimizer
from optimization.unconstrained.gradient_descent import GradientDescent
from utils import to_categorical

plt.style.use('ggplot')


class NeuralNetworkLossFunction(OptimizationFunction):

    def __init__(self, X, y, neural_net, loss):
        super().__init__(X.shape[1])
        self.X = X
        self.y = y
        self.neural_net = neural_net
        self.loss = loss

    def args(self):
        return self.X, self.y

    def function(self, packed_weights_biases, X, y):
        self.neural_net._unpack(packed_weights_biases)
        self.y_pred = self.neural_net.forward(X)
        return self.loss(self.y_pred, y) + np.sum(layer.w_reg(layer.w) + layer.b_reg(layer.b)
                                                  for layer in self.neural_net.layers) / X.shape[0]

    def jacobian(self, packed_weights_biases, X, y):
        return self.neural_net._pack(*self.neural_net.backward(self.delta(y)))

    def delta(self, y_true):
        return self.y_pred - y_true

    def plot(self, epochs, loss_history):
        fig, ax = plt.subplots()
        ax.plot(range(epochs), loss_history, 'b.', alpha=0.2)
        ax.set_title('model loss')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend(['train'])
        plt.show()


class NeuralNetwork(Layer, Learner):

    def __init__(self, *layers):
        assert isinstance(layers, (list, tuple))
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X.data

    def backward(self, delta):
        weights_grads = []
        biases_grads = []
        # back propagate
        last_layer = self.layers[-1]
        last_layer.data_vars['out'].set_error(delta)
        for layer in self.layers[::-1]:
            grads = layer.backward()
            if isinstance(layer, ParamLayer):
                weights_grads.append(grads['dw'] + layer.w_reg.lmbda * layer.w)
                if layer.use_bias:
                    biases_grads.append(grads['db'] + layer.b_reg.lmbda * layer.b)
        return weights_grads[::-1], biases_grads[::-1]

    @property
    def params(self):
        return ([layer.w for layer in self.layers if isinstance(layer, ParamLayer)],
                [layer.b for layer in self.layers if isinstance(layer, ParamLayer) and layer.use_bias])

    def _pack(self, weights, biases):
        return np.hstack([w.ravel() for w in weights + biases])

    def _unpack(self, packed_weights_biases):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ParamLayer):
                start, end, shape = self.weights_idx[i]
                layer.w = np.reshape(packed_weights_biases[start:end], shape)
                start, end = self.biases_idx[i]
                layer.b = packed_weights_biases[start:end]

    def _store_meta_info(self):
        # store meta information for the parameters
        self.weights_idx = []
        self.biases_idx = []
        start = 0
        # save sizes and indices of weights for faster unpacking
        for layer in self.layers:
            if isinstance(layer, ParamLayer):
                fan_in, fan_out = compute_fans(layer.w_shape)
                end = start + (fan_in * fan_out)
                self.weights_idx.append((start, end, (fan_in, fan_out)))
                start = end
        # save sizes and indices of biases for faster unpacking
        for layer in self.layers:
            if isinstance(layer, ParamLayer):
                fan_in, fan_out = compute_fans(layer.w_shape)
                end = start + fan_out
                self.biases_idx.append((start, end))
                start = end

    def fit(self, X, y, loss, optimizer=GradientDescent, learning_rate=0.01, momentum_type='none', momentum=0.9,
            epochs=100, batch_size=None, k_folds=0, max_f_eval=1000, early_stopping=True, verbose=False, plot=False):
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        if isinstance(self.layers[-1]._a, Linear):
            self.task = 'regression'
        else:
            self.task = 'classification'
            y = to_categorical(y)

        self._store_meta_info()

        packed_weights_biases = self._pack(*self.params)

        loss = NeuralNetworkLossFunction(X, y, self, loss)
        if issubclass(optimizer, LineSearchOptimizer):
            opt = optimizer(f=loss, wrt=packed_weights_biases, batch_size=batch_size,
                            max_iter=epochs, max_f_eval=max_f_eval, verbose=verbose).minimize()
            if opt[2] is not 'optimal':
                warnings.warn("max_iter or max_f_eval reached and the optimization hasn't converged yet")
        else:
            opt = optimizer(f=loss, wrt=packed_weights_biases, step_rate=learning_rate, momentum_type=momentum_type,
                            momentum=momentum, batch_size=batch_size, max_iter=epochs, verbose=verbose).minimize()
            if opt[2] is not 'optimal':
                warnings.warn("max_iter reached and the optimization hasn't converged yet")
        self._unpack(opt[0])

        if plot:
            loss.plot(epochs, opt[1])

        return self

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1) if self.task is 'classification' else self.forward(X)
