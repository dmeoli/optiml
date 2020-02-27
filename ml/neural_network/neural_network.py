import autograd.numpy as np
from sklearn.preprocessing import LabelBinarizer

from ml.learning import Learner
from ml.losses import mean_squared_error
from ml.neural_network.activations import Linear
from ml.neural_network.layers import Layer
from ml.regularizers import l2
from optimization.optimization_function import OptimizationFunction
from optimization.optimizer import LineSearchOptimizer
from optimization.unconstrained.gradient_descent import GradientDescent


class NeuralNetworkLossFunction(OptimizationFunction):

    def __init__(self, X, y, neural_net, loss, regularizer=l2, lmbda=0.01):
        super().__init__(X.shape[1])
        self.X = X
        self.y = y
        self.neural_net = neural_net
        self.loss = loss
        self.regularizer = regularizer
        self.lmbda = lmbda

    def x_star(self):
        if self.loss is mean_squared_error:
            return np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)  # or np.linalg.lstsq(X, y)[0]

    def f_star(self):
        if self.x_star() is not None:
            return self.loss(self.neural_net.forward(self.X), self.y)
        return super().f_star()

    def args(self):
        return self.X, self.y

    def function(self, packed_weights_biases, X, y):
        self.neural_net._unpack(packed_weights_biases)
        return (self.loss(self.neural_net.forward(X), y) +
                self.regularizer(packed_weights_biases, self.lmbda) / X.shape[0])

    def jacobian(self, packed_weights_biases, X, y):
        return self.neural_net._pack(*self.neural_net.backward(self.delta(self.neural_net.forward(X), y)))

    def hessian(self, packed_weights_biases, X, y):
        return super().hessian(packed_weights_biases)

    def delta(self, y_pred, y_true):
        return y_pred - y_true


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
            weights_grads.append(grads['w'])
            biases_grads.append(grads['b'])
        return weights_grads[::-1], biases_grads[::-1]

    @property
    def params(self):
        return [layer.w for layer in self.layers], [layer.b for layer in self.layers]

    def _pack(self, weights, biases):
        return np.hstack([w.ravel() for w in weights + biases])

    def _unpack(self, packed_weights_biases):
        for i, layer in enumerate(self.layers):
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
            fan_in, fan_out = layer.fan_in, layer.fan_out
            end = start + (fan_in * fan_out)
            self.weights_idx.append((start, end, (fan_in, fan_out)))
            start = end
        # save sizes and indices of biases for faster unpacking
        for layer in self.layers:
            end = start + layer.fan_out
            self.biases_idx.append((start, end))
            start = end

    def fit(self, X, y, loss, optimizer=GradientDescent, learning_rate=0.01, epochs=100, batch_size=None,
            regularizer=l2, lmbda=0.01, max_f_eval=15000, verbose=False):
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        if isinstance(self.layers[-1]._a, Linear):
            self.task = 'regression'
        else:
            self.task = 'classification'
            self.lb = LabelBinarizer().fit(y)
            y = self.lb.transform(y)

        self._store_meta_info()

        packed_weights_biases = self._pack(*self.params)

        loss = NeuralNetworkLossFunction(X, y, self, loss, regularizer, lmbda)
        if issubclass(optimizer, LineSearchOptimizer):
            wrt = optimizer(f=loss, wrt=packed_weights_biases, batch_size=batch_size,
                            max_iter=epochs, max_f_eval=max_f_eval, verbose=verbose).minimize()[0]
        else:
            wrt = optimizer(f=loss, wrt=packed_weights_biases, step_rate=learning_rate,
                            batch_size=batch_size, max_iter=epochs, verbose=verbose).minimize()[0]
        self._unpack(wrt)

        return self

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1) if self.task is 'classification' else self.forward(X)
