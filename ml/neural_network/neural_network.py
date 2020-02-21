import numpy as np
from sklearn.preprocessing import LabelBinarizer

from ml.initializers import glorot_uniform, he_uniform, zeros
from ml.learning import Learner
from ml.neural_network.activations import Sigmoid, SoftMax, Linear, ReLU
from ml.neural_network.losses import NeuralNetworkLossFunction
from optimization.optimizer import LineSearchOptimizer


class NeuralNetwork(Learner):

    def __init__(self, hidden_layer_sizes, activations):
        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError('hidden_layer_sizes must be > 0')
        if len(activations) != len(hidden_layer_sizes):
            raise ValueError('the number of activation functions cannot be different than the number of hidden layers')
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.activation_funcs = list(activation() for activation in activations)

    def _pack(self, weights, biases):
        return np.hstack([l.ravel() for l in weights + biases])

    def _unpack(self, packed_weights_bias):
        for i in range(self.n_layers - 1):
            start, end, shape = self.weight_idx[i]
            self.weights[i] = np.reshape(packed_weights_bias[start:end], shape)
            start, end = self.bias_idx[i]
            self.biases[i] = packed_weights_bias[start:end]

    def _initialize(self, y, layer_units):
        # set all attributes, allocate weights etc for first call
        # initialize parameters
        self.n_iter = 0
        self.t = 0
        self.n_outputs = y.shape[1]

        # compute the number of layers
        self.n_layers = len(layer_units)

        # output for regression
        if self.task == 'regression':
            self.out_activation = Linear()
        # output for multi class
        elif self._label_binarizer.y_type_ == 'multiclass':
            self.out_activation = SoftMax()
        # output for binary class and multi-label
        else:
            self.out_activation = Sigmoid()

        # initialize weights and bias layers
        self.weights, self.biases = map(list, zip(
            *[self._init_weights(layer_units[i], layer_units[i + 1], self.activation_funcs[i])
              for i in range(self.n_layers - 2)]))

        # for output layer, use the rule according to the
        # activation function in the previous layer
        weight_init, bias_init = self._init_weights(
            layer_units[self.n_layers - 2], layer_units[self.n_layers - 1], self.activation_funcs[self.n_layers - 3])
        self.weights.append(weight_init)
        self.biases.append(bias_init)

    def _init_weights(self, fan_in, fan_out, activation):
        if isinstance(activation, ReLU):
            weight_init = he_uniform(fan_in, fan_out)[0]
        else:
            weight_init = glorot_uniform(fan_in, fan_out)[0]
        bias_init = zeros(fan_out)
        return weight_init, bias_init

    def forward(self, activations):
        for i in range(self.n_layers - 1):
            activations[i + 1] = np.dot(activations[i], self.weights[i])
            activations[i + 1] += self.biases[i]
            # for the hidden layers
            if (i + 1) != (self.n_layers - 1):
                activations[i + 1] = self.activation_funcs[i].function(activations[i + 1])
        # for the last layer
        activations[i + 1] = self.out_activation.function(activations[i + 1])
        return activations

    def _compute_loss_grad(self, layer, n_samples, activations, deltas, weight_grads, bias_grads):
        weight_grads[layer] = np.dot(activations[layer].T, deltas[layer])
        weight_grads[layer] += (self.loss.lmbda * self.weights[layer])
        weight_grads[layer] /= n_samples
        bias_grads[layer] = np.mean(deltas[layer], 0)
        return weight_grads, bias_grads

    def backward(self, X, y, activations, deltas, weight_grads, bias_grads):
        n_samples = X.shape[0]
        # backward propagate
        last = self.n_layers - 2
        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy,
        # softmax and categorical cross entropy, and
        # identity with squared loss
        deltas[last] = activations[-1] - y
        # compute gradient for the last layer
        weight_grads, bias_grads = self._compute_loss_grad(
            last, n_samples, activations, deltas, weight_grads, bias_grads)
        # iterate over the hidden layers
        for i in range(self.n_layers - 2, 0, -1):
            deltas[i - 1] = np.dot(deltas[i], self.weights[i].T)
            deltas[i - 1] *= self.activation_funcs[i - 1].derivative(activations[i])
            weight_grads, bias_grads = self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, weight_grads, bias_grads)
        return weight_grads, bias_grads

    def fit(self, X, y, loss, optimizer, regularization_type='l2', lmbda=0.0001,
            batch_size=None, verbose=False, plot=False):

        try:
            self._label_binarizer = LabelBinarizer()
            self._label_binarizer.fit(y)
            self.classes_ = self._label_binarizer.classes_
            y = self._label_binarizer.transform(y)
            self.task = 'classification'
        except:
            self.task = 'regression'

        n_samples, n_features = X.shape

        # ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs = y.shape[1]

        loss = loss(X, y, regularization_type=regularization_type, lmbda=lmbda)

        self.layer_units = ([n_features] + list(self.hidden_layer_sizes) + [self.n_outputs])

        self._initialize(y, self.layer_units)

        # initialize lists
        self.activations = [X] + [None] * (len(self.layer_units) - 1)
        self.deltas = [None] * (len(self.activations) - 1)

        self.weight_grads = [np.empty((n_fan_in, n_fan_out))
                             for n_fan_in, n_fan_out in zip(self.layer_units[:-1], self.layer_units[1:])]

        self.bias_grads = [np.empty(n_fan_out) for n_fan_out in self.layer_units[1:]]

        # store meta information for the parameters
        self.weight_idx = []
        self.bias_idx = []
        start = 0
        # save sizes and indices of weights for faster unpacking
        for i in range(self.n_layers - 1):
            n_fan_in, n_fan_out = self.layer_units[i], self.layer_units[i + 1]
            end = start + (n_fan_in * n_fan_out)
            self.weight_idx.append((start, end, (n_fan_in, n_fan_out)))
            start = end
        # save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers - 1):
            end = start + self.layer_units[i + 1]
            self.bias_idx.append((start, end))
            start = end

        packed_weights_bias = self._pack(self.weights, self.biases)

        self.loss = NeuralNetworkLossFunction(self, loss)
        if issubclass(optimizer, LineSearchOptimizer):
            wrt = optimizer(f=self.loss, wrt=packed_weights_bias, batch_size=batch_size,
                            max_iter=1000, max_f_eval=15000).minimize()[0]
        else:
            wrt = optimizer(f=self.loss, wrt=packed_weights_bias, batch_size=batch_size, max_iter=1000).minimize()[0]
        self.loss_ = self.loss.function(wrt, X, y)  # TODO and if we use batch (?)
        self._unpack(wrt)
        return self

    def predict(self, X):
        layer_units = [X.shape[1]] + self.hidden_layer_sizes + [self.n_outputs]
        # initialize layers
        activations = [X]
        for i in range(self.n_layers - 1):
            activations.append(np.empty((X.shape[0], layer_units[i + 1])))
        # forward propagate
        self.forward(activations)
        y_pred = activations[-1]
        if self.task == 'classification':
            if self.n_outputs == 1:
                y_pred = y_pred.ravel()
            return self._label_binarizer.inverse_transform(y_pred)
        else:
            if y_pred.shape[1] == 1:
                return y_pred.ravel()
            return y_pred
