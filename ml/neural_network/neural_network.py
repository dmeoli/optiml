import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import column_or_1d

from ml.initializers import glorot_uniform, he_uniform, zeros
from ml.learning import Learner
from ml.losses import CrossEntropy, MeanSquaredError
from ml.neural_network.activations import Sigmoid, SoftMax, Linear, ReLU
from ml.neural_network.losses import NeuralNetworkLossFunction
from optimization.optimizer import LineSearchOptimizer
from optimization.unconstrained.quasi_newton import BFGS


class NeuralNetwork(Learner):

    def __init__(self, hidden_layer_sizes, activations, loss=CrossEntropy, optimizer=BFGS, regularization_type='l2',
                 lmbda=0.0001, batch_size=None, shuffle=True, early_stopping=True, verbose=False, plot=False):
        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError('hidden_layer_sizes must be > 0')
        if len(activations) != len(hidden_layer_sizes):
            raise ValueError('the number of activation functions cannot be different than the number of hidden layers')
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.activation_funcs = list(activation() for activation in activations)
        self.loss = loss
        self.optimizer = optimizer
        self.regularization_type = regularization_type
        self.lmbda = lmbda
        self.batch_size = batch_size
        if not isinstance(shuffle, bool):
            raise ValueError('shuffle must be either True or False')
        self.shuffle = shuffle
        if not isinstance(early_stopping, bool):
            raise ValueError('early_stopping must be either True or False')
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.plot = plot

    def _pack(self, weights, bias):
        return np.hstack([l.ravel() for l in weights + bias])

    def _unpack(self, packed_weights_bias):
        for i in range(self.n_layers - 1):
            start, end, shape = self.weights_idx[i]
            self.weights[i] = np.reshape(packed_weights_bias[start:end], shape)
            start, end = self.bias_idx[i]
            self.bias[i] = packed_weights_bias[start:end]

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
        self.weights, self.bias = map(list, zip(
            *[self._init_weights(layer_units[i], layer_units[i + 1], self.activation_funcs[i])
              for i in range(self.n_layers - 2)]))

        # for output layer, use the rule according to the
        # activation function in the previous layer
        weights_init, bias_init = self._init_weights(
            layer_units[self.n_layers - 2], layer_units[self.n_layers - 1], self.activation_funcs[self.n_layers - 3])
        self.weights.append(weights_init)
        self.bias.append(bias_init)

    def _init_weights(self, fan_in, fan_out, activation):
        if isinstance(activation, ReLU):
            weights_init = he_uniform(fan_in, fan_out)[0]
        else:
            weights_init = glorot_uniform(fan_in, fan_out)[0]
        bias_init = zeros(fan_out)
        return weights_init, bias_init

    def forward(self, activations):
        for i in range(self.n_layers - 1):
            activations[i + 1] = np.dot(activations[i], self.weights[i])
            activations[i + 1] += self.bias[i]
            # for the hidden layers
            if (i + 1) != (self.n_layers - 1):
                activations[i + 1] = self.activation_funcs[i].function(activations[i + 1])
        # for the last layer
        activations[i + 1] = self.out_activation.function(activations[i + 1])
        return activations

    def _compute_loss_grad(self, layer, n_samples, activations, deltas, weights_grads, bias_grads):
        weights_grads[layer] = np.dot(activations[layer].T, deltas[layer])
        weights_grads[layer] += (self.lmbda * self.weights[layer])
        weights_grads[layer] /= n_samples
        bias_grads[layer] = np.mean(deltas[layer], 0)
        return weights_grads, bias_grads

    def backward(self, X, y, activations, deltas, weights_grads, bias_grads):
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
        weights_grads, bias_grads = self._compute_loss_grad(
            last, n_samples, activations, deltas, weights_grads, bias_grads)
        # iterate over the hidden layers
        for i in range(self.n_layers - 2, 0, -1):
            deltas[i - 1] = np.dot(deltas[i], self.weights[i].T)
            deltas[i - 1] *= self.activation_funcs[i - 1].derivative(activations[i])
            weights_grads, bias_grads = self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, weights_grads, bias_grads)
        return weights_grads, bias_grads

    def fit(self, X, y):
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)

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

        self.loss = self.loss(X, y, regularization_type=self.regularization_type, lmbda=self.lmbda)

        self.layer_units = ([n_features] + list(self.hidden_layer_sizes) + [self.n_outputs])

        self._initialize(y, self.layer_units)

        # initialize lists
        self.activations = [X] + [None] * (len(self.layer_units) - 1)
        self.deltas = [None] * (len(self.activations) - 1)

        self.weights_grads = [np.empty((n_fan_in_, n_fan_out_))
                              for n_fan_in_, n_fan_out_ in zip(self.layer_units[:-1], self.layer_units[1:])]

        self.bias_grads = [np.empty(n_fan_out) for n_fan_out in self.layer_units[1:]]

        # store meta information for the parameters
        self.weights_idx = []
        self.bias_idx = []
        start = 0
        # save sizes and indices of weights for faster unpacking
        for i in range(self.n_layers - 1):
            n_fan_in, n_fan_out = self.layer_units[i], self.layer_units[i + 1]
            end = start + (n_fan_in * n_fan_out)
            self.weights_idx.append((start, end, (n_fan_in, n_fan_out)))
            start = end
        # save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers - 1):
            end = start + self.layer_units[i + 1]
            self.bias_idx.append((start, end))
            start = end

        packed_weights_bias = self._pack(self.weights, self.bias)

        nn_loss = NeuralNetworkLossFunction(self, self.loss)
        if issubclass(self.optimizer, LineSearchOptimizer):
            wrt = self.optimizer(f=nn_loss, wrt=packed_weights_bias, batch_size=self.batch_size,
                                 max_iter=1000, max_f_eval=15000).minimize()[0]
        else:
            wrt = self.optimizer(f=nn_loss, wrt=packed_weights_bias, batch_size=self.batch_size,
                                 max_iter=1000).minimize()[0]
        self.loss_ = nn_loss.function(wrt, X, y)  # TODO and if we use batch (?)
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


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    nn = NeuralNetwork(hidden_layer_sizes=(4, 4), activations=(Sigmoid, Sigmoid),
                       loss=CrossEntropy, optimizer=BFGS).fit(X, y)
    pred = nn.predict(X)
    print(pred)
    print(accuracy_score(y, pred))


    def mean_euclidean_error(y_true, y_pred):
        assert y_true.shape == y_pred.shape
        return np.sum([np.linalg.norm(t - p) for t, p in zip(y_true, y_pred)]) / y_true.shape[0]


    ml_cup_train = np.delete(np.genfromtxt('../data/ML-CUP19/ML-CUP19-TR.csv', delimiter=','), 0, 1)
    X_train, y_train = ml_cup_train[:, :-2], ml_cup_train[:, -2:]

    X_exam = np.delete(np.genfromtxt('../data/ML-CUP19/ML-CUP19-TS.csv', delimiter=','), 0, 1)

    nn = NeuralNetwork(hidden_layer_sizes=(20, 20), activations=(Sigmoid, Sigmoid),
                       loss=MeanSquaredError, optimizer=BFGS).fit(X_train, y_train)
    pred = nn.predict(X_train)
    print(mean_squared_error(y_train, pred))
    print(mean_euclidean_error(y_train, pred))
