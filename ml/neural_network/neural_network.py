import warnings

import numpy as np
from sklearn.base import is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics._regression import _check_reg_targets
from sklearn.model_selection import train_test_split
from sklearn.neural_network._stochastic_optimizers import SGDOptimizer, AdamOptimizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import _safe_indexing
from sklearn.utils import column_or_1d
from sklearn.utils import gen_batches
from sklearn.utils import shuffle

from ml.initializers import glorot_uniform, he_uniform, zeros
from ml.learning import Learner
from ml.losses import CrossEntropy, MeanSquaredError
from ml.neural_network.activations import Sigmoid, SoftMax, Linear, ReLU, Activation
from ml.neural_network.losses import NeuralNetworkLossFunction
from optimization.unconstrained.quasi_newton import BFGS


class NeuralNetwork(Learner):
    _estimator_type = "classifier"

    def __init__(self, hidden_layer_sizes, activations, loss=CrossEntropy,
                 optimizer='adam', regularization_type='l2', lmbda=0.0001,
                 batch_size='auto', learning_rate_type='constant',
                 learning_rate=0.001, power_t=0.5, max_iter=200,
                 shuffle=True, tol=1e-4, verbose=False, momentum=0.9,
                 nesterov_momentum=True, early_stopping=True,
                 validation_fraction=0.1, n_iter_no_change=10, max_fun=15000):
        self.activations = activations
        # if a single activation function is given, use it for each hidden layer
        if isinstance(self.activations, Activation):
            self.activation_funcs = [self.activations] * len(hidden_layer_sizes)
        else:
            self.activation_funcs = self.activations
        self.optimizer = optimizer
        self.lmbda = lmbda
        self.batch_size = batch_size
        self.learning_rate_type = learning_rate_type
        self.learning_rate = learning_rate
        self.power_t = power_t
        self.max_iter = max_iter
        self.loss = loss
        self.regularization_type = regularization_type
        self.hidden_layer_sizes = hidden_layer_sizes
        self.shuffle = shuffle
        self.tol = tol
        self.verbose = verbose
        self.momentum = momentum
        self.nesterov_momentum = nesterov_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle must be either True or False, got %s." % self.shuffle)
        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0, got %s." % self.max_iter)
        if self.max_fun <= 0:
            raise ValueError("max_fun must be > 0, got %s." % self.max_fun)
        if self.lmbda < 0.0:
            raise ValueError("alpha must be >= 0, got %s." % self.lmbda)
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0, got %s." % self.learning_rate)
        if self.momentum > 1 or self.momentum < 0:
            raise ValueError("momentum must be >= 0 and <= 1, got %s" % self.momentum)
        if not isinstance(self.nesterov_momentum, bool):
            raise ValueError("nesterov_momentum must be either True or False, got %s." % self.nesterov_momentum)
        if not isinstance(self.early_stopping, bool):
            raise ValueError("early_stopping must be either True or False, got %s." % self.early_stopping)
        if self.validation_fraction < 0 or self.validation_fraction >= 1:
            raise ValueError("validation_fraction must be >= 0 and < 1, got %s" % self.validation_fraction)
        if np.any(np.array(self.hidden_layer_sizes) <= 0):
            raise ValueError("hidden_layer_sizes must be > 0, got %s." % self.hidden_layer_sizes)
        if len(self.activation_funcs) != len(self.hidden_layer_sizes):
            raise ValueError("Number of activation functions cannot be different than the number of hidden layers")
        # if self.eps <= 0.0:
        #     raise ValueError("epsilon must be > 0, got %s." % self.eps)
        if self.n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be > 0, got %s." % self.n_iter_no_change)
        if self.learning_rate_type not in ('constant', 'invscaling', 'adaptive'):
            raise ValueError("learning rate %s is not supported. " % self.learning_rate)
        # if not isinstance(self.optimizer, Optimizer):
        #     raise ValueError("The solver %s is not supported. Expected one of Optimizer subclass" % self.optimizer)

    def _pack(self, weights, bias):
        return np.hstack([l.ravel() for l in weights + bias])

    def _unpack(self, packed_weights_bias):
        for i in range(self.n_layers - 1):
            start, end, shape = self.weights_idx[i]
            self.weights[i] = np.reshape(packed_weights_bias[start:end], shape)
            start, end = self.bias_idx[i]
            self.bias[i] = packed_weights_bias[start:end]

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

    def _initialize(self, y, layer_units):
        # set all attributes, allocate weights etc for first call
        # Initialize parameters
        self.n_iter = 0
        self.t = 0
        self.n_outputs = y.shape[1]

        # Compute the number of layers
        self.n_layers = len(layer_units)

        # output for regression
        if not is_classifier(self):
            self.out_activation = Linear()
        # output for multi class
        elif self._label_binarizer.y_type_ == 'multiclass':
            self.out_activation = SoftMax()
        # output for binary class and multi-label
        else:
            self.out_activation = Sigmoid()

        # Initialize coefficient and intercept layers
        self.weights, self.bias = map(list, zip(
            *[self._init_weights(layer_units[i], layer_units[i + 1], self.activation_funcs[i])
              for i in range(self.n_layers - 2)]))

        # for output layer, use the rule according to the
        # activation function in the previous layer
        weights_init, bias_init = self._init_weights(
            layer_units[self.n_layers - 2], layer_units[self.n_layers - 1], self.activation_funcs[self.n_layers - 3])
        self.weights.append(weights_init)
        self.bias.append(bias_init)

        if self.optimizer in ['sgd', 'adam']:
            self.loss_curve_ = []
            self._no_improvement_count = 0
            if self.early_stopping:
                self.validation_scores_ = []
                self.best_validation_score_ = -np.inf
            else:
                self.best_loss_ = np.inf

    def _init_weights(self, fan_in, fan_out, activation):
        if isinstance(activation, ReLU):
            weights_init = he_uniform(fan_in, fan_out)[0]
        else:
            weights_init = glorot_uniform(fan_in, fan_out)[0]
        bias_init = zeros(fan_out)
        return weights_init, bias_init

    def _fit_lbfgs(self, X, y):
        # store meta information for the parameters
        self.weights_idx = []
        self.bias_idx = []
        start = 0
        # save sizes and indices of coefficients for faster unpacking
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

        # run LBFGS
        packed_weights_bias = self._pack(self.weights, self.bias)

        nn_loss = NeuralNetworkLossFunction(self, self.loss)
        wrt, status = BFGS(f=nn_loss, wrt=packed_weights_bias, batch_size=None,
                           max_iter=self.max_iter, max_f_eval=self.max_fun).minimize()
        self.loss_ = nn_loss.function(wrt, X, y)
        self._unpack(wrt)

    def _fit_stochastic(self, X, y):

        if not hasattr(self, '_optimizer'):
            params = self.weights + self.bias

            if self.optimizer == 'sgd':
                self._optimizer = SGDOptimizer(params, self.learning_rate, self.learning_rate_type,
                                               self.momentum, self.nesterov_momentum, self.power_t)
            elif self.optimizer == 'adam':
                self._optimizer = AdamOptimizer(params, self.learning_rate)

        # early_stopping in partial_fit doesn't make sense
        early_stopping = self.early_stopping
        if early_stopping:
            # don't stratify in multilabel classification
            should_stratify = is_classifier(self) and self.n_outputs == 1
            stratify = y if should_stratify else None
            X, X_val, y, y_val = train_test_split(X, y, test_size=self.validation_fraction, stratify=stratify)
            if is_classifier(self):
                y_val = self._label_binarizer.inverse_transform(y_val)
        else:
            X_val = None
            y_val = None

        n_samples = X.shape[0]
        sample_idx = np.arange(n_samples, dtype=int)

        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            batch_size = np.clip(self.batch_size, 1, n_samples)

        for it in range(self.max_iter):
            if self.shuffle:
                # Only shuffle the sample indices instead of X and y to
                # reduce the memory footprint. These indices will be used
                # to slice the X and y.
                sample_idx = shuffle(sample_idx)

            accumulated_loss = 0.0
            for batch_slice in gen_batches(n_samples, batch_size):
                if self.shuffle:
                    X_batch = _safe_indexing(X, sample_idx[batch_slice])
                    y_batch = y[sample_idx[batch_slice]]
                else:
                    X_batch = X[batch_slice]
                    y_batch = y[batch_slice]

                self.activations[0] = X_batch

                # forward propagate
                self.activations = self.forward(self.activations)
                self.loss.predict = lambda *args: self.activations[-1]  # monkeypatch
                batch_loss = self.loss.function(self.weights, X_batch, y_batch)
                weights_grads, bias_grads = self.backward(
                    X_batch, y_batch, self.activations, self.deltas, self.weights_grads, self.bias_grads)
                accumulated_loss += batch_loss * (batch_slice.stop - batch_slice.start)

                # update weights
                grads = weights_grads + bias_grads
                self._optimizer.update_params(grads)

            self.n_iter += 1
            self.loss_ = accumulated_loss / X.shape[0]

            self.t += n_samples
            self.loss_curve_.append(self.loss_)
            if self.verbose:
                print("Iteration %d, loss = %.8f" % (self.n_iter, self.loss_))

            # update no_improvement_count based on training loss or
            # validation score according to early_stopping
            self._update_no_improvement_count(early_stopping, X_val, y_val)

            # for learning rate that needs to be updated at iteration end
            self._optimizer.iteration_ends(self.t)

            if self._no_improvement_count > self.n_iter_no_change:
                # not better than last `n_iter_no_change` iterations by tol
                # stop or decrease learning rate
                if early_stopping:
                    msg = ("Validation score did not improve more than "
                           "tol=%f for %d consecutive epochs." % (self.tol, self.n_iter_no_change))
                else:
                    msg = ("Training loss did not improve more than tol=%f"
                           " for %d consecutive epochs." % (self.tol, self.n_iter_no_change))

                is_stopping = self._optimizer.trigger_stopping(msg, self.verbose)
                if is_stopping:
                    break
                else:
                    self._no_improvement_count = 0

            if self.n_iter == self.max_iter:
                warnings.warn(
                    "Stochastic Optimizer: Maximum iterations (%d) "
                    "reached and the optimization hasn't converged yet."
                    % self.max_iter, ConvergenceWarning)

        if early_stopping:
            # restore best weights
            self.weights = self._best_weights
            self.bias = self._best_bias

    def _update_no_improvement_count(self, early_stopping, X_val, y_val):
        if early_stopping:
            # compute validation score, use that for stopping
            self.validation_scores_.append(self.score(X_val, y_val))

            if self.verbose:
                print("Validation score: %f" % self.validation_scores_[-1])
            # update best parameters
            # use validation_scores_, not loss_curve_
            # let's hope no-one overloads .score with mse
            last_valid_score = self.validation_scores_[-1]

            if last_valid_score < (self.best_validation_score_ + self.tol):
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0

            if last_valid_score > self.best_validation_score_:
                self.best_validation_score_ = last_valid_score
                self._best_weights = [c.copy() for c in self.weights]
                self._best_bias = [i.copy() for i in self.bias]
        else:
            if self.loss_curve_[-1] > self.best_loss_ - self.tol:
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0
            if self.loss_curve_[-1] < self.best_loss_:
                self.best_loss_ = self.loss_curve_[-1]

    def fit(self, X, y):
        X, y = self._validate_input(X, y)
        n_samples, n_features = X.shape

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs = y.shape[1]

        self.loss = self.loss(X, y, regularization_type=self.regularization_type, lmbda=self.lmbda)

        self.layer_units = ([n_features] + list(self.hidden_layer_sizes) + [self.n_outputs])

        self._initialize(y, self.layer_units)

        # Initialize lists
        self.activations = [X] + [None] * (len(self.layer_units) - 1)
        self.deltas = [None] * (len(self.activations) - 1)

        self.weights_grads = [np.empty((n_fan_in_, n_fan_out_))
                              for n_fan_in_, n_fan_out_ in zip(self.layer_units[:-1], self.layer_units[1:])]

        self.bias_grads = [np.empty(n_fan_out) for n_fan_out in self.layer_units[1:]]

        # run the Stochastic optimization solver
        if self.optimizer in ['sgd', 'adam']:
            self._fit_stochastic(X, y)

        # run the LBFGS solver
        elif self.optimizer == 'lbfgs':
            self._fit_lbfgs(X, y)
        return self

    def _predict(self, X):

        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, '__iter__'):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        layer_units = [X.shape[1]] + hidden_layer_sizes + [self.n_outputs]

        # Initialize layers
        activations = [X]

        for i in range(self.n_layers - 1):
            activations.append(np.empty((X.shape[0], layer_units[i + 1])))
        # forward propagate
        self.forward(activations)
        y_pred = activations[-1]

        return y_pred

    def _validate_input(self, X, y):
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)
        self._label_binarizer = LabelBinarizer()
        self._label_binarizer.fit(y)
        self.classes_ = self._label_binarizer.classes_
        y = self._label_binarizer.transform(y)
        return X, y

    def predict(self, X):
        y_pred = self._predict(X)
        if self.n_outputs == 1:
            y_pred = y_pred.ravel()
        return self._label_binarizer.inverse_transform(y_pred)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class NeuralNetworkReg(NeuralNetwork):
    _estimator_type = "regressor"

    def __init__(self, hidden_layer_sizes, activations, loss=MeanSquaredError,
                 regularization_type='l2', optimizer='adam', lmbda=0.0001,
                 batch_size='auto', learning_rate_type='constant',
                 learning_rate=0.001, power_t=0.5, max_iter=200, shuffle=True, tol=1e-4,
                 verbose=False, momentum=0.9,
                 nesterov_momentum=True, early_stopping=False,
                 validation_fraction=0.1, n_iter_no_change=10, max_fun=15000):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activations=activations, optimizer=optimizer, lmbda=lmbda,
            batch_size=batch_size, learning_rate_type=learning_rate_type,
            learning_rate=learning_rate, power_t=power_t,
            max_iter=max_iter, loss=loss, regularization_type=regularization_type,
            shuffle=shuffle, tol=tol, verbose=verbose, momentum=momentum,
            nesterov_momentum=nesterov_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, max_fun=max_fun)

    def predict(self, X):
        y_pred = self._predict(X)
        if y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred

    def _validate_input(self, X, y):
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)
        return X, y

    def score(self, X, y):
        y_pred = self.predict(X)
        # XXX: Remove the check in 0.23
        y_type, _, _, _ = _check_reg_targets(y, y_pred, None)
        if y_type == 'continuous-multioutput':
            warnings.warn("The default value of multioutput (not exposed in "
                          "score method) will change from 'variance_weighted' "
                          "to 'uniform_average' in 0.23 to keep consistent "
                          "with 'metrics.r2_score'. To specify the default "
                          "value manually and avoid the warning, please "
                          "either call 'metrics.r2_score' directly or make a "
                          "custom scorer with 'metrics.make_scorer' (the "
                          "built-in scorer 'r2' uses "
                          "multioutput='uniform_average').", FutureWarning)
        return r2_score(y, y_pred, multioutput='variance_weighted')


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

    X, y = load_iris(return_X_y=True)
    nn = NeuralNetwork(hidden_layer_sizes=(4, 4),
                       activations=(Sigmoid(), Sigmoid()),
                       optimizer='lbfgs', max_iter=1000,
                       early_stopping=True, verbose=True).fit(X, y)
    pred = nn.predict(X)
    print(pred)
    print(accuracy_score(y, pred))


    # from sklearn.datasets import load_boston
    # from sklearn.metrics import mean_squared_error
    #
    # X, y = load_boston(return_X_y=True)
    # nn = NeuralNetworkReg(hidden_layer_sizes=(5, 3),
    #                       activations=(Tanh(), Tanh()),
    #                       optimizer='lbfgs', max_iter=1000).fit(X, y)
    # print(mean_squared_error(y, nn.predict(X)))
    #
    # from sklearn.neural_network import MLPRegressor
    #
    # nnr = MLPRegressor(hidden_layer_sizes=(5, 3),
    #                    activation='tanh',
    #                    solver='lbfgs', max_iter=1000).fit(X, y)
    # print(mean_squared_error(y, nn.predict(X)))

    def mean_euclidean_error(y_true, y_pred):
        return np.sum(np.linalg.norm(t - o) for t, o in zip(y_true, y_pred)) / y_true.shape[0]


    ml_cup = np.delete(np.genfromtxt('../data/ML-CUP19/ML-CUP19-TR.csv', delimiter=','), 0, 1)
    X, y = ml_cup[:, :-2], ml_cup[:, -2:]

    nn = NeuralNetworkReg(hidden_layer_sizes=(20, 20),
                          activations=(Sigmoid(), Sigmoid()),
                          optimizer='lbfgs', max_iter=15000).fit(X, y)
    pred = nn.predict(X)
    print(mean_squared_error(y, pred))
    print(mean_euclidean_error(y, pred))
