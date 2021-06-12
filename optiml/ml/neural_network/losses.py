from abc import ABC

import autograd.numpy as np
from scipy.special import xlogy

from .activations import Linear
from .layers import ParamLayer
from .regularizers import L2
from ...opti import OptimizationFunction


class NeuralNetworkLoss(OptimizationFunction, ABC):

    def __init__(self, neural_net, X, y):
        super(NeuralNetworkLoss, self).__init__(X.shape[1])
        self.neural_net = neural_net
        self.X = X
        self.y = y

    def args(self):
        return self.X, self.y

    def loss(self, y_pred, y_true):
        raise NotImplementedError

    def delta(self, y_pred, y_true):
        return y_pred - y_true

    def function(self, packed_coef_inter, X_batch=None, y_batch=None):
        if X_batch is None:
            X_batch = self.X
        if y_batch is None:
            y_batch = self.y

        self.neural_net._unpack(packed_coef_inter)

        n_samples = X_batch.shape[0]
        coef_regs = np.sum(layer.coef_reg(layer.coef_) for layer in self.neural_net.layers
                           if isinstance(layer, ParamLayer)) / (2 * n_samples)
        inter_regs = np.sum(layer.inter_reg(layer.inter_) for layer in self.neural_net.layers
                            if isinstance(layer, ParamLayer) and layer.fit_intercept) / (2 * n_samples)
        return 1 / (2 * n_samples) * self.loss(self.neural_net.forward(X_batch), y_batch) + coef_regs + inter_regs

    def jacobian(self, packed_coef_inter, X_batch=None, y_batch=None):
        if X_batch is None:
            X_batch = self.X
        if y_batch is None:
            y_batch = self.y

        self.neural_net._unpack(packed_coef_inter)

        n_samples = X_batch.shape[0]
        delta = 1 / n_samples * self.delta(self.neural_net.forward(X_batch), y_batch)
        return self.neural_net._pack(*self.neural_net.backward(delta))


class MeanSquaredError(NeuralNetworkLoss):

    def x_star(self):
        if (len(self.neural_net.layers) == 1 and
                isinstance(self.neural_net.layers[-1].activation, Linear) and
                isinstance(self.neural_net.layers[-1].coef_reg, L2) and
                not self.neural_net.layers[-1].fit_intercept):
            if not hasattr(self, 'x_opt'):
                if self.neural_net.layers[-1].coef_reg.lmbda == 0.:
                    self.x_opt = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
                else:
                    self.x_opt = np.linalg.inv(self.X.T.dot(self.X) + np.eye(self.ndim) *
                                               self.neural_net.layers[-1].coef_reg.lmbda).dot(self.X.T).dot(self.y)
            return self.x_opt
        return np.full(fill_value=np.nan, shape=self.ndim)

    def f_star(self):
        if not np.isnan(self.x_star()).all():
            return self.function(self.x_star())
        return np.inf

    def loss(self, y_pred, y_true):
        return np.sum(np.square(y_pred - y_true))


class MeanAbsoluteError(NeuralNetworkLoss):

    def loss(self, y_pred, y_true):
        return np.sum(np.abs(y_pred - y_true))

    def delta(self, y_pred, y_true):
        return np.sign(y_pred - y_true)


class BinaryCrossEntropy(NeuralNetworkLoss):
    """Binary Cross-Entropy aka Sigmoid Cross-Entropy loss
    function for binary and multi-label classification
    or regression between 0 and 1 with sigmoid output layer"""

    def loss(self, y_pred, y_true):
        return -np.sum(xlogy(y_true, y_pred) + xlogy(1. - y_true, 1. - y_pred))


class CategoricalCrossEntropy(NeuralNetworkLoss):
    """Categorical Cross-Entropy loss function for multi-class (single-label)
    classification with softmax output layer and one-hot encoded target data"""

    def loss(self, y_pred, y_true):
        return -np.sum(xlogy(y_true, y_pred))

    def delta(self, y_pred, y_true):
        # according to: https://deepnotes.io/softmax-crossentropy
        one_hot_mask = y_true.astype(bool)
        y_pred[one_hot_mask] -= 1.
        return y_pred


class SparseCategoricalCrossEntropy(NeuralNetworkLoss):
    """Sparse Categorical Cross-Entropy loss function for multi-class
    (single-label) classification with softmax output layer"""

    def loss(self, y_pred, y_true):
        assert y_pred.shape[0] == y_true.shape[0]
        return -np.sum(np.log(y_pred[np.arange(y_pred.shape[0]), y_true.astype(int).ravel()]))

    def delta(self, y_pred, y_true):
        y_pred[np.arange(y_pred.shape[0]), y_true.astype(int).ravel()] -= 1.
        return y_pred


mean_squared_error = MeanSquaredError
mean_absolute_error = MeanAbsoluteError
binary_cross_entropy = BinaryCrossEntropy
categorical_cross_entropy = CategoricalCrossEntropy
sparse_categorical_cross_entropy = SparseCategoricalCrossEntropy
