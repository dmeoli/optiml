import autograd.numpy as np
from scipy.special import xlogy

from .activations import Linear
from .layers import ParamLayer
from .regularizers import L2
from ...optimization import OptimizationFunction


class LossFunction(OptimizationFunction):

    def __init__(self, neural_net, X, y):
        super().__init__(X.shape[1])
        self.neural_net = neural_net
        self.X = X
        self.y = y

    def f_star(self):
        if self.x_star() is not np.nan:
            return self.function(self.x_star())
        return super().f_star()

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

        coef_regs = np.sum(layer.coef_reg(layer.coef_) for layer in self.neural_net.layers
                           if isinstance(layer, ParamLayer)) / X_batch.shape[0]
        inter_regs = np.sum(layer.inter_reg(layer.inter_) for layer in self.neural_net.layers
                            if isinstance(layer, ParamLayer) and layer.fit_intercept) / X_batch.shape[0]
        return self.loss(self.neural_net.forward(X_batch), y_batch) + coef_regs + inter_regs

    def jacobian(self, packed_coef_inter, X_batch=None, y_batch=None):
        if X_batch is None:
            X_batch = self.X
        if y_batch is None:
            y_batch = self.y

        delta = self.delta(self.neural_net.forward(X_batch), y_batch)
        return self.neural_net._pack(*self.neural_net.backward(delta))

    def __call__(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


class MeanSquaredError(LossFunction):

    def x_star(self):
        if (len(self.neural_net.layers) == 1 and
                isinstance(self.neural_net.layers[-1].activation, Linear) and
                isinstance(self.neural_net.layers[-1].coef_reg, L2) and
                not self.neural_net.layers[-1].fit_intercept):
            if not hasattr(self, 'x_opt'):
                if self.neural_net.layers[-1].coef_reg.lmbda == 0.:
                    self.x_opt = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
                else:
                    self.x_opt = np.linalg.inv(self.X.T.dot(self.X) + np.identity(self.ndim) *
                                               self.neural_net.layers[-1].coef_reg.lmbda).dot(self.X.T).dot(self.y)
            return self.x_opt

    def loss(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return np.mean(np.square(y_pred - y_true))


class BinaryCrossEntropy(LossFunction):
    """Binary Cross-Entropy aka Sigmoid Cross-Entropy loss
    function for binary (possibly multi-label) classification
    or regression between 0 and 1 with sigmoid output layer"""

    def loss(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return -np.mean(xlogy(y_true, y_pred) + xlogy(1. - y_true, 1. - y_pred))


class CategoricalCrossEntropy(LossFunction):
    """Categorical Cross-Entropy loss function for multi-class (single-label)
    classification with softmax output layer and one-hot encoded target data"""

    def loss(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return -np.mean(xlogy(y_true, y_pred))

    def delta(self, y_pred, y_true):
        # according to: https://deepnotes.io/softmax-crossentropy
        one_hot_mask = y_true.astype(np.bool)
        y_pred[one_hot_mask] -= 1.
        return y_pred / len(y_pred)


class SparseCategoricalCrossEntropy(LossFunction):
    """Sparse Categorical Cross-Entropy loss function for multi-class
    (single-label) classification with softmax output layer"""

    def loss(self, y_pred, y_true):
        assert y_pred.shape[0] == y_true.shape[0]
        return -np.mean(np.log(y_pred[np.arange(y_pred.shape[0]), y_true.astype(np.int32).ravel()]))

    def delta(self, y_pred, y_true):
        y_pred[np.arange(y_pred.shape[0]), y_true.astype(np.int32).ravel()] -= 1.
        return y_pred / len(y_pred)


mean_squared_error = MeanSquaredError
binary_cross_entropy = BinaryCrossEntropy
categorical_cross_entropy = CategoricalCrossEntropy
sparse_categorical_cross_entropy = SparseCategoricalCrossEntropy
