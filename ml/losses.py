import autograd.numpy as np
from scipy.special import xlogy

from ml.neural_network.activations import softmax


class LossFunction:

    def function(self, y_pred, y_true):
        raise NotImplementedError

    def delta(self, y_pred, y_true):
        return y_pred - y_true

    def __call__(self, y_pred, y_true):
        return self.function(y_pred, y_true)


class MeanSquaredError(LossFunction):

    def function(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))


class MeanAbsoluteError(LossFunction):

    def function(self, y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))


class BinaryCrossEntropy(LossFunction):
    """Sigmoid Cross Entropy for multi-label classification"""

    def function(self, y_pred, y_true):
        return -np.mean(xlogy(y_true, y_pred) + xlogy(1. - y_true, 1. - y_pred))


class CategoricalCrossEntropy(LossFunction):
    """SoftMax Cross Entropy for multi-class classification"""

    def __init__(self, from_logits=False):
        self.from_logits = from_logits

    def function(self, y_pred, y_true):
        y_pred = softmax(y_pred) if self.from_logits else y_pred
        return -np.mean(np.sum(xlogy(y_true, y_pred), axis=-1))

    def delta(self, y_pred, y_true):
        # according to: https://deepnotes.io/softmax-crossentropy
        grad = softmax(y_pred) if self.from_logits else y_pred
        one_hot_mask = y_true.astype(np.bool)
        grad[one_hot_mask] -= 1.
        return grad / len(grad)


class SparseCategoricalCrossEntropy(LossFunction):
    """Sparse SoftMax Cross Entropy for multi-class classification"""

    def __init__(self, from_logits=False):
        self.from_logits = from_logits

    def function(self, y_pred, y_true):
        sm = softmax(y_pred) if self.from_logits else y_pred
        return -np.mean(np.log(sm[np.arange(sm.shape[0]), y_true.ravel()]))

    def delta(self, y_pred, y_true):
        grad = softmax(y_pred) if self.from_logits else y_pred
        grad[np.arange(grad.shape[0]), y_true.ravel()] -= 1.
        return grad / len(grad)


mean_squared_error = MeanSquaredError()
mean_absolute_error = MeanAbsoluteError()
binary_cross_entropy = BinaryCrossEntropy()
categorical_cross_entropy = CategoricalCrossEntropy()
sparse_categorical_cross_entropy = SparseCategoricalCrossEntropy()
