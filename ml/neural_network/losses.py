import autograd.numpy as np
from scipy.special import xlogy


class Loss:

    def function(self, y_pred, y_true):
        raise NotImplementedError

    def jacobian(self, y_pred, y_true):
        return y_pred - y_true

    def __call__(self, y_pred, y_true):
        return self.function(y_pred, y_true)


class MeanSquaredError(Loss):

    def function(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return np.mean(np.square(y_pred - y_true))


class MeanAbsoluteError(Loss):

    def function(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return np.mean(np.abs(y_pred - y_true))


class SparseCategoricalCrossEntropy(Loss):
    """Multi-class (single-label) classification with softmax output layer"""

    def function(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return -np.mean(np.log(y_pred[np.arange(y_pred.shape[0]), y_true.ravel()]))

    def delta(self, y_pred, y_true):
        y_pred[np.arange(y_pred.shape[0]), y_true.ravel()] -= 1.
        return y_pred / len(y_pred)


class CategoricalCrossEntropy(Loss):
    """Multi-class (single-label) classification with softmax
    output layer and one-hot encoded target data"""

    def function(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return -np.mean(xlogy(y_true, y_pred))

    def delta(self, y_pred, y_true):
        # according to: https://deepnotes.io/softmax-crossentropy
        one_hot_mask = y_true.astype(np.bool)
        y_pred[one_hot_mask] -= 1.
        return y_pred / len(y_pred)


class BinaryCrossEntropy(Loss):
    """Multi-class or multi-label classification or regression
     between 0 and 1 with sigmoid output layer"""

    def function(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return -np.mean(xlogy(y_true, y_pred) + xlogy(1. - y_true, 1. - y_pred))


mean_squared_error = MeanSquaredError()
mean_absolute_error = MeanAbsoluteError()
sparse_categorical_cross_entropy = SparseCategoricalCrossEntropy()
categorical_cross_entropy = CategoricalCrossEntropy()
binary_cross_entropy = BinaryCrossEntropy()
