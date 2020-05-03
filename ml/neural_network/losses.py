import autograd.numpy as np
from scipy.special import xlogy


class Loss:

    def __init__(self, x_min=-10, x_max=10, y_min=-10, y_max=10):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def function(self, y_pred, y_true):
        raise NotImplementedError

    def jacobian(self, y_pred, y_true):
        return y_pred - y_true

    def __call__(self, y_pred, y_true):
        return self.function(y_pred, y_true)


class MeanSquaredError(Loss):

    def function(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return 0.5 * np.mean(np.square(y_pred - y_true))


class CategoricalCrossEntropy(Loss):
    """Categorical Cross-Entropy loss function for multi-class (single-label)
    classification with softmax output layer and one-hot encoded target data"""

    def function(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return -np.mean(xlogy(y_true, y_pred))

    def jacobian(self, y_pred, y_true):
        # according to: https://deepnotes.io/softmax-crossentropy
        one_hot_mask = y_true.astype(np.bool)
        y_pred[one_hot_mask] -= 1.
        return y_pred / len(y_pred)


class SparseCategoricalCrossEntropy(Loss):
    """Sparse Categorical Cross-Entropy loss function for multi-class
    (single-label) classification with softmax output layer"""

    def function(self, y_pred, y_true):
        assert y_pred.shape[0] == y_true.shape[0]
        return -np.mean(np.log(y_pred[np.arange(y_pred.shape[0]), y_true.astype(np.int32).ravel()]))

    def jacobian(self, y_pred, y_true):
        y_pred[np.arange(y_pred.shape[0]), y_true.astype(np.int32).ravel()] -= 1.
        return y_pred / len(y_pred)


class BinaryCrossEntropy(Loss):
    """Binary Cross-Entropy aka Sigmoid Cross-Entropy loss
    function for binary (possibly multi-label) classification
    or regression between 0 and 1 with sigmoid output layer"""

    def function(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return -np.mean(xlogy(y_true, y_pred) + xlogy(1. - y_true, 1. - y_pred))


mean_squared_error = MeanSquaredError()
sparse_categorical_cross_entropy = SparseCategoricalCrossEntropy()
categorical_cross_entropy = CategoricalCrossEntropy()
binary_cross_entropy = BinaryCrossEntropy()
