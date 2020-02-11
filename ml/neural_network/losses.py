import numpy as np

from ml.neural_network.activations import Softmax


class Loss:

    def __init__(self, loss, delta):
        self.data = loss
        self.delta = delta

    def __repr__(self):
        return str(self.data)


class LossFunction:

    def __init__(self):
        self._pred = None
        self._target = None

    def function(self, prediction, target):
        raise NotImplementedError

    @property
    def delta(self):
        raise NotImplementedError

    def _store_pred_target(self, prediction, target):
        p = prediction.data
        self._pred = p
        self._target = target


class MSE(LossFunction):

    def function(self, prediction, target):
        self.prediction = prediction.data
        self.target = target
        loss = np.mean(np.square(self.delta))
        return Loss(loss, self.delta)

    @property
    def delta(self):
        return self.prediction - self.target


class CrossEntropy(LossFunction):

    def __init__(self):
        super().__init__()
        self._eps = 1e-6

    def function(self, prediction, target):
        raise NotImplementedError

    @property
    def delta(self):
        raise NotImplementedError


class SoftMaxCrossEntropy(CrossEntropy):

    def function(self, prediction, target):
        t = target if target.dtype is np.float32 else target.astype(np.float32)
        self._store_pred_target(prediction, t)
        loss = -np.mean(np.sum(t * np.log(self._pred), axis=-1))
        return Loss(loss, self.delta)

    @property
    def delta(self):
        # according to: https://deepnotes.io/softmax-crossentropy
        onehot_mask = self._target.astype(np.bool)
        grad = self._pred.copy()
        grad[onehot_mask] -= 1.
        return grad / len(grad)


class SoftMaxCrossEntropyWithLogits(CrossEntropy):

    def function(self, prediction, target):
        self._store_pred_target(prediction, target)
        sm = Softmax().function(self._pred)
        loss = -np.mean(np.sum(target * np.log(sm), axis=-1))
        return Loss(loss, self.delta)

    @property
    def delta(self):
        grad = Softmax().function(self._pred)
        onehot_mask = self._target.astype(np.bool)
        grad[onehot_mask] -= 1.
        return grad / len(grad)


class SigmoidCrossEntropy(CrossEntropy):

    def function(self, prediction, target):
        self._store_pred_target(prediction, target)
        p = self._pred
        loss = -np.mean(target * np.log(p + self._eps) + (1. - target) * np.log(1 - p + self._eps))
        return Loss(loss, self.delta)

    @property
    def delta(self):
        return self._pred - self._target
