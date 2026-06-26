from abc import ABC

import numpy as np
from autograd.scipy.special import expit


class Activation(ABC):
    """
    Base abstract class for all activation functions. Subclasses must
    implement ``function`` and its element-wise derivative ``jacobian``.
    """

    def function(self, x):
        raise NotImplementedError

    def jacobian(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.function(x)


class Linear(Activation):
    """Identity (linear) activation function: f(x) = x."""

    def function(self, x):
        return x

    def jacobian(self, x):
        return np.ones_like(x)


class ReLU(Activation):
    """Rectified linear unit activation function: f(x) = max(0, x)."""

    def function(self, x):
        return np.maximum(0., x)

    def jacobian(self, x):
        return np.where(x > 0, 1., 0.)


class Tanh(Activation):
    """Hyperbolic tangent activation function: f(x) = tanh(x)."""

    def function(self, x):
        return np.tanh(x)

    def jacobian(self, x):
        return 1. - np.square(self.function(x))


class Sigmoid(Activation):
    """Logistic sigmoid activation function: f(x) = 1 / (1 + exp(-x))."""

    def function(self, x):
        return expit(x)

    def jacobian(self, x):
        x = self.function(x)
        return x * (1. - x)


class SoftMax(Activation):
    """Softmax activation function: f(x)_i = exp(x_i) / sum_j exp(x_j)."""

    def function(self, x, axis=-1):
        exps = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exps / np.sum(exps, axis=axis, keepdims=True)

    def jacobian(self, x):
        return np.ones_like(x)


linear = Linear()
relu = ReLU()
tanh = Tanh()
sigmoid = Sigmoid()
softmax = SoftMax()
