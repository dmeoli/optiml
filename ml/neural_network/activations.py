import numpy as np


class Activation:

    def function(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Linear(Activation):

    def function(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class ReLU(Activation):

    def function(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, np.ones_like(x), np.zeros_like(x))


class Tanh(Activation):

    def function(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1. - np.square(np.tanh(x))


class Sigmoid(Activation):

    def function(self, x):
        return 1. / (1. + np.exp(-x))

    def derivative(self, x):
        f = self.function(x)
        return f * (1. - f)


class Softmax(Activation):

    def function(self, x, axis=-1):
        exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def derivative(self, x):
        return np.ones_like(x)
