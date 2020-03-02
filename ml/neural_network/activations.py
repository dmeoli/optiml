import numpy as np
from scipy.special import expit


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
        return np.maximum(0., x)

    def derivative(self, x):
        return np.where(x > 0, 1., 0.)


class LeakyReLU(Activation):

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def function(self, x):
        return np.maximum(x, self.alpha * x)

    def derivative(self, x):
        return np.where(x > 0, 1., self.alpha)


class ELU(Activation):

    def __init__(self, alpha=1.):
        self.alpha = alpha

    def function(self, x):
        return np.maximum(x, self.alpha * (np.exp(x) - 1.))

    def derivative(self, x):
        return np.where(x > 0, 1., self.function(x) + self.alpha)


class Tanh(Activation):

    def function(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1. - np.square(self.function(x))


class Sigmoid(Activation):

    def function(self, x):
        return expit(x)

    def derivative(self, x):
        x = self.function(x)
        return x * (1. - x)


class SoftPlus(Activation):

    def function(self, x):
        return np.log(1. + np.exp(x))

    def derivative(self, x):
        return 1. / (1. + np.exp(-x))


class SoftMax(Activation):

    def function(self, x, axis=-1):
        exps = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exps / np.sum(exps, axis=axis, keepdims=True)

    def derivative(self, x):
        return np.ones_like(x)


linear = Linear()
relu = ReLU()
leaky_relu = LeakyReLU()
elu = ELU()
tanh = Tanh()
sigmoid = Sigmoid()
softplus = SoftPlus()
softmax = SoftMax()
