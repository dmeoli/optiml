import numpy as np


class Activation:

    def function(self, x):
        return NotImplementedError

    def derivative(self, x):
        return NotImplementedError


class Sigmoid(Activation):

    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)


class Relu(Activation):

    def function(self, x):
        return np.max(0, x)

    def derivative(self, x):
        return 1 if x > 0 else 0


class Elu(Activation):

    def function(self, x, alpha=0.01):
        return x if x > 0 else alpha * (np.exp(x) - 1)

    def derivative(self, x, alpha=0.01):
        return 1 if x > 0 else alpha * np.exp(x)


class Tanh(Activation):

    def function(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - (x ** 2)


class LeakyRelu(Activation):

    def function(self, x, alpha=0.01):
        return x if x > 0 else alpha * x

    def derivative(self, x, alpha=0.01):
        return 1 if x > 0 else alpha


class Softmax(Activation):

    def function(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def derivative(self, x):
        return np.diagflat(x) - np.dot(x, x.T)
