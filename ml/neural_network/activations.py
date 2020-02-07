from abc import ABC

import numpy as np


class Activation(ABC):

    def function(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Sigmoid(Activation):

    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)


class ReLU(Activation):

    def function(self, x):
        return np.max(0, x)

    def derivative(self, x):
        return 1 if x > 0 else 0


class ELU(Activation):

    def function(self, x, alpha=0.01):
        return x if x > 0 else alpha * (np.exp(x) - 1)

    def derivative(self, x, alpha=0.01):
        return 1 if x > 0 else alpha * np.exp(x)


class Tanh(Activation):

    def function(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - (x ** 2)


class LeakyReLU(Activation):

    def function(self, x, alpha=0.01):
        return x if x > 0 else alpha * x

    def derivative(self, x, alpha=0.01):
        return 1 if x > 0 else alpha


class Softmax(Activation):

    def function(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def derivative(self, x):
        return np.diagflat(x) - np.dot(x, x.T)
