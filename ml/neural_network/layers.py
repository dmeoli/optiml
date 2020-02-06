from statistics import stdev

import numpy as np

from ml.neural_network.activations import Sigmoid, Softmax
from ml.initializers import random_uniform
from utils import gaussian_kernel, conv1D


class Node:
    """
    A single unit of a layer in a neural network
    :param weights: weights between parent nodes and current node
    :param value: value of current node
    """

    def __init__(self, weights=None, value=None):
        self.value = value
        self.weights = weights or []


class Layer:
    """
    A layer in a neural network based on a computational graph.
    :param size: number of units in the current layer
    """

    def __init__(self, size=3):
        self.nodes = np.array([Node() for _ in range(size)])

    def forward(self, inputs):
        """Define the operation to get the output of this layer"""
        raise NotImplementedError


class InputLayer(Layer):
    """1D input layer. Layer size is the same as input vector size."""

    def __init__(self, size=3):
        super().__init__(size)

    def forward(self, inputs):
        """Take each value of the inputs to each unit in the layer."""
        assert len(self.nodes) == len(inputs)
        for node, inp in zip(self.nodes, inputs):
            node.value = inp
        return inputs


class OutputLayer(Layer):
    """1D softmax output layer."""

    def __init__(self, size=3):
        super().__init__(size)

    def forward(self, inputs):
        assert len(self.nodes) == len(inputs)
        res = Softmax().function(inputs)
        for node, val in zip(self.nodes, res):
            node.value = val
        return res


class DenseLayer(Layer):
    """
    1D dense layer in a neural network.
    :param in_size: (int) input vector size
    :param out_size: (int) output vector size
    :param activation: activation function
    """

    def __init__(self, in_size=3, out_size=3, activation=Sigmoid, initializer=random_uniform):
        super().__init__(out_size)
        self.out_size = out_size
        self.inputs = None
        self.activation = activation()
        for node in self.nodes:
            node.weights = initializer(in_size)

    def forward(self, inputs):
        self.inputs = inputs
        res = []
        for unit in self.nodes:
            val = self.activation.function(np.dot(unit.weights, inputs))
            unit.value = val
            res.append(val)
        return res


class ConvLayer1D(Layer):

    def __init__(self, size=3, kernel_size=3):
        super().__init__(size)
        # init convolution kernel as gaussian kernel
        for node in self.nodes:
            node.weights = gaussian_kernel(kernel_size)

    def forward(self, features):
        # each node in layer takes a channel in the features
        assert len(self.nodes) == len(features)
        res = []
        for node, feature in zip(self.nodes, features):
            out = conv1D(feature, node.weights)
            res.append(out)
            node.value = out
        return res


class MaxPoolingLayer1D(Layer):

    def __init__(self, size=3, kernel_size=3):
        super().__init__(size)
        self.kernel_size = kernel_size
        self.inputs = None

    def forward(self, features):
        assert len(self.nodes) == len(features)
        res = []
        self.inputs = features
        # do max pooling for each channel in features
        for i in range(len(self.nodes)):
            feature = features[i]
            # get the max value in a kernel_size * kernel_size area
            out = [max(feature[i:i + self.kernel_size])
                   for i in range(len(feature) - self.kernel_size + 1)]
            res.append(out)
            self.nodes[i].value = out
        return res


class BatchNormalizationLayer(Layer):

    def __init__(self, size, eps=0.001):
        super().__init__(size)
        self.eps = eps
        # self.weights = [beta, gamma]
        self.weights = [0, 0]
        self.inputs = None

    def forward(self, inputs):
        # mean value of inputs
        mu = sum(inputs) / len(inputs)
        # standard error of inputs
        stderr = stdev(inputs)
        self.inputs = inputs
        res = []
        # get normalized value of each input
        for i in range(len(self.nodes)):
            val = [(inputs[i] - mu) * self.weights[0] / np.sqrt(self.eps + stderr ** 2) + self.weights[1]]
            res.append(val)
            self.nodes[i].value = val
        return res
