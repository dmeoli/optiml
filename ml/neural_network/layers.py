from statistics import stdev

import numpy as np

from ml.neural_network.activations import Sigmoid


class Node:
    """
    A node in a computational graph contains the pointer to all its parents.
    :param val: value of current node
    :param parents: a container of all parents of current node
    """

    def __init__(self, val=None, parents=None):
        if parents is None:
            parents = []
        self.val = val
        self.parents = parents

    def __repr__(self):
        return "<Node {}>".format(self.val)


class NNUnit(Node):
    """
    A single unit of a layer in a neural network
    :param weights: weights between parent nodes and current node
    :param value: value of current node
    """

    def __init__(self, weights=None, value=None):
        super().__init__(value)
        self.weights = weights or []


class Layer:
    """
    A layer in a neural network based on a computational graph.
    :param size: number of units in the current layer
    """

    def __init__(self, size=3):
        self.nodes = [NNUnit() for _ in range(size)]

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
            node.val = inp
        return inputs


def softmax1D(x):
    """Return the softmax vector of input vector x."""
    return np.exp(x) / sum(np.exp(x))


class OutputLayer(Layer):
    """1D softmax output layer."""

    def __init__(self, size=3):
        super().__init__(size)

    def forward(self, inputs):
        assert len(self.nodes) == len(inputs)
        res = softmax1D(inputs)
        for node, val in zip(self.nodes, res):
            node.val = val
        return res


class DenseLayer(Layer):
    """
    1D dense layer in a neural network.
    :param in_size: (int) input vector size
    :param out_size: (int) output vector size
    :param activation: activation function
    """

    def __init__(self, in_size=3, out_size=3, activation=None):
        super().__init__(out_size)
        self.out_size = out_size
        self.inputs = None
        self.activation = Sigmoid() if not activation else activation
        # initialize weights
        for node in self.nodes:
            node.weights = np.random.uniform(-0.5, 0.5, in_size)

    def forward(self, inputs):
        self.inputs = inputs
        res = []
        # get the output value of each unit
        for unit in self.nodes:
            val = self.activation.function(np.dot(unit.weights, inputs))
            unit.val = val
            res.append(val)
        return res


def gaussian_kernel(size=3):
    def gaussian(mean, st_dev, x):
        """Given the mean and standard deviation of a distribution, it returns the probability of x."""
        return 1 / (np.sqrt(2 * np.pi) * st_dev) * np.exp(-0.5 * (float(x - mean) / st_dev) ** 2)

    return [gaussian((size - 1) / 2, 0.1, x) for x in range(size)]


def conv1D(x, k):
    """1D convolution. x: input vector; K: kernel vector."""
    return np.convolve(x, k, mode='same')


class ConvLayer1D(Layer):
    """
    1D convolution layer of in neural network.
    :param kernel_size: convolution kernel size
    """

    def __init__(self, size=3, kernel_size=3):
        super().__init__(size)
        # init convolution kernel as gaussian kernel
        for node in self.nodes:
            node.weights = gaussian_kernel(kernel_size)

    def forward(self, features):
        # each node in layer takes a channel in the features
        assert len(self.nodes) == len(features)
        res = []
        # compute the convolution output of each channel, store it in node.val
        for node, feature in zip(self.nodes, features):
            out = conv1D(feature, node.weights)
            res.append(out)
            node.val = out
        return res


class MaxPoolingLayer1D(Layer):
    """
    1D max pooling layer in a neural network.
    :param kernel_size: max pooling area size
    """

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
            self.nodes[i].val = out
        return res


class BatchNormalizationLayer(Layer):
    """Batch normalization layer."""

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
            self.nodes[i].val = val
        return res
