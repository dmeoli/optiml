import numpy as np

from ml.neural_network.activations import Sigmoid, Softmax
from ml.neural_network.initializers import random_uniform


class Node:
    """
    A single unit of a layer in a neural network
    :param weights: weights between parent nodes and current node
    :param value: value of current node
    """

    def __init__(self, w=None, value=None):
        self.value = value
        self.w = w or []


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
            node.w = initializer(in_size)

    def forward(self, inputs):
        assert len(self.nodes) == len(inputs)
        self.inputs = inputs
        res = []
        for unit in self.nodes:
            val = self.activation.function(np.dot(unit.w, inputs))
            unit.value = val
            res.append(val)
        return res
