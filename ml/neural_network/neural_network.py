import random

import numpy as np

from ml.neural_network.layers import InputLayer, DenseLayer
from ml.learning import Learner
from ml.losses import mean_squared_error_loss
from utils import element_wise_product, vector_add, scalar_vector_product, matrix_multiplication, map_vector


def get_batch(examples, batch_size=1):
    """Split examples into multiple batches"""
    for i in range(0, len(examples), batch_size):
        yield examples[i: i + batch_size]


def init_examples(examples, idx_i, idx_t, o_units):
    """Init examples from dataset.examples."""

    inputs, targets = {}, {}
    for i, e in enumerate(examples):
        # input values of e
        inputs[i] = [e[i] for i in idx_i]

        if o_units > 1:
            # one-hot representation of e's target
            t = [0 for i in range(o_units)]
            t[e[idx_t]] = 1
            targets[i] = t
        else:
            # target value of e
            targets[i] = [e[idx_t]]

    return inputs, targets


def BackPropagation(inputs, targets, theta, net, loss):
    """
    The back-propagation algorithm for multilayer networks in only one epoch, to calculate gradients of theta.
    :param inputs: a batch of inputs in an array. Each input is an iterable object
    :param targets: a batch of targets in an array. Each target is an iterable object
    :param theta: parameters to be updated
    :param net: a list of predefined layer objects representing their linear sequence
    :param loss: a predefined loss function taking array of inputs and targets
    :return: gradients of theta, loss of the input batch
    """

    assert len(inputs) == len(targets)
    o_units = len(net[-1].nodes)
    n_layers = len(net)
    batch_size = len(inputs)

    gradients = [[[] for _ in layer.nodes] for layer in net]
    total_gradients = [[[0] * len(node.weights) for node in layer.nodes] for layer in net]

    batch_loss = 0

    # iterate over each example in batch
    for e in range(batch_size):
        i_val = inputs[e]
        t_val = targets[e]

        # forward pass and compute batch loss
        for i in range(1, n_layers):
            layer_out = net[i].forward(i_val)
            i_val = layer_out
        batch_loss += loss(t_val, layer_out)

        # initialize delta
        delta = [[] for _ in range(n_layers)]

        previous = [layer_out[i] - t_val[i] for i in range(o_units)]
        h_layers = n_layers - 1

        # backward pass
        for i in range(h_layers, 0, -1):
            layer = net[i]
            derivative = [layer.activation.derivative(node.val) for node in layer.nodes]
            delta[i] = element_wise_product(previous, derivative)
            # pass to layer i-1 in the next iteration
            previous = matrix_multiplication([delta[i]], theta[i])[0]
            # compute gradient of layer i
            gradients[i] = [scalar_vector_product(d, net[i].inputs) for d in delta[i]]

        # add gradient of current example to batch gradient
        total_gradients = vector_add(total_gradients, gradients)

    return total_gradients, batch_loss


def stochastic_gradient_descent(X, net, loss, epochs=1000, l_rate=0.01, batch_size=1, verbose=None):
    """
    Gradient descent algorithm to update the learnable parameters of a network.
    :return: the updated network
    """

    for e in range(epochs):
        total_loss = 0
        random.shuffle(X)
        weights = [[node.weights for node in layer.nodes] for layer in net]

        for batch in get_batch(X, batch_size):
            inputs, targets = init_examples(batch, [x for x in range(X.shape[1])], X.shape[1], len(net[-1].nodes))
            # compute gradients of weights
            gs, batch_loss = BackPropagation(inputs, targets, weights, net, loss)
            # update weights with gradient descent
            weights = vector_add(weights, scalar_vector_product(-l_rate, gs))
            total_loss += batch_loss
            # update the weights of network each batch
            for i in range(len(net)):
                if weights[i]:
                    for j in range(len(weights[i])):
                        net[i].nodes[j].weights = weights[i][j]

        if verbose and (e + 1) % verbose == 0:
            print("epoch:{}, total_loss:{}".format(e + 1, total_loss))

    return net


def adam(X, net, loss, epochs=1000, rho=(0.9, 0.999), delta=1 / 10 ** 8,
         l_rate=0.001, batch_size=1, verbose=None):
    """
    Adam optimizer to update the learnable parameters of a network.
    Required parameters are similar to gradient descent.
    :return the updated network
    """

    # init s, r and t
    s = [[[0] * len(node.weights) for node in layer.nodes] for layer in net]
    r = [[[0] * len(node.weights) for node in layer.nodes] for layer in net]
    t = 0

    # repeat util converge
    for e in range(epochs):
        # total loss of each epoch
        total_loss = 0
        random.shuffle(X)
        weights = [[node.weights for node in layer.nodes] for layer in net]

        for batch in get_batch(X, batch_size):
            t += 1
            inputs, targets = init_examples(batch, [x for x in range(X.shape[1])], X.shape[1], len(net[-1].nodes))

            # compute gradients of weights
            gs, batch_loss = BackPropagation(inputs, targets, weights, net, loss)

            # update s, r, s_hat and r_gat
            s = vector_add(scalar_vector_product(rho[0], s),
                           scalar_vector_product((1 - rho[0]), gs))
            r = vector_add(scalar_vector_product(rho[1], r),
                           scalar_vector_product((1 - rho[1]), element_wise_product(gs, gs)))
            s_hat = scalar_vector_product(1 / (1 - rho[0] ** t), s)
            r_hat = scalar_vector_product(1 / (1 - rho[1] ** t), r)

            # rescale r_hat
            r_hat = map_vector(lambda x: 1 / (np.sqrt(x) + delta), r_hat)

            # delta weights
            delta_theta = scalar_vector_product(-l_rate, element_wise_product(s_hat, r_hat))
            weights = vector_add(weights, delta_theta)
            total_loss += batch_loss

            # update the weights of network each batch
            for i in range(len(net)):
                if weights[i]:
                    for j in range(len(weights[i])):
                        net[i].nodes[j].weights = weights[i][j]

        if verbose and (e + 1) % verbose == 0:
            print("epoch:{}, total_loss:{}".format(e + 1, total_loss))

    return net


class PerceptronLearner(Learner):
    """
    Simple perceptron neural network.
    """

    def __init__(self, l_rate=0.01, epochs=100, optimizer=stochastic_gradient_descent,
                 batch_size=1, verbose=None):
        self.l_rate = l_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y):
        input_size = X.shape[1]
        output_size = len(set(y))

        # initialize the network, add dense layer
        raw_net = [InputLayer(input_size), DenseLayer(input_size, output_size)]

        # update the network
        self.learned_net = self.optimizer(X, raw_net, mean_squared_error_loss, self.epochs, l_rate=self.l_rate,
                                          batch_size=self.batch_size, verbose=self.verbose)

    def predict(self, x):
        layer_out = self.learned_net[1].forward(x)
        return layer_out.index(max(layer_out))


class NeuralNetLearner(Learner):
    """
    Simple dense multilayer neural network.
    :param hidden_layer_sizes: size of hidden layers in the form of a list
    """

    def __init__(self, hidden_layer_sizes, l_rate=0.01, epochs=100,
                 optimizer=stochastic_gradient_descent, batch_size=1, verbose=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.l_rate = l_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y):
        input_size = X.shape[1]
        output_size = len(set(y))

        # initialize the network
        raw_net = [InputLayer(input_size)]
        # add hidden layers
        hidden_input_size = input_size
        for h_size in self.hidden_layer_sizes:
            raw_net.append(DenseLayer(hidden_input_size, h_size))
            hidden_input_size = h_size
        raw_net.append(DenseLayer(hidden_input_size, output_size))

        # update parameters of the network
        self.learned_net = self.optimizer(X, raw_net, mean_squared_error_loss, self.epochs, l_rate=self.l_rate,
                                          batch_size=self.batch_size, verbose=self.verbose)

    def predict(self, x):
        n_layers = len(self.learned_net)

        layer_input = x
        layer_out = x

        # get the output of each layer by forward passing
        for i in range(1, n_layers):
            layer_out = self.learned_net[i].forward(layer_input)
            layer_input = layer_out

        return layer_out.index(max(layer_out))
