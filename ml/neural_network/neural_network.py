import random

import numpy as np

from ml.neural_network.layers import InputLayer, DenseLayer


def init_examples(examples, idx_i, idx_t, o_units):
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


def stochastic_gradient_descent(dataset, net, loss, epochs=1000, l_rate=0.01, batch_size=1, verbose=False):
    examples = dataset.examples  # init data

    for e in range(epochs):
        total_loss = 0
        random.shuffle(examples)
        weights = [[node.weights for node in layer.nodes] for layer in net]

        inputs, targets = init_examples(examples, dataset.inputs, dataset.target, len(net[-1].nodes))
        # compute gradients of weights
        gs, batch_loss = BackPropagation(inputs, targets, weights, net, loss)
        # update weights with gradient descent
        weights = [x + y for x, y in zip(weights, [np.array(tg) * -l_rate for tg in gs])]
        total_loss += batch_loss

        # update the weights of network each batch
        for i in range(len(net)):
            if weights[i].size != 0:
                for j in range(len(weights[i])):
                    net[i].nodes[j].weights = weights[i][j]

        if verbose:
            print("epoch:{}, total_loss:{}".format(e + 1, total_loss))

    return net


def vector_add(a, b):
    if not (a and b):
        return a or b
    if hasattr(a, '__iter__') and hasattr(b, '__iter__'):
        assert len(a) == len(b)
        return list(map(vector_add, a, b))
    else:
        try:
            return a + b
        except TypeError:
            raise Exception('Inputs must be in the same size!')


def scalar_vector_product(x, y):
    return [scalar_vector_product(x, _y) for _y in y] if hasattr(y, '__iter__') else x * y


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

        previous = np.array([layer_out[i] - t_val[i] for i in range(o_units)])
        h_layers = n_layers - 1

        # backward pass
        for i in range(h_layers, 0, -1):
            layer = net[i]
            derivative = np.array([layer.activation.derivative(node.value) for node in layer.nodes])
            delta[i] = previous * derivative
            # pass to layer i-1 in the next iteration
            previous = np.matmul([delta[i]], theta[i])[0]
            # compute gradient of layer i
            gradients[i] = [scalar_vector_product(d, net[i].inputs) for d in delta[i]]

        # add gradient of current example to batch gradient
        total_gradients = vector_add(total_gradients, gradients)

    return total_gradients, batch_loss


def mean_squared_error_loss(x, y):
    """Min square loss function. x and y are 1D iterable objects."""
    return (1.0 / len(x)) * sum((_x - _y) ** 2 for _x, _y in zip(x, y))


def NeuralNetLearner(dataset, hidden_layer_sizes, l_rate=0.01, epochs=1000, batch_size=1,
                     optimizer=stochastic_gradient_descent, verbose=False):
    """
    Simple dense multilayer neural network.
    :param hidden_layer_sizes: size of hidden layers in the form of a list
    """

    input_size = len(dataset.inputs)
    output_size = len(dataset.values[dataset.target])

    # initialize the network
    raw_net = [InputLayer(input_size)]
    # add hidden layers
    hidden_input_size = input_size
    for h_size in hidden_layer_sizes:
        raw_net.append(DenseLayer(hidden_input_size, h_size))
        hidden_input_size = h_size
    raw_net.append(DenseLayer(hidden_input_size, output_size))

    # update parameters of the network
    learned_net = optimizer(dataset, raw_net, mean_squared_error_loss, epochs=epochs,
                            l_rate=l_rate, batch_size=batch_size, verbose=verbose)

    def predict(example):
        n_layers = len(learned_net)

        layer_input = example
        layer_out = example

        # get the output of each layer by forward passing
        for i in range(1, n_layers):
            layer_out = learned_net[i].forward(layer_input)
            layer_input = layer_out

        return layer_out.index(max(layer_out))

    return predict


def PerceptronLearner(dataset, l_rate=0.01, epochs=1000, batch_size=1,
                      optimizer=stochastic_gradient_descent, verbose=False):
    """
    Simple perceptron neural network.
    """

    input_size = len(dataset.inputs)
    output_size = len(dataset.values[dataset.target])

    # initialize the network, add dense layer
    raw_net = [InputLayer(input_size), DenseLayer(input_size, output_size)]

    # update the network
    learned_net = optimizer(dataset, raw_net, mean_squared_error_loss, epochs=epochs,
                            l_rate=l_rate, batch_size=batch_size, verbose=verbose)

    def predict(example):
        layer_out = learned_net[1].forward(example)
        return layer_out.index(max(layer_out))

    return predict
