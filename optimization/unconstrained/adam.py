import random

import numpy as np

from ml.neural_network import BackPropagation, get_batch, init_examples
from utils import vector_add, scalar_vector_product, map_vector, element_wise_product


def Adam(x, inputs, target, net, loss, epochs=1000, rho=(0.9, 0.999), delta=1 / 10 ** 8,
         l_rate=0.001, batch_size=1, verbose=None):
    """
    Adam optimizer to update the learnable parameters of a network.
    Required parameters are similar to gradient descent.
    :return the updated network
    """

    # init s,r and t
    s = [[[0] * len(node.weights) for node in layer.nodes] for layer in net]
    r = [[[0] * len(node.weights) for node in layer.nodes] for layer in net]
    t = 0

    # repeat util converge
    for e in range(epochs):
        # total loss of each epoch
        total_loss = 0
        random.shuffle(x)
        weights = [[node.weights for node in layer.nodes] for layer in net]

        for batch in get_batch(x, batch_size):
            t += 1
            inputs, targets = init_examples(batch, inputs, target, len(net[-1].nodes))

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
