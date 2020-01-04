import numpy as np


# TODO normalized squared error, weighted squared error and Minkowski error

def cross_entropy_loss(x, y):
    """Cross entropy loss function. x and y are 1D iterable objects."""
    return (-1.0 / len(x)) * sum(_x * np.log(_y) + (1 - _x) * np.log(1 - _y) for _x, _y in zip(x, y))


def mean_squared_error_loss(x, y):
    """Mean squared error loss function. x and y are 1D iterable objects."""
    return (1.0 / len(x)) * sum((_x - _y) ** 2 for _x, _y in zip(x, y))
