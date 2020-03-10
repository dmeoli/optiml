import random

import numpy as np


def not_test(func):
    """Decorator to mark a function or method as not a test"""
    func.__test__ = False
    return func


def load_monk(dataset):
    if dataset not in (1, 2, 3):
        raise ValueError('unknown monk dataset type {}'.format(dataset))
    monks = [np.delete(np.genfromtxt('./ml/data/monks/monks-' + str(dataset) + '.' + type), -1, axis=1)
             for type in ('train', 'test')]
    return (monks[0][:, 1:], monks[1][:, 1:],  # X_train, X_test
            monks[0][:, 0].ravel(), monks[1][:, 0].ravel())  # y_train, y_test


def iter_mini_batches(Xy, batch_size):
    """Return an iterator that successively yields tuples containing aligned
    mini batches of size batch_size from sliceable objects given in Xy, in
    random order without replacement.
    Because different containers might require slicing over different
    dimensions, the dimension of each container has to be givens as a list
    dims.
    :param: Xy: arrays to be sliced into mini batches in alignment with the others
    :param: batch_size: size of each batch
    :return: infinite iterator of mini batches in random order (without replacement)
    """

    if Xy[0].shape[0] != Xy[1].shape[0]:
        raise ValueError('X and y have unequal lengths')

    if batch_size > Xy[0].shape[0]:
        raise ValueError('batch_size must be less or equal than the number of examples')

    n_batches, rest = divmod(len(Xy[0]), batch_size)
    if rest:
        n_batches += 1

    while True:
        idx = list(range(n_batches))
        while True:
            random.shuffle(idx)
            for i in idx:
                start = i * batch_size
                stop = (i + 1) * batch_size
                yield [param[slice(start, stop)] for param in Xy]


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix."""
    y = np.array(y, dtype=np.int)
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
