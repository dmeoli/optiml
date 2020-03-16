import os
import random

import numpy as np
from scipy.linalg import ldl
from sklearn.preprocessing import OneHotEncoder


def not_test(func):
    """Decorator to mark a function or method as not a test"""
    func.__test__ = False
    return func


def cholesky_solve(A, b):
    """Symmetric positive definite matrix"""
    # A = L L^T
    L = np.linalg.cholesky(A)
    # L y = b => y = L^-1 b
    y = np.linalg.inv(L).dot(b)
    # L^T x = y => x = L^T^-1 y
    return np.linalg.inv(L.T).dot(y)


def cholesky_ldl_solve(A, b):
    """Symmetric indefinite matrix"""
    # A = L D L^T
    L, D, _ = ldl(A)
    # L y = b => y = L^-1 b
    y = np.linalg.inv(L).dot(b)
    # D z = y => z = D^-1 y
    z = np.linalg.inv(D).dot(y)
    # L^T x = z => x = L^T^-1 z
    return np.linalg.inv(L.T).dot(z)


def load_monk(n_dataset):
    if n_dataset not in (1, 2, 3):
        raise ValueError('unknown monk dataset type {}'.format(n_dataset))
    path = os.path.dirname(os.path.abspath(__file__))
    monks = [np.delete(np.genfromtxt(path + '/ml/data/monks/monks-' + str(n_dataset) + '.' + type), -1, axis=1)
             for type in ('train', 'test')]
    return (OneHotEncoder().fit_transform(monks[0][:, 1:]).toarray(),  # X_train
            OneHotEncoder().fit_transform(monks[1][:, 1:]).toarray(),  # X_test
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
