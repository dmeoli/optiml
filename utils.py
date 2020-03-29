import os
import random

import numpy as np
from sklearn.preprocessing import OneHotEncoder


def cholesky_solve(A, b):
    """Solve a symmetric positive definite linear
    system A x = b using Cholesky factorization"""
    L = np.linalg.cholesky(A)  # complexity O(n^3/3)
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


def ldl_solve(ldl_factor, b):
    """Solve a symmetric indefinite linear system
    A x = b using the LDL^T Cholesky factorization."""
    L, D, P = ldl_factor  # complexity O(n^3/3)
    return np.linalg.solve(L.T, (np.linalg.solve(D, np.linalg.solve(L, b[P]))))


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


def generate_linearly_separable_data():
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = -np.ones(len(X2))
    return X1, y1, X2, y2


def gen_non_linearly_separable_data():
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0, 0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, 50)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 50)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
    y2 = -np.ones(len(X2))
    return X1, y1, X2, y2


def gen_lin_separable_overlap_data():
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[1.5, 1.0], [1.0, 1.5]])
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = -np.ones(len(X2))
    return X1, y1, X2, y2


def split_data(X1, y1, X2, y2, percent):
    cut_off = int(len(X1) * percent)

    # training data
    X1_train = X1[:cut_off]
    y1_train = y1[:cut_off]
    X2_train = X2[:cut_off]
    y2_train = y2[:cut_off]

    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))

    # test data
    X1_test = X1[cut_off:]
    y1_test = y1[cut_off:]
    X2_test = X2[cut_off:]
    y2_test = y2[cut_off:]

    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))

    return X_train, y_train, X_test, y_test
