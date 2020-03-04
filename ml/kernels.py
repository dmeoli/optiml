import numpy as np


def linear_kernel(X, y=None):
    if y is None:
        y = X
    return np.dot(X, y.T)


def polynomial_kernel(X, y=None, r=0., degree=3.):
    """A non-stationary kernel well suited for problems
    where all the training data is normalized"""
    if y is None:
        y = X
    return (np.dot(X, y.T) + r) ** degree


def rbf_kernel(X, y=None, gamma='scale'):
    """Radial-basis function kernel (aka squared-exponential kernel)."""
    if y is None:
        y = X
    gamma = 1. / (X.shape[1] * X.var()) if gamma is 'scale' else 1. / X.shape[1]  # auto
    # ||x - y|| ^ 2 = (x - y)^T * (x - y) = x^T * x + y^T * y - 2 * x^T * y
    # according to: https://stats.stackexchange.com/questions/239008/rbf-kernel-algorithm-python
    return np.exp(-gamma * (np.sum(X * X, axis=1).reshape((-1, 1)) +
                            np.sum(y * y, axis=1).reshape((1, -1)) - 2. * np.dot(X, y.T)))


def sigmoid_kernel(X, y=None, r=0.):
    if y is None:
        y = X
    return np.tanh(np.dot(X, y.T) + r)
