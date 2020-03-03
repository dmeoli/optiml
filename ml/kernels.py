import numpy as np


def linear_kernel(X, y=None):
    if y is None:
        y = X
    return np.dot(X, y.T)


def polynomial_kernel(X, y=None, degree=3.):
    if y is None:
        y = X
    return (1. + np.dot(X, y.T)) ** degree


def rbf_kernel(X, y=None, gamma='scale'):
    """Radial-basis function kernel (aka squared-exponential kernel)."""
    if y is None:
        y = X
    gamma = 1. / (X.shape[1] * X.var()) if gamma is 'scale' else 1. / X.shape[1]  # auto
    return np.exp(-gamma * (-2. * np.dot(X, y.T) +
                            np.sum(X * X, axis=1).reshape((-1, 1)) + np.sum(y * y, axis=1).reshape((1, -1))))
