import numpy as np


def linear_kernel(X, y):
    return np.dot(X, y.T)


def polynomial_kernel(X, y, degree=3.):
    """A non-stationary kernel well suited for problems
    where all the training data is normalized"""
    return np.dot(X, y.T) ** degree


def rbf_kernel(X, y, gamma='scale'):
    """Radial-basis function kernel (aka squared-exponential kernel)."""
    gamma = 1. / (X.shape[1] * X.var()) if gamma is 'scale' else 1. / X.shape[1]  # auto
    # according to: https://stats.stackexchange.com/questions/239008/rbf-kernel-algorithm-python
    return np.exp(-gamma * (np.dot(X ** 2, np.ones((X.shape[1], y.shape[0]))) +
                            np.dot(np.ones((X.shape[0], X.shape[1])), y.T ** 2) - 2. * np.dot(X, y.T)))
