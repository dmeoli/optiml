from abc import ABC

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import check_pairwise_arrays, euclidean_distances, manhattan_distances
from sklearn.utils.extmath import safe_sparse_dot


class Kernel(BaseEstimator, ABC):
    """
    Base abstract class for all kernel functions. A kernel computes the
    pairwise similarities (i.e., the Gram matrix) between two sets of samples.

    Subclasses must implement ``__call__``.
    """

    def __call__(self, X, Y=None):
        """
        Compute the kernel (Gram) matrix between X and Y.

        Parameters
        ----------

        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the kernel function.

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the kernel function. If None, ``Y`` is
            set to ``X`` and the kernel matrix between X and itself is computed.

        Returns
        -------

        K : ndarray of shape (n_samples_X, n_samples_Y)
            The computed kernel matrix.
        """
        raise NotImplementedError


class LinearKernel(Kernel):
    """
    Compute the linear kernel between X and Y:

        K(X, Y) = <X, Y>
    """

    def __call__(self, X, Y=None):
        X, Y = check_pairwise_arrays(X, Y)
        return safe_sparse_dot(X, Y.T, dense_output=True)


class PolyKernel(Kernel):
    """
    Compute the polynomial kernel between X and Y:

        K(X, Y) = (gamma <X, Y> + coef0)^degree

    Parameters
    ----------

    degree : int, default=3
        Degree of the polynomial kernel function.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for kernel function.

        - if `gamma='scale'` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if `gamma='auto'`, uses 1 / n_features.

    coef0 : float, default=0.0
        Independent term in kernel function.
    """

    def __init__(self, degree=3, gamma='scale', coef0=0.):
        if not degree > 0:
            raise ValueError('degree must be > 0')
        self.degree = degree
        if isinstance(gamma, str):
            if gamma not in ('scale', 'auto'):
                raise ValueError(f'unknown gamma type {gamma}')
        elif not gamma > 0:
            raise ValueError('gamma must be > 0')
        self.gamma = gamma
        self.coef0 = coef0

    def __call__(self, X, Y=None):
        X, Y = check_pairwise_arrays(X, Y)
        gamma = (1. / (X.shape[1] * X.var()) if self.gamma == 'scale' else
                 1. / X.shape[1] if self.gamma == 'auto' else self.gamma)
        return (gamma * safe_sparse_dot(X, Y.T, dense_output=True) + self.coef0) ** self.degree


class GaussianKernel(Kernel):
    """
    Compute the gaussian RBF kernel between X and Y:

        K(X, Y) = exp(-gamma ||X - Y||_2^2)

    Parameters
    ----------

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for kernel function.

        - if `gamma='scale'` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if `gamma='auto'`, uses 1 / n_features.
    """

    def __init__(self, gamma='scale'):
        if isinstance(gamma, str):
            if gamma not in ('scale', 'auto'):
                raise ValueError(f'unknown gamma type {gamma}')
        elif not gamma > 0:
            raise ValueError('gamma must be > 0')
        self.gamma = gamma

    def __call__(self, X, Y=None):
        X, Y = check_pairwise_arrays(X, Y)
        gamma = (1. / (X.shape[1] * X.var()) if self.gamma == 'scale' else
                 1. / X.shape[1] if self.gamma == 'auto' else self.gamma)
        return np.exp(-gamma * euclidean_distances(X, Y, squared=True))


class LaplacianKernel(Kernel):
    """
    Compute the laplacian RBF kernel between X and Y:

        K(X, Y) = exp(-gamma ||X - Y||_1)

    Parameters
    ----------

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for kernel function.

        - if `gamma='scale'` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if `gamma='auto'`, uses 1 / n_features.
    """

    def __init__(self, gamma='scale'):
        if isinstance(gamma, str):
            if gamma not in ('scale', 'auto'):
                raise ValueError(f'unknown gamma type {gamma}')
        elif not gamma > 0:
            raise ValueError('gamma must be > 0')
        self.gamma = gamma

    def __call__(self, X, Y=None):
        X, Y = check_pairwise_arrays(X, Y)
        gamma = (1. / (X.shape[1] * X.var()) if self.gamma == 'scale' else
                 1. / X.shape[1] if self.gamma == 'auto' else self.gamma)
        return np.exp(-gamma * manhattan_distances(X, Y))


class SigmoidKernel(Kernel):
    """
    Compute the sigmoid kernel between X and Y:

        K(X, Y) = tanh(gamma <X, Y> + coef0)

    Parameters
    ----------

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for kernel function.

        - if `gamma='scale'` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if `gamma='auto'`, uses 1 / n_features.

    coef0 : float, default=0.0
        Independent term in kernel function.
    """

    def __init__(self, gamma='scale', coef0=0.):
        if isinstance(gamma, str):
            if gamma not in ('scale', 'auto'):
                raise ValueError(f'unknown gamma type {gamma}')
        elif not gamma > 0:
            raise ValueError('gamma must be > 0')
        self.gamma = gamma
        self.coef0 = coef0

    def __call__(self, X, Y=None):
        X, Y = check_pairwise_arrays(X, Y)
        gamma = (1. / (X.shape[1] * X.var()) if self.gamma == 'scale' else
                 1. / X.shape[1] if self.gamma == 'auto' else self.gamma)
        return np.tanh(gamma * safe_sparse_dot(X, Y.T, dense_output=True) + self.coef0)


linear = LinearKernel()
poly = PolyKernel()
gaussian = GaussianKernel()
laplacian = LaplacianKernel()
sigmoid = SigmoidKernel()
