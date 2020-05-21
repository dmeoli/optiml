import numpy as np


class Kernel:

    def __call__(self, X, Y=None):
        pass


class LinearKernel(Kernel):
    """
    Compute the linear kernel between X and Y:

        K(X, Y) = <X, Y>
    """

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        return np.dot(X, Y.T)


class PolyKernel(Kernel):
    """
    Compute the polynomial kernel between X and Y:

        K(X, Y) = (gamma <X, Y> + coef0)^degree
    """

    def __init__(self, degree=3, gamma='scale', coef0=0.):
        if not degree > 0:
            raise ValueError('degree must be > 0')
        self.degree = degree
        if isinstance(gamma, str):
            if gamma not in ('scale', 'auto'):
                raise ValueError(f'unknown gamma type {gamma}')
        else:
            if not gamma > 0:
                raise ValueError('gamma must be > 0')
        self.gamma = gamma
        self.coef0 = coef0

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        gamma = (1. / (X.shape[1] * X.var()) if self.gamma == 'scale' else  # auto
                 1. / X.shape[1] if isinstance(self.gamma, str) else self.gamma)
        return (gamma * np.dot(X, Y.T) + self.coef0) ** self.degree


class RBFKernel(Kernel):
    """
    Compute the rbf (gaussian) kernel between X and Y:

        K(x, y) = exp(-gamma ||x-y||_2^2)
    """

    def __init__(self, gamma='scale'):
        if isinstance(gamma, str):
            if gamma not in ('scale', 'auto'):
                raise ValueError(f'unknown gamma type {gamma}')
        else:
            if not gamma > 0:
                raise ValueError('gamma must be > 0')
        self.gamma = gamma

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        gamma = (1. / (X.shape[1] * X.var()) if self.gamma == 'scale' else  # auto
                 1. / X.shape[1] if isinstance(self.gamma, str) else self.gamma)
        return np.exp(-gamma * np.linalg.norm(X[:, np.newaxis] - Y[np.newaxis, :], axis=2) ** 2)


class LaplacianKernel(Kernel):
    """
    Compute the laplacian kernel between X and Y:

        K(x, y) = exp(-gamma ||x-y||_1)
    """

    def __init__(self, gamma='scale'):
        if isinstance(gamma, str):
            if gamma not in ('scale', 'auto'):
                raise ValueError(f'unknown gamma type {gamma}')
        else:
            if not gamma > 0:
                raise ValueError('gamma must be > 0')
        self.gamma = gamma

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        gamma = (1. / (X.shape[1] * X.var()) if self.gamma == 'scale' else  # auto
                 1. / X.shape[1] if isinstance(self.gamma, str) else self.gamma)
        return np.exp(-gamma * np.linalg.norm(X[:, np.newaxis] - Y[np.newaxis, :], ord=1, axis=2))


class SigmoidKernel(Kernel):
    """
    Compute the sigmoid kernel between X and Y:

        K(X, Y) = tanh(gamma <X, Y> + coef0)
    """

    def __init__(self, gamma='scale', coef0=0.):
        if isinstance(gamma, str):
            if gamma not in ('scale', 'auto'):
                raise ValueError(f'unknown gamma type {gamma}')
        else:
            if not gamma > 0:
                raise ValueError('gamma must be > 0')
        self.gamma = gamma
        self.coef0 = coef0

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        gamma = (1. / (X.shape[1] * X.var()) if self.gamma == 'scale' else  # auto
                 1. / X.shape[1] if isinstance(self.gamma, str) else self.gamma)
        return np.tanh(gamma * np.dot(X, Y.T) + self.coef0)


linear = LinearKernel()
poly = PolyKernel()
rbf = RBFKernel()
laplacian = LaplacianKernel()
sigmoid = SigmoidKernel()
