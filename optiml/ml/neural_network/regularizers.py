from abc import ABC

import numpy as np


class Regularizer(ABC):
    """
    Base abstract class for all regularizers. A regularizer penalizes the
    magnitude of the parameters and exposes both its ``function`` and its
    ``jacobian``.
    """

    def __init__(self, lmbda=0.):
        """
        Parameters
        ----------

        lmbda : float, default=0.
            Regularization strength. The higher the value, the stronger
            the penalty on the parameters.
        """
        self.lmbda = lmbda

    def function(self, theta):
        raise NotImplementedError

    def jacobian(self, theta):
        raise NotImplementedError

    def __call__(self, theta):
        return self.function(theta)


class L1(Regularizer):
    """
    L1 (Lasso) regularizer:

        R(theta) = lmbda * sum(abs(theta))
    """

    def __init__(self, lmbda=0.):
        """
        Parameters
        ----------

        lmbda : float, default=0.
            Regularization strength.
        """
        super(L1, self).__init__(lmbda)

    def function(self, theta):
        return self.lmbda * np.sum(np.abs(theta))

    def jacobian(self, theta):
        return self.lmbda * np.sign(theta)


class L2(Regularizer):
    """
    L2 (Ridge) regularizer:

        R(theta) = lmbda * sum(theta^2)
    """

    def __init__(self, lmbda=0.):
        """
        Parameters
        ----------

        lmbda : float, default=0.
            Regularization strength.
        """
        super(L2, self).__init__(lmbda)

    def function(self, theta):
        return self.lmbda * np.sum(np.square(theta))

    def jacobian(self, theta):
        return self.lmbda * theta


l1 = L1()
l2 = L2()
