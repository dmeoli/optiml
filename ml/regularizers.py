import numpy as np


class Regularizer:
    def __init__(self, lmbda=0.):
        self.lmbda = lmbda

    def function(self, theta):
        raise NotImplementedError

    def jacobian(self, theta):
        raise NotImplementedError

    def __call__(self, theta):
        return self.function(theta)


class L1(Regularizer):
    def __init__(self, lmbda=0.):
        super().__init__(lmbda)

    def function(self, theta):
        return self.lmbda * np.sum(np.abs(theta))

    def jacobian(self, theta):
        return self.lmbda * np.sign(theta)


class L2(Regularizer):
    def __init__(self, lmbda=0.):
        super().__init__(lmbda)

    def function(self, theta):
        return self.lmbda * np.sum(np.square(theta))

    def jacobian(self, theta):
        return self.lmbda * 2 * theta


l1 = L1()
l2 = L2()
