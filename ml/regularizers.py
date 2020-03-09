import numpy as np


class Regularizer:
    def __init__(self, lmbda=0.01):
        self.lmbda = lmbda

    def function(self, theta):
        raise NotImplementedError

    def derivative(self, theta):
        raise NotImplementedError

    def __call__(self, theta):
        return self.function(theta)


class L1(Regularizer):
    def __init__(self, lmbda=0.01):
        super().__init__(lmbda)

    def function(self, theta):
        return self.lmbda * np.sum(np.abs(theta))


class L2(Regularizer):
    def __init__(self, lmbda=0.01):
        super().__init__(lmbda)

    def function(self, theta):
        return self.lmbda * np.sum(np.square(theta))

    def derivative(self, theta):
        return self.lmbda * 2 * theta


l1 = L1()
l2 = L2()
