import numpy as np


class Regularizer:
    def __init__(self, lmbda=0.01):
        self.lmbda = lmbda

    def function(self, theta):
        raise NotImplementedError

    def jacobian(self, theta):
        raise NotImplementedError

    def __call__(self, theta):
        return self.function(theta)


class L1(Regularizer):
    def __init__(self, lmbda=0.01):
        super().__init__(lmbda)

    def function(self, theta):
        return self.lmbda * np.sum(np.abs(theta))  # or np.linalg.norm(theta, ord=1) ** 2

    def jacobian(self, theta):
        return self.lmbda * np.sign(theta)


class L2(Regularizer):
    def __init__(self, lmbda=0.01):
        super().__init__(lmbda)

    def function(self, theta):
        return self.lmbda * np.sum(np.square(theta))  # or np.linalg.norm(theta) ** 2

    def jacobian(self, theta):
        return self.lmbda * 2 * theta


l1 = L1()
l2 = L2()
