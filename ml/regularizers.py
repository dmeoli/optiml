import numpy as np


class Regularizer:
    def __init__(self, lmbda=0.01):
        self.lmbda = lmbda


class L1(Regularizer):
    def __init__(self, lmbda=0.01):
        super().__init__(lmbda)

    def __call__(self, theta):
        return self.lmbda * np.sum(np.abs(theta))


class L2(Regularizer):
    def __init__(self, lmbda=0.01):
        super().__init__(lmbda)

    def __call__(self, theta):
        return self.lmbda * np.sum(np.square(theta))
