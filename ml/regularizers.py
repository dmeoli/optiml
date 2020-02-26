import numpy as np


def l1(theta, lmbda=0.01):
    return lmbda * np.sum(np.abs(theta))


def l2(theta, lmbda=0.01):
    return lmbda * np.sum(np.square(theta))
