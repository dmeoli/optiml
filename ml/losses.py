import autograd.numpy as np
from scipy.special import xlogy


def mean_squared_error(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    return np.mean(np.square(y_pred - y_true))


def mean_absolute_error(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    return np.mean(np.abs(y_pred - y_true))


def cross_entropy(y_pred, y_true):
    """Multi-class classification"""
    assert y_pred.shape == y_true.shape
    return -np.mean(xlogy(y_true, y_pred))


def binary_cross_entropy(y_pred, y_true):
    """Multi-label classification"""
    assert y_pred.shape == y_true.shape
    return -np.mean(xlogy(y_true, y_pred) + xlogy(1. - y_true, 1. - y_pred))
