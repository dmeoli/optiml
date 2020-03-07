import numpy as np


def accuracy_score(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    return np.mean(np.equal(y_pred, y_true))


def mean_euclidean_error(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    return np.mean(np.linalg.norm(y_pred - y_true, axis=y_true.ndim - 1))


def r2_score(y_pred, y_true):
    return 1. - (np.sum(np.square(y_pred - y_true)) /  # sum of square of residuals
                 np.sum(np.square(y_true - np.mean(y_true))))  # total sum of squares
