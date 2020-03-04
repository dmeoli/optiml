import numpy as np


def accuracy_score(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    return np.mean(np.equal(y_pred, y_true))


def mean_euclidean_error(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    return np.mean(np.linalg.norm(y_pred - y_true, axis=y_true.ndim - 1))
