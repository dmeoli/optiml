import numpy as np


def accuracy_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return np.where(y_true == y_pred, 1., 0.).mean()


def mean_squared_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return np.mean(np.square(y_true - y_pred))


def mean_absolute_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return np.mean(np.abs(y_true - y_pred))


def mean_euclidean_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return np.sum([np.linalg.norm(t - p) for t, p in zip(y_true, y_pred)]) / y_true.shape[0]
