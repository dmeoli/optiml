import numpy as np


def accuracy_score(y_pred, y_true):
    return np.mean(np.equal(y_pred, y_true))


def mean_euclidean_error(y_pred, y_true):
    return np.mean([np.linalg.norm(p - t) for t, p in zip(y_pred, y_true)])
