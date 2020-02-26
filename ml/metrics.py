import numpy as np


def accuracy_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return np.mean(np.equal(y_true, y_pred))


def mean_euclidean_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return np.mean([np.linalg.norm(t - p) for t, p in zip(y_true, y_pred)])
