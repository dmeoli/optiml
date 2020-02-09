import numpy as np


def mean_squared_error(y, y_pred):
    return ((y - y_pred) ** 2).mean()


def r2_score(y, y_pred):
    return 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))


def accuracy(predictions, labels):
    assert predictions.shape == labels.shape
    p, l = predictions.astype(np.int32), labels.astype(np.int32)
    return np.where(p == l, 1., 0.).mean()


def true_pos(predictions, labels):
    return np.count_nonzero((predictions == 1) & (labels == 1))


def true_neg(predictions, labels):
    return np.count_nonzero((predictions == 0) & (labels == 0))


def false_pos(predictions, labels):
    return np.count_nonzero((predictions == 1) & (labels == 0))


def false_neg(predictions, labels):
    return np.count_nonzero((predictions == 0) & (labels == 1))


def true_pos_rate(predictions, labels):
    # TP / (FP + FN) = TP / P
    p, l = predictions, labels
    return true_pos(p, l) / np.count_nonzero(labels)


def false_pos_rate(predictions, labels):
    # FP / (FP + TN) = FP / N
    p, l = predictions, labels
    return false_neg(p, l) / (len(labels) - np.count_nonzero(labels))
