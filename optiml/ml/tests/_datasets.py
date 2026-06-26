import os
import urllib.request

import numpy as np
import pytest

# The Boston house-prices dataset has been removed from scikit-learn since version
# 1.2 due to ethical concerns. To keep the existing regression tests (and their
# accuracy thresholds) reproducible, it is loaded here directly from its original
# source, exactly as suggested in the scikit-learn deprecation notice, and cached
# locally to avoid downloading it more than once.

_BOSTON_URL = 'http://lib.stat.cmu.edu/datasets/boston'
_BOSTON_CACHE = os.path.join(os.path.dirname(__file__), 'data', 'boston.npz')


def load_boston(return_X_y=True):
    """
    Load and return the Boston house-prices dataset (regression).

    The data (506 samples, 13 features) is fetched from its original StatLib
    source and cached locally; if it is not available (e.g., no network) the
    calling test is skipped rather than failed.

    :param return_X_y: (bool, default True): if True return ``(data, target)``,
                       otherwise the same tuple (kept for API compatibility with
                       the former ``sklearn.datasets.load_boston``).
    :return:           ``(X, y)`` with X of shape (506, 13) and y of shape (506,).
    """
    if os.path.exists(_BOSTON_CACHE):
        with np.load(_BOSTON_CACHE) as cache:
            return cache['data'], cache['target']

    try:
        with urllib.request.urlopen(_BOSTON_URL, timeout=30) as response:
            raw = response.read().decode()
    except Exception as e:  # no network or source unavailable
        pytest.skip(f'Boston dataset not available: {e}')

    # the 22 header lines are textual; the rest is a flat stream of 506 * 14
    # floating point numbers (13 features + the target) laid out over two lines
    # per record, so it is enough to parse all the numeric tokens and reshape
    values = np.array(' '.join(raw.splitlines()[22:]).split(), dtype=float).reshape(-1, 14)
    data, target = values[:, :13], values[:, 13]

    os.makedirs(os.path.dirname(_BOSTON_CACHE), exist_ok=True)
    np.savez(_BOSTON_CACHE, data=data, target=target)

    return data, target
