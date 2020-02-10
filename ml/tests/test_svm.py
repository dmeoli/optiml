import numpy as np
import pytest
from sklearn.datasets import load_iris

from ml.svm import MultiSVM

X, y = load_iris(return_X_y=True)
X, y = X, y


def test_svm():
    svm = MultiSVM().fit(X, y)
    # assert grade_learner(svm, iris_tests) == 1
    # assert np.isclose(err_ratio(svm, X, y), 0.04)


if __name__ == "__main__":
    pytest.main()
