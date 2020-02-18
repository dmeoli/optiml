import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from ml.svm import MultiSVM


def test_svm():
    X, y = load_iris(return_X_y=True)
    svm = MultiSVM().fit(X, y)
    assert accuracy_score(y, svm.predict(X)) >= 0.96


if __name__ == "__main__":
    pytest.main()
