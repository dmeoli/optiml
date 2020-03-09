import pytest
from sklearn.datasets import load_iris

from ml.learning import MultiClassClassifier
from ml.metrics import accuracy_score
from ml.svm import SVC


def test_svm():
    X, y = load_iris(return_X_y=True)
    svm = MultiClassClassifier(SVC()).fit(X, y)
    assert accuracy_score(y, svm.predict(X)) >= 0.96


if __name__ == "__main__":
    pytest.main()
