import pytest
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split

from ml.learning import MultiClassClassifier
from ml.metrics import accuracy_score, r2_score
from ml.svm.kernels import linear_kernel
from ml.svm.svm import SVC, SVR


def test_svr():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
    svr = SVR(kernel=linear_kernel).fit(X_train, y_train)
    assert r2_score(svr.predict(X_test), y_test) >= 0.45


def test_svc():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
    svc = MultiClassClassifier(SVC()).fit(X_train, y_train)
    assert accuracy_score(svc.predict(X_test), y_test) >= 0.85


if __name__ == "__main__":
    pytest.main()
