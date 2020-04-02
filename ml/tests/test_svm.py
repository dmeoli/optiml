import pytest
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from ml.kernels import rbf_kernel, linear_kernel
from ml.svm import SVC, SVR


def test_svr():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    svr = SVR(kernel=linear_kernel).fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.65


def test_svc():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    svc = OneVsOneClassifier(SVC(kernel=rbf_kernel)).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.9


if __name__ == "__main__":
    pytest.main()
