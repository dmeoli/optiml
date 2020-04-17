import pytest
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from ml.svm import SVC, SVR


def test_svr():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75)
    svr = SVR(kernel='linear').fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.6


def test_svc():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    svc = OneVsRestClassifier(SVC(kernel='rbf')).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.9


if __name__ == "__main__":
    pytest.main()
