import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor

from ml.metrics import accuracy_score, mean_euclidean_error
from ml.kernels import rbf_kernel
from ml.svm import SVC, SVR
from utils import load_ml_cup


def test_svr():
    X, y = load_ml_cup()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    svr = MultiOutputRegressor(SVR(kernel=rbf_kernel)).fit(X_train, y_train)
    assert mean_euclidean_error(svr.predict(X_test), y_test) <= 1.2


def test_svc():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    svc = OneVsRestClassifier(SVC(kernel=rbf_kernel)).fit(X_train, y_train)
    assert accuracy_score(svc.predict(X_test), y_test) >= 0.9


if __name__ == "__main__":
    pytest.main()
