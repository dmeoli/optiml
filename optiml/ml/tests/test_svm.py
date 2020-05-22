import pytest
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from optiml.ml.svm import LinearSVC, SVC, LinearSVR, SVR
from optiml.ml.svm.kernels import linear, rbf
from optiml.ml.svm.losses import hinge, squared_hinge, epsilon_insensitive, squared_epsilon_insensitive
from optiml.optimization.unconstrained.line_search import SteepestGradientDescent
from optiml.optimization.unconstrained.stochastic import StochasticGradientDescent


# def test_solve_linear_svr_as_primal_with_line_search_optimizer():
#     X, y = load_boston(return_X_y=True)
#     X_scaled = StandardScaler().fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
#     svr = LinearSVR(loss=squared_epsilon_insensitive, optimizer=SteepestGradientDescent,
#                     learning_rate=0.1, max_iter=1000)
#     svr.fit(X_train, y_train)
#     assert svr.score(X_test, y_test) >= 0.75


# def test_solve_linear_svr_as_primal_with_stochastic_optimizer():
#     X, y = load_boston(return_X_y=True)
#     X_scaled = StandardScaler().fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
#     svr = LinearSVR(loss=epsilon_insensitive, optimizer=StochasticGradientDescent,
#                     learning_rate=0.01, max_iter=1000)
#     svr.fit(X_train, y_train)
#     assert svr.score(X_test, y_test) >= 0.75


# TODO add solve linear svr as dual and as lagrangian dual relaxation

def test_solve_svr_with_smo():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = SVR(kernel=linear).fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.75


def test_solve_svr_as_qp_with_cvxopt():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = SVR(kernel=linear, optimizer='cvxopt').fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.75


def test_solve_linear_svc_as_primal_with_line_search_optimizer():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(LinearSVC(loss=squared_hinge, penalty='l2', optimizer=SteepestGradientDescent,
                                        learning_rate=0.1, max_iter=1000))
    svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57


def test_solve_linear_svc_as_primal_with_stochastic_optimizer():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(LinearSVC(loss=hinge, penalty='l2', optimizer=StochasticGradientDescent,
                                        learning_rate=0.01, max_iter=1000))
    svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57


# TODO add solve linear svc as dual and as lagrangian dual relaxation

def test_solve_svc_with_smo():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(SVC(kernel=rbf)).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.95


def test_solve_svc_as_qp_with_cvxopt():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(SVC(kernel=rbf, optimizer='cvxopt')).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.95


if __name__ == "__main__":
    pytest.main()
