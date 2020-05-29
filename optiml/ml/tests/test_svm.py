import pytest
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from optiml.ml.svm import PrimalSVC, DualSVC, PrimalSVR, DualSVR
from optiml.ml.svm.kernels import linear, rbf
from optiml.ml.svm.losses import hinge, squared_hinge, epsilon_insensitive, squared_epsilon_insensitive
from optiml.opti.qp.bcqp import ProjectedGradient, ActiveSet, InteriorPoint, FrankWolfe
from optiml.opti.unconstrained.line_search import SteepestGradientDescent
from optiml.opti.unconstrained.stochastic import StochasticGradientDescent, AdaGrad


def test_solve_linear_svr_with_line_search_optimizer():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = PrimalSVR(loss=squared_epsilon_insensitive, optimizer=SteepestGradientDescent,
                    learning_rate=0.1, max_iter=1000)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_linear_svr_with_stochastic_optimizer():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = PrimalSVR(loss=epsilon_insensitive, optimizer=StochasticGradientDescent,
                    learning_rate=0.01, max_iter=1000)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_svr_with_smo():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(kernel=linear).fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_svr_as_qp_with_cvxopt():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(kernel=linear, optimizer='cvxopt').fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


# def test_solve_svr_as_bcqp_lagrangian_relaxation_with_projected_gradient():
#     X, y = load_boston(return_X_y=True)
#     X_scaled = StandardScaler().fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
#     svr = DualSVR(kernel=linear, optimizer=ProjectedGradient).fit(X_train, y_train)
#     assert svr.score(X_test, y_test) >= 0.77


# def test_solve_svr_as_bcqp_lagrangian_relaxation_with_active_set():
#     X, y = load_boston(return_X_y=True)
#     X_scaled = StandardScaler().fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
#     svr = DualSVR(kernel=linear, optimizer=ActiveSet).fit(X_train, y_train)
#     assert svr.score(X_test, y_test) >= 0.77


# def test_solve_svr_as_bcqp_lagrangian_relaxation_with_interior_point():
#     X, y = load_boston(return_X_y=True)
#     X_scaled = StandardScaler().fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
#     svr = DualSVR(kernel=linear, optimizer=InteriorPoint).fit(X_train, y_train)
#     assert svr.score(X_test, y_test) >= 0.77
#
#
# def test_solve_svr_as_bcqp_lagrangian_relaxation_with_frank_wolfe():
#     X, y = load_boston(return_X_y=True)
#     X_scaled = StandardScaler().fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
#     svr = DualSVR(kernel=linear, optimizer=FrankWolfe).fit(X_train, y_train)
#     assert svr.score(X_test, y_test) >= 0.77


# def test_solve_svr_as_qp_lagrangian_relaxation_with_line_search_optimizer():
#     X, y = load_boston(return_X_y=True)
#     X_scaled = StandardScaler().fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
#     svr = DualSVR(kernel=linear, optimizer=SteepestGradientDescent).fit(X_train, y_train)
#     assert svr.score(X_test, y_test) >= 0.77


# def test_solve_svr_as_qp_lagrangian_relaxation_with_stochastic_optimizer():
#     X, y = load_boston(return_X_y=True)
#     X_scaled = StandardScaler().fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
#     svr = DualSVR(kernel=linear, optimizer=AdaGrad).fit(X_train, y_train)
#     assert svr.score(X_test, y_test) >= 0.77


def test_solve_linear_svc_with_line_search_optimizer():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, penalty='l2', optimizer=SteepestGradientDescent,
                                        learning_rate=0.1, max_iter=1000))
    svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57


def test_solve_linear_svc_with_stochastic_optimizer():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(PrimalSVC(loss=hinge, penalty='l2', optimizer=StochasticGradientDescent,
                                        learning_rate=0.01, max_iter=1000))
    svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57


def test_solve_svc_with_smo():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=rbf)).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_as_qp_with_cvxopt():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=rbf, optimizer='cvxopt')).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_as_bcqp_with_projected_gradient():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=rbf, optimizer=ProjectedGradient)).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_as_bcqp_with_active_set():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=rbf, optimizer=ActiveSet)).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_as_bcqp_with_interior_point():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=rbf, optimizer=InteriorPoint)).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_as_bcqp_with_frank_wolfe():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=rbf, optimizer=FrankWolfe)).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_as_bcqp_lagrangian_relaxation_with_line_search_optimizer():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=rbf, optimizer=SteepestGradientDescent)).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_as_bcqp_lagrangian_relaxation_with_stochastic_optimizer():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=rbf, optimizer=AdaGrad)).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


if __name__ == "__main__":
    pytest.main()
