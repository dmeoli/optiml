import pytest
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from optiml.ml.svm import PrimalSVC, DualSVC, PrimalSVR, DualSVR
from optiml.ml.svm.kernels import linear, gaussian
from optiml.ml.svm.losses import hinge, squared_hinge, epsilon_insensitive, squared_epsilon_insensitive
from optiml.opti.constrained import ProjectedGradient, ActiveSet, InteriorPoint, FrankWolfe
from optiml.opti.unconstrained import ProximalBundle
from optiml.opti.unconstrained.line_search import (SteepestGradientDescent, ConjugateGradient,
                                                   HeavyBallGradient, Newton, BFGS)
from optiml.opti.unconstrained.stochastic import (StochasticGradientDescent, Adam, AMSGrad,
                                                  AdaMax, AdaGrad, AdaDelta, RProp, RMSProp)


def test_solve_linear_svr_with_line_search_optimizers():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svr = PrimalSVR(loss=squared_epsilon_insensitive, optimizer=SteepestGradientDescent)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77

    svr = PrimalSVR(loss=squared_epsilon_insensitive, optimizer=ConjugateGradient)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77

    svr = PrimalSVR(loss=squared_epsilon_insensitive, optimizer=HeavyBallGradient)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77

    svr = PrimalSVR(loss=squared_epsilon_insensitive, optimizer=Newton)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77

    svr = PrimalSVR(loss=squared_epsilon_insensitive, optimizer=BFGS)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_linear_svr_with_stochastic_optimizers():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svr = PrimalSVR(loss=squared_epsilon_insensitive, optimizer=StochasticGradientDescent)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77

    svr = PrimalSVR(loss=squared_epsilon_insensitive, optimizer=Adam)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77

    svr = PrimalSVR(loss=squared_epsilon_insensitive, optimizer=AMSGrad)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77

    svr = PrimalSVR(loss=squared_epsilon_insensitive, optimizer=AdaMax)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77

    # svr = PrimalSVR(loss=squared_epsilon_insensitive, optimizer=AdaGrad)
    # svr.fit(X_train, y_train)
    # assert svr.score(X_test, y_test) >= 0.77

    # svr = PrimalSVR(loss=squared_epsilon_insensitive, optimizer=AdaDelta)
    # svr.fit(X_train, y_train)
    # assert svr.score(X_test, y_test) >= 0.77

    svr = PrimalSVR(loss=squared_epsilon_insensitive, optimizer=RProp)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77

    svr = PrimalSVR(loss=squared_epsilon_insensitive, optimizer=RMSProp)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_linear_svr_with_proximal_bundle():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = PrimalSVR(loss=epsilon_insensitive, optimizer=ProximalBundle)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_svr_with_smo():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(kernel=linear)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_svr_as_bcqp_with_cvxopt():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(kernel=linear, optimizer='cvxopt', use_explicit_eq=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_svr_as_qp_with_cvxopt():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(kernel=linear, optimizer='cvxopt', use_explicit_eq=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_svr_as_bcqp_with_projected_gradient():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(kernel=linear, optimizer=ProjectedGradient)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_svr_as_bcqp_with_active_set():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(kernel=linear, optimizer=ActiveSet, nonposdef_solver='minres')
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_svr_as_bcqp_with_interior_point():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(kernel=linear, optimizer=InteriorPoint)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_svr_as_bcqp_with_frank_wolfe():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(kernel=linear, optimizer=FrankWolfe)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_svr_as_bcqp_lagrangian_relaxation_with_stochastic_optimizers():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svr = DualSVR(kernel=linear, optimizer=StochasticGradientDescent,
                  nonposdef_solver='minres', use_explicit_eq=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.53

    svr = DualSVR(kernel=linear, optimizer=Adam, nonposdef_solver='minres', use_explicit_eq=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.53

    svr = DualSVR(kernel=linear, optimizer=AMSGrad, nonposdef_solver='minres', use_explicit_eq=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.53

    svr = DualSVR(kernel=linear, optimizer=AdaMax, nonposdef_solver='minres', use_explicit_eq=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.53

    svr = DualSVR(kernel=linear, optimizer=AdaGrad, nonposdef_solver='minres', use_explicit_eq=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.53

    svr = DualSVR(kernel=linear, optimizer=AdaDelta, nonposdef_solver='minres', use_explicit_eq=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.53

    svr = DualSVR(kernel=linear, optimizer=RProp, nonposdef_solver='minres', use_explicit_eq=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.53

    svr = DualSVR(kernel=linear, optimizer=RMSProp, nonposdef_solver='minres', use_explicit_eq=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.53


def test_solve_svr_as_qp_lagrangian_relaxation_with_stochastic_optimizers():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svr = DualSVR(kernel=linear, optimizer=StochasticGradientDescent,
                  nonposdef_solver='minres', use_explicit_eq=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.48

    svr = DualSVR(kernel=linear, optimizer=Adam, nonposdef_solver='minres', use_explicit_eq=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.48

    svr = DualSVR(kernel=linear, optimizer=AMSGrad, nonposdef_solver='minres', use_explicit_eq=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.48

    svr = DualSVR(kernel=linear, optimizer=AdaMax, nonposdef_solver='minres', use_explicit_eq=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.48

    svr = DualSVR(kernel=linear, optimizer=AdaGrad, nonposdef_solver='minres', use_explicit_eq=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.48

    svr = DualSVR(kernel=linear, optimizer=AdaDelta, nonposdef_solver='minres', use_explicit_eq=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.48

    svr = DualSVR(kernel=linear, optimizer=RProp, nonposdef_solver='minres', use_explicit_eq=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.48

    svr = DualSVR(kernel=linear, optimizer=RMSProp, nonposdef_solver='minres', use_explicit_eq=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.48


def test_solve_linear_svc_with_line_search_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, optimizer=SteepestGradientDescent))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, optimizer=ConjugateGradient))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, optimizer=HeavyBallGradient))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, optimizer=Newton))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, optimizer=BFGS))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57


def test_solve_linear_svc_with_stochastic_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, optimizer=StochasticGradientDescent))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, optimizer=Adam))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, optimizer=AMSGrad))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, optimizer=AdaMax))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, optimizer=AdaGrad))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    # svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, optimizer=AdaDelta))
    # svc = svc.fit(X_train, y_train)
    # assert svc.score(X_test, y_test) >= 0.57

    svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, optimizer=RProp))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, optimizer=RMSProp))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57


def test_solve_linear_svc_with_proximal_bundle():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(PrimalSVC(loss=hinge, optimizer=ProximalBundle))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57


def test_solve_svc_with_smo():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=gaussian)).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_as_bcqp_with_cvxopt():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer='cvxopt', use_explicit_eq=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_as_qp_with_cvxopt():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer='cvxopt', use_explicit_eq=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_as_bcqp_with_projected_gradient():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=ProjectedGradient))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_as_bcqp_with_active_set():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=ActiveSet, nonposdef_solver='minres'))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_as_bcqp_with_interior_point():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=InteriorPoint)).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_as_bcqp_with_frank_wolfe():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=FrankWolfe)).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_as_bcqp_lagrangian_relaxation_with_stochastic_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=StochasticGradientDescent,
                                      nonposdef_solver='minres', use_explicit_eq=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=Adam, nonposdef_solver='minres',
                                      use_explicit_eq=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=AMSGrad, nonposdef_solver='minres',
                                      use_explicit_eq=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=AdaMax, nonposdef_solver='minres',
                                      use_explicit_eq=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=AdaGrad, nonposdef_solver='minres',
                                      use_explicit_eq=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=AdaDelta, nonposdef_solver='minres',
                                      use_explicit_eq=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=RProp, nonposdef_solver='minres',
                                      use_explicit_eq=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=RMSProp, nonposdef_solver='minres',
                                      use_explicit_eq=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_as_qp_lagrangian_relaxation_with_stochastic_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=StochasticGradientDescent,
                                      nonposdef_solver='minres', use_explicit_eq=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=Adam, nonposdef_solver='minres',
                                      use_explicit_eq=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=AMSGrad, nonposdef_solver='minres',
                                      use_explicit_eq=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=AdaMax, nonposdef_solver='minres',
                                      use_explicit_eq=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=AdaGrad, nonposdef_solver='minres',
                                      use_explicit_eq=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=AdaDelta, nonposdef_solver='minres',
                                      use_explicit_eq=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=RProp, nonposdef_solver='minres',
                                      use_explicit_eq=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=RMSProp, nonposdef_solver='minres',
                                      use_explicit_eq=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


if __name__ == "__main__":
    pytest.main()
