import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler

from optiml.ml.svm import PrimalSVC, DualSVC
from optiml.ml.svm.kernels import gaussian
from optiml.ml.svm.losses import hinge, squared_hinge
from optiml.opti.constrained import ProjectedGradient, ActiveSet, InteriorPoint, FrankWolfe
from optiml.opti.unconstrained import ProximalBundle
from optiml.opti.unconstrained.line_search import (Subgradient, SteepestGradientDescent, ConjugateGradient,
                                                   HeavyBallGradient, Newton, BFGS)
from optiml.opti.unconstrained.stochastic import (StochasticGradientDescent, Adam, AMSGrad,
                                                  AdaMax, AdaGrad, AdaDelta, RProp, RMSProp)


def test_solve_primal_l1_svc_with_proximal_bundle():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(PrimalSVC(loss=hinge, optimizer=ProximalBundle))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.60


def test_solve_dual_l1_svc_with_smo():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian)).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_dual_l1_svc_with_with_cvxopt():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer='cvxopt', reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer='cvxopt', reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_dual_l1_svc_with_with_bcqp_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=ProjectedGradient))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=ActiveSet))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=InteriorPoint)).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=FrankWolfe)).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_dual_l1_svc_with_reg_intercept_with_line_search_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=Subgradient, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(
        DualSVC(loss=hinge, kernel=gaussian, optimizer=SteepestGradientDescent, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=ConjugateGradient, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=HeavyBallGradient, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_dual_l1_svc_with_unreg_intercept_with_line_search_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=Subgradient, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=SteepestGradientDescent,
                                      reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=ConjugateGradient, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=HeavyBallGradient, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94


def test_solve_dual_l1_svc_with_proximal_bundle():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=ProximalBundle,
                                      max_iter=150, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=ProximalBundle,
                                      max_iter=150, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.84


def test_solve_dual_l1_svc_with_reg_intercept_with_stochastic_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=StochasticGradientDescent,
                                      reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=AdaGrad, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=AdaDelta, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=RProp, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_dual_l1_svc_with_unreg_intercept_with_stochastic_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=StochasticGradientDescent,
                                      reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=AdaGrad, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=AdaDelta, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94

    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer=RProp, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94


def test_solve_primal_l2_svc_with_line_search_optimizers():
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


def test_solve_primal_l2_svc_with_stochastic_optimizers():
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

    svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, optimizer=AdaDelta, learning_rate=1.))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, optimizer=RProp))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, optimizer=RMSProp))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57


def test_solve_dual_l2_svc_with_with_cvxopt():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer='cvxopt', reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer='cvxopt', reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_dual_l2_svc_with_reg_intercept_with_line_search_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian,
                                      optimizer=SteepestGradientDescent, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.92

    svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian,
                                      optimizer=ConjugateGradient, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.92

    svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian,
                                      optimizer=HeavyBallGradient, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.92


def test_solve_dual_l2_svc_with_unreg_intercept_with_line_search_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian,
                                      optimizer=SteepestGradientDescent, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94

    svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian,
                                      optimizer=ConjugateGradient, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94

    svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian,
                                      optimizer=HeavyBallGradient, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94


def test_solve_dual_l2_svc_with_reg_intercept_with_stochastic_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian,
                                      optimizer=StochasticGradientDescent, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.92

    svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian,
                                      optimizer=AdaGrad, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.92

    svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer=AdaDelta,
                                      learning_rate=1., reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.92

    svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian,
                                      optimizer=RProp, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.92


def test_solve_dual_l2_svc_with_unreg_intercept_with_stochastic_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    # TODO needs to be fixed
    # svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian,
    #                                   optimizer=StochasticGradientDescent, reg_intercept=False))
    # svc = svc.fit(X_train, y_train)
    # assert svc.score(X_test, y_test) >= 0.94

    svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian,
                                      optimizer=AdaGrad, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94

    svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian,
                                      optimizer=AdaDelta, learning_rate=1., reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94

    svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian,
                                      optimizer=RProp, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94


if __name__ == "__main__":
    pytest.main()
