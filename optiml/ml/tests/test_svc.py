import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier as OVR
from sklearn.preprocessing import MinMaxScaler

from optiml.ml.svm import PrimalSVC, DualSVC
from optiml.ml.svm.kernels import gaussian
from optiml.ml.svm.losses import hinge, squared_hinge
from optiml.opti.constrained import ProjectedGradient, ActiveSet, InteriorPoint, FrankWolfe
from optiml.opti.unconstrained import ProximalBundle
from optiml.opti.unconstrained.line_search import SteepestGradientDescent, ConjugateGradient, Newton, BFGS
from optiml.opti.unconstrained.stochastic import (StochasticGradientDescent, Adam, AMSGrad,
                                                  AdaMax, AdaGrad, AdaDelta, RMSProp)


def test_solve_primal_l1_svc_with_line_search_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(PrimalSVC(loss=hinge, optimizer=SteepestGradientDescent))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=hinge, optimizer=ConjugateGradient))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=hinge, optimizer=Newton))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=hinge, optimizer=BFGS))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57


def test_solve_primal_l1_svc_with_stochastic_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(PrimalSVC(loss=hinge, optimizer=StochasticGradientDescent))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=hinge, optimizer=Adam))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=hinge, optimizer=AMSGrad))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=hinge, optimizer=AdaMax))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=hinge, optimizer=AdaGrad))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=hinge, optimizer=AdaDelta, learning_rate=1.))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=hinge, optimizer=RMSProp))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57


def test_solve_primal_l1_svc_with_proximal_bundle():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)
    svc = OVR(PrimalSVC(loss=hinge, optimizer=ProximalBundle))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57


def test_solve_dual_l1_svc_with_smo():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)
    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer='smo'))
    svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_dual_l1_svc_with_cvxopt():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer='cvxopt', reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer='cvxopt', reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_dual_l1_svc_with_reg_intercept_with_bcqp_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=ProjectedGradient, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=ActiveSet, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=InteriorPoint, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=FrankWolfe, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_dual_l1_svc_with_proximal_bundle():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=ProximalBundle, max_iter=150, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=ProximalBundle, max_iter=150, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_dual_l1_svc_with_reg_intercept_with_adaptive_stochastic_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=Adam, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=AMSGrad, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=AdaMax, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=AdaGrad, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=AdaDelta,
                      learning_rate=1., reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=RMSProp, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_dual_l1_svc_with_unreg_intercept_with_adaptive_stochastic_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=Adam, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=AMSGrad, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=AdaMax, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=AdaGrad, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=AdaDelta,
                      learning_rate=1., reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=hinge, kernel=gaussian, optimizer=RMSProp, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_primal_l2_svc_with_line_search_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(PrimalSVC(loss=squared_hinge, optimizer=SteepestGradientDescent))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=squared_hinge, optimizer=ConjugateGradient))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=squared_hinge, optimizer=Newton))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=squared_hinge, optimizer=BFGS))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57


def test_solve_primal_l2_svc_with_stochastic_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(PrimalSVC(loss=squared_hinge, optimizer=StochasticGradientDescent))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=squared_hinge, optimizer=Adam))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=squared_hinge, optimizer=AMSGrad))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=squared_hinge, optimizer=AdaMax))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=squared_hinge, optimizer=AdaGrad))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=squared_hinge, optimizer=AdaDelta, learning_rate=1.))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(PrimalSVC(loss=squared_hinge, optimizer=RMSProp))
    svc = svc.fit(X_train, y_train)
    assert (np.allclose(np.hstack((estimator.coef_, estimator.intercept_)), estimator.loss.x_star())
            for estimator in svc.estimators_)
    assert svc.score(X_test, y_test) >= 0.57


def test_solve_dual_l2_svc_with_cvxopt():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer='cvxopt', reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer='cvxopt', reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_dual_l2_svc_with_reg_intercept_with_adaptive_stochastic_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer=Adam, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer=AMSGrad, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer=AdaMax, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer=AdaGrad, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer=AdaDelta,
                      learning_rate=1., reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer=RMSProp, reg_intercept=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_dual_l2_svc_with_unreg_intercept_with_adaptive_stochastic_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer=Adam, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer=AMSGrad, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer=AdaMax, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer=AdaGrad, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer=AdaDelta,
                      learning_rate=1., reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer=RMSProp, reg_intercept=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


if __name__ == "__main__":
    pytest.main()
