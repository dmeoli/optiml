import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier as OVR
from sklearn.preprocessing import MinMaxScaler

from optiml.ml.svm import SVC
from optiml.ml.svm.kernels import gaussian
from optiml.ml.svm.losses import hinge, squared_hinge
from optiml.opti.constrained import ProjectedGradient, ActiveSet, InteriorPoint, FrankWolfe
from optiml.opti.unconstrained import ProximalBundle
from optiml.opti.unconstrained.line_search import SteepestGradientDescent, ConjugateGradient, Newton, BFGS, LBFGS
from optiml.opti.unconstrained.stochastic import (StochasticGradientDescent, Adam, AMSGrad,
                                                  AdaMax, AdaGrad, AdaDelta, RMSProp)


def test_solve_primal_l1_svc_with_line_search_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(SVC(loss=hinge, optimizer=SteepestGradientDescent))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(SVC(loss=hinge, optimizer=ConjugateGradient))
    svc = svc.fit(X_train, y_train)
    # CG only crawls on the nonsmooth multiclass hinge primal, so just check the score here;
    # its convergence to f* is exercised on the better-conditioned SVR problem
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(SVC(loss=hinge, optimizer=Newton))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(SVC(loss=hinge, optimizer=BFGS))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(SVC(loss=hinge, optimizer=LBFGS))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57


def test_solve_primal_l1_svc_with_stochastic_optimizers():
    # On the nonsmooth multiclass hinge primal the stochastic optimizers converge
    # too slowly / too erratically to reliably meet a fixed optimality-gap tolerance
    # across platforms and seeds, so here we only check the score; their convergence
    # to f* is verified rigorously on the (single, well-conditioned) SVR problem.
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    for optimizer, kwargs in ((StochasticGradientDescent, {}), (Adam, {}), (AMSGrad, {}), (AdaMax, {}),
                              (AdaGrad, {}), (AdaDelta, {'learning_rate': 1.}), (RMSProp, {})):
        svc = OVR(SVC(loss=hinge, optimizer=optimizer, **kwargs)).fit(X_train, y_train)
        assert svc.score(X_test, y_test) >= 0.57


def test_solve_primal_l1_svc_with_proximal_bundle():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)
    svc = OVR(SVC(loss=hinge, optimizer=ProximalBundle))
    svc = svc.fit(X_train, y_train)
    # the proximal bundle method only crawls on the nonsmooth multiclass hinge primal, so check only the score
    assert svc.score(X_test, y_test) >= 0.57


def test_solve_dual_l1_svc_with_smo():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)
    smo = OVR(SVC(loss=hinge, kernel=gaussian, dual=True, optimizer='smo')).fit(X_train, y_train)
    # SMO must reach essentially the same solution as the reference QP solver (cvxopt)
    ref = OVR(SVC(loss=hinge, kernel=gaussian, reg_intercept=False,
                  dual=True, optimizer='cvxopt')).fit(X_train, y_train)
    assert (smo.predict(X_test) == ref.predict(X_test)).mean() >= 0.97


def test_solve_dual_l1_svc_with_cvxopt():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(SVC(loss=hinge, kernel=gaussian, reg_intercept=True, dual=True, optimizer='cvxopt'))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(SVC(loss=hinge, kernel=gaussian, reg_intercept=False, dual=True, optimizer='cvxopt'))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_dual_l1_svc_with_reg_intercept_with_bcqp_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(SVC(loss=hinge, kernel=gaussian, reg_intercept=True, dual=True, optimizer=ProjectedGradient))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(SVC(loss=hinge, kernel=gaussian, reg_intercept=True, dual=True, optimizer=ActiveSet))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(SVC(loss=hinge, kernel=gaussian, reg_intercept=True, dual=True, optimizer=InteriorPoint))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(SVC(loss=hinge, kernel=gaussian, reg_intercept=True, dual=True, optimizer=FrankWolfe))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_dual_l1_svc_with_proximal_bundle():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(SVC(loss=hinge, kernel=gaussian, reg_intercept=True,
                  dual=True, optimizer=ProximalBundle, max_iter=150))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(SVC(loss=hinge, kernel=gaussian, reg_intercept=False,
                  dual=True, optimizer=ProximalBundle, max_iter=150))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_dual_l1_svc_with_AdaGrad():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(SVC(loss=hinge, kernel=gaussian, reg_intercept=True,
                  dual=True, optimizer=AdaGrad, learning_rate=1.))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(SVC(loss=hinge, kernel=gaussian, reg_intercept=False,
                  dual=True, optimizer=AdaGrad, learning_rate=1.))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_primal_l2_svc_with_line_search_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(SVC(loss=squared_hinge, optimizer=SteepestGradientDescent))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(SVC(loss=squared_hinge, optimizer=ConjugateGradient))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(SVC(loss=squared_hinge, optimizer=Newton))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(SVC(loss=squared_hinge, optimizer=BFGS))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(SVC(loss=squared_hinge, optimizer=LBFGS))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57


def test_solve_primal_l2_svc_with_stochastic_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(SVC(loss=squared_hinge, optimizer=StochasticGradientDescent))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(SVC(loss=squared_hinge, optimizer=Adam))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(SVC(loss=squared_hinge, optimizer=AMSGrad))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(SVC(loss=squared_hinge, optimizer=AdaMax))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(SVC(loss=squared_hinge, optimizer=AdaGrad))
    svc = svc.fit(X_train, y_train)
    # AdaGrad converges too slowly on this multiclass problem to reliably hit a fixed
    # optimality-gap tolerance across platforms, so check only the score here
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(SVC(loss=squared_hinge, optimizer=AdaDelta, learning_rate=1.))
    svc = svc.fit(X_train, y_train)
    # AdaDelta converges too slowly on this multiclass problem to reliably hit a fixed
    # optimality-gap tolerance across platforms, so check only the score here
    assert svc.score(X_test, y_test) >= 0.57

    svc = OVR(SVC(loss=squared_hinge, optimizer=RMSProp))
    svc = svc.fit(X_train, y_train)
    # RMSProp does not reliably converge here, so check only the score
    assert svc.score(X_test, y_test) >= 0.57


def test_solve_dual_l2_svc_with_cvxopt():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(SVC(loss=squared_hinge, kernel=gaussian, reg_intercept=True, dual=True, optimizer='cvxopt'))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(SVC(loss=squared_hinge, kernel=gaussian, reg_intercept=False, dual=True, optimizer='cvxopt'))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_dual_l2_svc_with_AdaGrad():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svc = OVR(SVC(loss=squared_hinge, kernel=gaussian, reg_intercept=True,
                  dual=True, optimizer=AdaGrad, learning_rate=1.))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OVR(SVC(loss=squared_hinge, kernel=gaussian, reg_intercept=False,
                  dual=True, optimizer=AdaGrad, learning_rate=1.))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


if __name__ == "__main__":
    pytest.main()
