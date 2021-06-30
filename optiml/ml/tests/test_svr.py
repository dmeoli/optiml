import numpy as np
import pytest
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from optiml.ml.svm import SVR
from optiml.ml.svm.kernels import linear
from optiml.ml.svm.losses import epsilon_insensitive, squared_epsilon_insensitive
from optiml.opti.constrained import ProjectedGradient, ActiveSet, InteriorPoint, FrankWolfe
from optiml.opti.unconstrained import ProximalBundle
from optiml.opti.unconstrained.line_search import SteepestGradientDescent, ConjugateGradient, Newton, BFGS
from optiml.opti.unconstrained.stochastic import (StochasticGradientDescent, Adam, AMSGrad,
                                                  AdaMax, AdaGrad, AdaDelta, RMSProp)


def test_solve_primal_l1_svr_with_line_search_optimizers():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svr = SVR(loss=epsilon_insensitive, optimizer=SteepestGradientDescent)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=epsilon_insensitive, optimizer=ConjugateGradient)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=epsilon_insensitive, optimizer=Newton)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=epsilon_insensitive, optimizer=BFGS)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67


def test_solve_primal_l1_svr_with_stochastic_optimizers():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svr = SVR(loss=epsilon_insensitive, optimizer=StochasticGradientDescent)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=epsilon_insensitive, optimizer=Adam)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=epsilon_insensitive, optimizer=AMSGrad)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=epsilon_insensitive, optimizer=AdaMax)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=epsilon_insensitive, optimizer=AdaGrad, learning_rate=1.)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=epsilon_insensitive, optimizer=AdaDelta, learning_rate=1., max_iter=3000)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=epsilon_insensitive, optimizer=RMSProp)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67


def test_solve_primal_l1_svr_with_proximal_bundle():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)
    svr = SVR(loss=epsilon_insensitive, optimizer=ProximalBundle)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.64


def test_solve_dual_l1_svr_with_smo():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)
    svr = SVR(loss=epsilon_insensitive, kernel=linear, dual=True, optimizer='smo')
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.67


def test_solve_dual_l1_svr_with_cvxopt():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svr = SVR(loss=epsilon_insensitive, kernel=linear, reg_intercept=True, dual=True, optimizer='cvxopt')
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=epsilon_insensitive, kernel=linear, reg_intercept=False, dual=True, optimizer='cvxopt')
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.67


def test_solve_dual_l1_svr_with_reg_intercept_with_bcqp_optimizers():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svr = SVR(loss=epsilon_insensitive, kernel=linear, reg_intercept=True, dual=True, optimizer=ProjectedGradient)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=epsilon_insensitive, kernel=linear, reg_intercept=True, dual=True, optimizer=ActiveSet)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=epsilon_insensitive, kernel=linear, reg_intercept=True, dual=True, optimizer=InteriorPoint)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=epsilon_insensitive, kernel=linear, reg_intercept=True, dual=True, optimizer=FrankWolfe)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.67


def test_solve_dual_l1_svr_with_proximal_bundle():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svr = SVR(loss=epsilon_insensitive, kernel=linear, reg_intercept=True,
              dual=True, optimizer=ProximalBundle, max_iter=150)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=epsilon_insensitive, kernel=linear, reg_intercept=False,
              dual=True, optimizer=ProximalBundle, max_iter=150)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.67


def test_solve_dual_l1_svr_with_AdaGrad():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svr = SVR(loss=epsilon_insensitive, kernel=linear, reg_intercept=True,
              dual=True, optimizer=AdaGrad, learning_rate=1.)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=epsilon_insensitive, kernel=linear, reg_intercept=False,
              dual=True, optimizer=AdaGrad, learning_rate=1.)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.67


def test_solve_primal_l2_svr_with_line_search_optimizers():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svr = SVR(loss=squared_epsilon_insensitive, optimizer=SteepestGradientDescent)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=squared_epsilon_insensitive, optimizer=ConjugateGradient)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=squared_epsilon_insensitive, optimizer=Newton)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=squared_epsilon_insensitive, optimizer=BFGS)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67


def test_solve_primal_l2_svr_with_stochastic_optimizers():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svr = SVR(loss=squared_epsilon_insensitive, optimizer=StochasticGradientDescent)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=squared_epsilon_insensitive, optimizer=Adam)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=squared_epsilon_insensitive, optimizer=AMSGrad)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=squared_epsilon_insensitive, optimizer=AdaMax, max_iter=3000)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=squared_epsilon_insensitive, optimizer=AdaGrad, learning_rate=1.)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 1e-4
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=squared_epsilon_insensitive, optimizer=AdaDelta, learning_rate=1., max_iter=5000)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 0.01  # relaxed
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=squared_epsilon_insensitive, optimizer=RMSProp)
    svr.fit(X_train, y_train)
    # (f_t - f*) / f*
    assert (svr.loss(np.hstack((svr.coef_, svr.intercept_))) - svr.loss.f_star()) / svr.loss.f_star() <= 0.01  # relaxed
    assert svr.score(X_test, y_test) >= 0.67


def test_solve_dual_l2_svr_with_cvxopt():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svr = SVR(loss=squared_epsilon_insensitive, kernel=linear, reg_intercept=True, dual=True, optimizer='cvxopt')
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=squared_epsilon_insensitive, kernel=linear, reg_intercept=False, dual=True, optimizer='cvxopt')
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.67


def test_solve_dual_l2_svr_with_AdaGrad():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=123456)

    svr = SVR(loss=squared_epsilon_insensitive, kernel=linear, reg_intercept=True,
              dual=True, optimizer=AdaGrad, learning_rate=1.)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.67

    svr = SVR(loss=squared_epsilon_insensitive, kernel=linear, reg_intercept=False,
              dual=True, optimizer=AdaGrad, learning_rate=1.)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.67


if __name__ == "__main__":
    pytest.main()
