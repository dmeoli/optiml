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
from optiml.opti.unconstrained.line_search import (Subgradient, SteepestGradientDescent, ConjugateGradient,
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

    svr = PrimalSVR(loss=squared_epsilon_insensitive, optimizer=AdaGrad, learning_rate=1.)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77

    svr = PrimalSVR(loss=squared_epsilon_insensitive, optimizer=AdaDelta, learning_rate=1.)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.74

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
    assert svr.score(X_test, y_test) >= 0.76


def test_solve_svr_with_smo():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(kernel=linear)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_svr_with_unreg_bias_with_with_cvxopt():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(loss=epsilon_insensitive, kernel=linear, optimizer='cvxopt', reg_bias=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77

    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(loss=squared_epsilon_insensitive, kernel=linear, optimizer='cvxopt', reg_bias=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_svr_with_reg_bias_with_with_cvxopt():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(loss=epsilon_insensitive, kernel=linear, optimizer='cvxopt', reg_bias=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77

    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(loss=squared_epsilon_insensitive, kernel=linear, optimizer='cvxopt', reg_bias=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_svr_with_unreg_bias_with_with_projected_gradient():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(kernel=linear, optimizer=ProjectedGradient)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_svr_with_unreg_bias_with_with_active_set():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(kernel=linear, optimizer=ActiveSet)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_svr_with_unreg_bias_with_with_interior_point():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(kernel=linear, optimizer=InteriorPoint)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_svr_with_unreg_bias_with_with_frank_wolfe():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svr = DualSVR(kernel=linear, optimizer=FrankWolfe)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.77


def test_solve_svr_with_unreg_bias_with_lagrangian_relaxation_with_line_search_optimizers():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svr = DualSVR(kernel=linear, optimizer=Subgradient, reg_bias=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.53

    svr = DualSVR(kernel=linear, optimizer=SteepestGradientDescent, reg_bias=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.53

    svr = DualSVR(kernel=linear, optimizer=ConjugateGradient, reg_bias=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.53

    svr = DualSVR(kernel=linear, optimizer=HeavyBallGradient, reg_bias=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.53


def test_solve_svr_with_reg_bias_with_lagrangian_relaxation_with_line_search_optimizers():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svr = DualSVR(kernel=linear, optimizer=Subgradient, reg_bias=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.48

    svr = DualSVR(kernel=linear, optimizer=SteepestGradientDescent, reg_bias=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.48

    svr = DualSVR(kernel=linear, optimizer=ConjugateGradient, reg_bias=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.48

    svr = DualSVR(kernel=linear, optimizer=HeavyBallGradient, reg_bias=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.48


def test_solve_svr_with_unreg_bias_with_lagrangian_relaxation_with_proximal_bundle():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svr = DualSVR(kernel=linear, optimizer=ProximalBundle, max_iter=150, reg_bias=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.53


def test_solve_svr_with_reg_bias_with_lagrangian_relaxation_with_proximal_bundle():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svr = DualSVR(kernel=linear, optimizer=ProximalBundle, max_iter=150, reg_bias=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.48


def test_solve_svr_with_unreg_bias_with_lagrangian_relaxation_with_stochastic_optimizers():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svr = DualSVR(kernel=linear, optimizer=StochasticGradientDescent, reg_bias=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.53

    svr = DualSVR(kernel=linear, optimizer=AdaGrad, reg_bias=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.53

    svr = DualSVR(kernel=linear, optimizer=AdaDelta, reg_bias=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.53

    svr = DualSVR(kernel=linear, optimizer=RProp, reg_bias=False)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.53


def test_solve_svr_with_reg_bias_with_lagrangian_relaxation_with_stochastic_optimizers():
    X, y = load_boston(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svr = DualSVR(kernel=linear, optimizer=StochasticGradientDescent, reg_bias=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.48

    svr = DualSVR(kernel=linear, optimizer=AdaGrad, reg_bias=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.48

    svr = DualSVR(kernel=linear, optimizer=AdaDelta, reg_bias=True)
    svr.fit(X_train, y_train)
    assert svr.score(X_test, y_test) >= 0.48

    svr = DualSVR(kernel=linear, optimizer=RProp, reg_bias=True)
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

    svc = OneVsRestClassifier(PrimalSVC(loss=squared_hinge, optimizer=AdaDelta, learning_rate=1.))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.57

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


def test_solve_svc_with_unreg_bias_with_with_cvxopt():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer='cvxopt', reg_bias=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer='cvxopt', reg_bias=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_with_reg_bias_with_with_cvxopt():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(loss=hinge, kernel=gaussian, optimizer='cvxopt', reg_bias=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(loss=squared_hinge, kernel=gaussian, optimizer='cvxopt', reg_bias=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_with_unreg_bias_with_with_projected_gradient():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=ProjectedGradient))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_with_unreg_bias_with_with_active_set():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=ActiveSet))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_with_unreg_bias_with_with_interior_point():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=InteriorPoint)).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_with_unreg_bias_with_with_frank_wolfe():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)
    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=FrankWolfe)).fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_with_unreg_bias_with_lagrangian_relaxation_with_line_search_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=Subgradient, reg_bias=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=SteepestGradientDescent, reg_bias=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=ConjugateGradient, reg_bias=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=HeavyBallGradient, reg_bias=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_with_reg_bias_with_lagrangian_relaxation_with_line_search_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=Subgradient, reg_bias=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=SteepestGradientDescent, reg_bias=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=ConjugateGradient, reg_bias=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=HeavyBallGradient, reg_bias=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94


def test_solve_svc_with_unreg_bias_with_lagrangian_relaxation_with_proximal_bundle():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=ProximalBundle, max_iter=150, reg_bias=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_with_reg_bias_with_lagrangian_relaxation_with_proximal_bundle():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=ProximalBundle, max_iter=150, reg_bias=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.84


def test_solve_svc_with_unreg_bias_with_lagrangian_relaxation_with_stochastic_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=StochasticGradientDescent, reg_bias=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=AdaGrad, reg_bias=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=AdaDelta, reg_bias=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=RProp, reg_bias=False))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.97


def test_solve_svc_with_reg_bias_with_lagrangian_relaxation_with_stochastic_optimizers():
    X, y = load_iris(return_X_y=True)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, random_state=1)

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=StochasticGradientDescent, reg_bias=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=AdaGrad, reg_bias=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=AdaDelta, reg_bias=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94

    svc = OneVsRestClassifier(DualSVC(kernel=gaussian, optimizer=RProp, reg_bias=True))
    svc = svc.fit(X_train, y_train)
    assert svc.score(X_test, y_test) >= 0.94


if __name__ == "__main__":
    pytest.main()
