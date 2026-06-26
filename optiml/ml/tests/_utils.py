import numpy as np

# Tolerance on the relative optimality gap (f(x) - f*) / f*, used to certify that
# an optimizer actually reached the (solver-certified) primal optimum f* computed
# by `SVMLoss.f_star`. Smooth losses (squared hinge / squared epsilon-insensitive)
# are minimized essentially to machine precision by every method, while the
# nonsmooth ones (hinge / epsilon-insensitive) are only reached within a looser
# tolerance by first-order and subgradient-type methods.
SMOOTH_TOL = 1e-4
NONSMOOTH_TOL = 5e-2


def optimality_gap(model):
    """Relative optimality gap (f(x) - f*) / f* of a fitted primal SVM model."""
    x = np.hstack((model.coef_, model.intercept_))
    return (model.loss(x) - model.loss.f_star()) / model.loss.f_star()


def assert_optimal(model, tol):
    """Assert that a fitted primal SVM model reached its optimum within ``tol``."""
    gap = optimality_gap(model)
    assert gap <= tol, f'relative optimality gap {gap:.2e} exceeds tolerance {tol:.0e}'


def assert_all_optimal(ovr, tol):
    """Assert optimality for each binary estimator of a fitted OvR SVM classifier."""
    for estimator in ovr.estimators_:
        assert_optimal(estimator, tol)
