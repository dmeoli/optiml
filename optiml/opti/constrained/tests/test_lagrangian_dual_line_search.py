import numpy as np
import pytest

from optiml.opti import Quadratic
from optiml.opti.constrained import LagrangianBoxConstrainedQuadratic, ProjectedGradient
from optiml.opti.unconstrained.line_search import (Subgradient, SteepestGradientDescent, HeavyBallGradient,
                                                   ConjugateGradient, Newton, BFGS)
from optiml.opti.utils import generate_box_constrained_quadratic

Q, q, ub = generate_box_constrained_quadratic(ndim=2, seed=6)
quad = Quadratic(Q, q)
dual = LagrangianBoxConstrainedQuadratic(primal=quad, ub=ub)
x_star = ProjectedGradient(quad=quad, ub=ub).minimize().x


def test_LagrangianDual_with_Subgradient():
    assert np.allclose(Subgradient(f=dual).minimize().primal_x, x_star, rtol=0.1)


def test_LagrangianDual_with_SteepestGradientDescent():
    assert np.allclose(SteepestGradientDescent(f=dual).minimize().primal_x, x_star)


def test_LagrangianDual_with_ConjugateGradient():
    assert np.allclose(ConjugateGradient(f=dual, wf='fr').minimize().primal_x, x_star)
    assert np.allclose(ConjugateGradient(f=dual, wf='hs').minimize().primal_x, x_star)
    assert np.allclose(ConjugateGradient(f=dual, wf='pr').minimize().primal_x, x_star)
    assert np.allclose(ConjugateGradient(f=dual, wf='dy').minimize().primal_x, x_star)


def test_LagrangianDual_with_HeavyBallGradient():
    assert np.allclose(HeavyBallGradient(f=dual).minimize().primal_x, x_star)


# def test_LagrangianDual_with_Newton():
#     assert np.allclose(Newton(f=dual).minimize().primal_x, x_star)


def test_LagrangianDual_with_BFGS():
    assert np.allclose(BFGS(f=dual).minimize().primal_x, x_star)


if __name__ == "__main__":
    pytest.main()
