import numpy as np
import pytest

from optiml.opti import Quadratic
from optiml.opti.constrained import LagrangianBoxConstrainedQuadratic, LagrangianDual, ProjectedGradient
from optiml.opti.unconstrained.line_search import (Subgradient, SteepestGradientDescent, HeavyBallGradient,
                                                   ConjugateGradient, BFGS)
from optiml.opti.utils import generate_box_constrained_quadratic

Q, q, ub = generate_box_constrained_quadratic(ndim=2)
quad = Quadratic(Q, q)
dual = LagrangianBoxConstrainedQuadratic(quad=quad, ub=ub)
x_star = ProjectedGradient(quad=quad, ub=ub).minimize().x


def test_LagrangianDual_with_Subgradient():
    assert np.allclose(LagrangianDual(f=dual, optimizer=Subgradient).minimize().primal_x, x_star, rtol=0.1)


def test_LagrangianDual_with_SteepestGradientDescent():
    assert np.allclose(LagrangianDual(f=dual, optimizer=SteepestGradientDescent).minimize().primal_x, x_star, rtol=0.1)


def test_LagrangianDual_with_ConjugateGradient():
    assert np.allclose(LagrangianDual(f=dual, optimizer=ConjugateGradient, wf='fr').minimize().primal_x,
                       x_star, rtol=0.1)
    assert np.allclose(LagrangianDual(f=dual, optimizer=ConjugateGradient, wf='hs').minimize().primal_x,
                       x_star, rtol=0.1)
    assert np.allclose(LagrangianDual(f=dual, optimizer=ConjugateGradient, wf='pr').minimize().primal_x,
                       x_star, rtol=0.1)
    assert np.allclose(LagrangianDual(f=dual, optimizer=ConjugateGradient, wf='dy').minimize().primal_x,
                       x_star, rtol=0.1)


def test_LagrangianDual_with_HeavyBallGradient():
    assert np.allclose(LagrangianDual(f=dual, optimizer=HeavyBallGradient).minimize().primal_x, x_star, rtol=0.1)


def test_LagrangianDual_with_BFGS():
    assert np.allclose(LagrangianDual(f=dual, optimizer=BFGS).minimize().primal_x, x_star, rtol=0.1)


if __name__ == "__main__":
    pytest.main()
