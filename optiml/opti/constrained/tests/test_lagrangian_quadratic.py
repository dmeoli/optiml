import numpy as np
import pytest

from optiml.opti import Quadratic
from optiml.opti.constrained import LagrangianQuadratic
from optiml.opti.unconstrained.stochastic import AdaGrad
from optiml.opti.utils import generate_box_constrained_quadratic


def test_LagrangianQuadratic():
    Q, q, ub = generate_box_constrained_quadratic(ndim=2)
    A, b, lb = [2, 7], np.zeros(1), np.zeros_like(q)
    ld = LagrangianQuadratic(primal=Quadratic(Q, q), A=A, b=b, lb=lb, ub=ub)
    assert np.allclose(AdaGrad(ld, step_size=1., epochs=15000).minimize().x, ld.x_star())


if __name__ == "__main__":
    pytest.main()
