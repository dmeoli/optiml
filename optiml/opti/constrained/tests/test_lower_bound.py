import numpy as np
import pytest

from optiml.opti import Quadratic
from optiml.opti.constrained import ProjectedGradient, ActiveSet, InteriorPoint, FrankWolfe
from optiml.opti.utils import generate_box_constrained_quadratic


@pytest.mark.parametrize('optimizer', [ProjectedGradient, ActiveSet, InteriorPoint, FrankWolfe])
def test_generic_lower_bound(optimizer):
    # the box-constrained optimizers must also handle a generic (nonzero) lower bound
    # lb <= x <= ub, not just the default 0 <= x <= ub
    Q, q, ub = generate_box_constrained_quadratic(ndim=5, seed=7)
    quad = Quadratic(Q, q)
    lb = ub / 4  # a nonzero lower bound strictly inside the box

    bcqp = optimizer(quad=quad, ub=ub, lb=lb)
    x = bcqp.minimize().x

    # the returned point must be feasible ...
    assert np.all(x >= lb - 1e-6) and np.all(x <= ub + 1e-6)
    # ... and (near) optimal, i.e., its objective matches the reference optimum
    # (x_star solves the same lb <= x <= ub program with a reliable QP solver)
    f_opt = quad.function(bcqp.x_star())
    assert (quad.function(x) - f_opt) / max(abs(f_opt), 1) <= 1e-2


if __name__ == "__main__":
    pytest.main()
