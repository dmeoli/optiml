import numpy as np
import pytest

from yase.optimization.unconstrained import quad1, quad2, Rosenbrock
from yase.optimization.unconstrained.line_search import BFGS


def test_quadratic():
    assert np.allclose(BFGS(f=quad1, x=np.random.uniform(size=2)).minimize().x, quad1.x_star())
    assert np.allclose(BFGS(f=quad2, x=np.random.uniform(size=2)).minimize().x, quad2.x_star())


def test_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(BFGS(f=rosen, x=np.random.uniform(size=2)).minimize().x, rosen.x_star())


if __name__ == "__main__":
    pytest.main()
