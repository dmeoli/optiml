import numpy as np
import pytest

from yase.optimization.optimizer import quad1, quad2, Rosenbrock
from yase.optimization.unconstrained.line_search import Subgradient


def test_quadratic():
    assert np.allclose(Subgradient(f=quad1, x=np.random.uniform(size=2), a_start=0.32).minimize()[0],
                       quad1.x_star(), rtol=0.1)
    assert np.allclose(Subgradient(f=quad2, x=np.random.uniform(size=2), a_start=0.52).minimize()[0],
                       quad2.x_star(), rtol=0.1)


def test_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(Subgradient(f=rosen, x=np.random.uniform(size=2), a_start=0.052).minimize()[0],
                       rosen.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
