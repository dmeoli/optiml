import numpy as np
import pytest

from optiml.opti import quad1, quad2
from optiml.opti.unconstrained import Rosenbrock
from optiml.opti.unconstrained.line_search import Subgradient


def test_quadratic():
    assert np.allclose(Subgradient(f=quad1, a_start=0.5).minimize().x, quad1.x_star(), rtol=0.1)
    assert np.allclose(Subgradient(f=quad2, a_start=0.5).minimize().x, quad2.x_star(), rtol=0.1)


def test_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(Subgradient(f=rosen, a_start=0.05).minimize().x, rosen.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
