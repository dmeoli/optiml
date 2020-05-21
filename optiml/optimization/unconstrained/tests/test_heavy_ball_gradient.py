import numpy as np
import pytest

from optiml.optimization.unconstrained import quad2, quad1, Rosenbrock
from optiml.optimization.unconstrained.line_search import HeavyBallGradient


def test_quadratic():
    assert np.allclose(HeavyBallGradient(f=quad1, x=np.random.uniform(size=2)).minimize().x, quad1.x_star())
    assert np.allclose(HeavyBallGradient(f=quad2, x=np.random.uniform(size=2)).minimize().x, quad2.x_star())


def test_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(HeavyBallGradient(f=rosen, x=np.random.uniform(size=2)).minimize().x, rosen.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
