import numpy as np
import pytest

from optiml.opti import quad1, quad2
from optiml.opti.unconstrained import Rosenbrock
from optiml.opti.unconstrained.line_search import ConjugateGradient


def test_ConjugateGradient_quadratic():
    assert np.allclose(ConjugateGradient(f=quad1).minimize().x, quad1.x_star())
    assert np.allclose(ConjugateGradient(f=quad2).minimize().x, quad2.x_star())


def test_ConjugateGradient_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(ConjugateGradient(f=rosen).minimize().x, rosen.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
