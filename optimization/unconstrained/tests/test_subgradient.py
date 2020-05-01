import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.line_search.subgradient import Subgradient


def test_quadratic():
    assert np.allclose(Subgradient(quad1, a_start=0.32).minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(Subgradient(quad2, a_start=0.52).minimize()[0], quad2.x_star(), rtol=0.1)


def test_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(Subgradient(obj, a_start=0.052).minimize()[0], obj.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
