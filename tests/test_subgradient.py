import numpy as np
import pytest

from optimization_test_functions import *
from unconstrained.subgradient import Subgradient


def test_Rosenbrock():
    x0 = [[-1], [1]]

    x, status = Subgradient(Rosenbrock(), x0)
    assert np.allclose(x, [[-0.99606882], [0.99863967]])
    assert status is 'stopped'


def test_Ackley():
    x0 = [[-1], [1]]

    x, status = Subgradient(Ackley(), x0)
    assert np.allclose(x, [[-0.99661898], [0.99661898]])
    assert status is 'stopped'


if __name__ == "__main__":
    pytest.main()
