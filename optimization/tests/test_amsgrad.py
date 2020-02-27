import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.amsgrad import AMSGrad


def test_AMSGrad_standard_momentum_quadratic():
    x, _ = AMSGrad(quad1, momentum_type='standard').minimize()
    assert np.allclose(x, quad1.x_star(), rtol=1e-3)

    x, _ = AMSGrad(quad2, momentum_type='standard').minimize()
    assert np.allclose(x, quad2.x_star(), rtol=1e-3)


def test_AMSGrad_standard_momentum_Rosenbrock():
    obj = Rosenbrock()
    x, _ = AMSGrad(obj, momentum_type='standard').minimize()
    assert np.allclose(x, obj.x_star(), rtol=0.1)


def test_AMSGrad_nesterov_momentum_quadratic():
    x, _ = AMSGrad(quad1, momentum_type='nesterov').minimize()
    assert np.allclose(x, quad1.x_star(), rtol=1e-3)

    x, _ = AMSGrad(quad2, momentum_type='nesterov').minimize()
    assert np.allclose(x, quad2.x_star(), rtol=1e-3)


def test_AMSGrad_nesterov_momentum_Rosenbrock():
    obj = Rosenbrock()
    x, _ = AMSGrad(obj, momentum_type='nesterov').minimize()
    assert np.allclose(x, obj.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
