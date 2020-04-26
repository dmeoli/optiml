import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.stochastic.amsgrad import AMSGrad


def test_AMSGrad_quadratic():
    assert np.allclose(AMSGrad(quad1, step_rate=0.1).minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(AMSGrad(quad2, step_rate=0.1).minimize()[0], quad2.x_star(), rtol=0.1)


def test_AMSGrad_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(AMSGrad(obj, step_rate=0.1).minimize()[0], obj.x_star(), rtol=0.1)


def test_AMSGrad_standard_momentum_quadratic():
    assert np.allclose(AMSGrad(quad1, momentum_type='standard').minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(AMSGrad(quad2, momentum_type='standard').minimize()[0], quad2.x_star(), rtol=0.1)


def test_AMSGrad_standard_momentum_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(AMSGrad(obj, momentum_type='standard').minimize()[0], obj.x_star(), rtol=0.1)


def test_AMSGrad_nesterov_momentum_quadratic():
    assert np.allclose(AMSGrad(quad1, momentum_type='nesterov').minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(AMSGrad(quad2, momentum_type='nesterov').minimize()[0], quad2.x_star(), rtol=0.1)


def test_AMSGrad_nesterov_momentum_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(AMSGrad(obj, momentum_type='nesterov').minimize()[0], obj.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
