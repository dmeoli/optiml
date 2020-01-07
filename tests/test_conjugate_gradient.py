import numpy as np
import pytest

from optimization.test_functions import *
from optimization.unconstrained.conjugate_gradient import NonLinearConjugateGradient, ConjugateGradient


def test_ConjugateGradient_quadratic():
    x, status = ConjugateGradient(gen_quad_1).minimize()
    assert np.allclose(x, gen_quad_1.x_star)
    assert status is 'optimal'

    x, status = ConjugateGradient(gen_quad_2).minimize()
    assert np.allclose(x, gen_quad_2.x_star)
    assert status is 'optimal'

    x, status = ConjugateGradient(gen_quad_5).minimize()
    assert np.allclose(x, gen_quad_5.x_star)
    assert status is 'optimal'


def test_NonLinearConjugateGradient_quadratic():
    x, status = NonLinearConjugateGradient(gen_quad_1).minimize()
    assert np.allclose(x, gen_quad_1.x_star)
    assert status is 'optimal'

    x, status = NonLinearConjugateGradient(gen_quad_2).minimize()
    assert np.allclose(x, gen_quad_2.x_star)
    assert status is 'optimal'

    x, status = NonLinearConjugateGradient(gen_quad_5).minimize()
    assert np.allclose(x, gen_quad_5.x_star)
    assert status is 'optimal'


def test_ConjugateGradient_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, status = ConjugateGradient(obj).minimize()
    assert np.allclose(x, obj.x_star)
    assert status is 'optimal'

    obj = Rosenbrock(autodiff=False)
    x, status = ConjugateGradient(obj).minimize()
    assert np.allclose(x, obj.x_star)
    assert status is 'optimal'


def test_NonLinearConjugateGradient_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, status = NonLinearConjugateGradient(obj).minimize()
    assert np.allclose(x, obj.x_star)
    assert status is 'optimal'

    obj = Rosenbrock(autodiff=False)
    x, status = NonLinearConjugateGradient(obj).minimize()
    assert np.allclose(x, obj.x_star)
    assert status is 'optimal'


if __name__ == "__main__":
    pytest.main()
