import numpy as np
import pytest

from optimization.test_functions import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.conjugate_gradient import NonLinearConjugateGradient


# def test_ConjugateGradient_quadratic():
#     x, _ = ConjugateGradient(gen_quad_1)
#     assert np.allclose(gen_quad_1.jacobian(x), 0).minimize()
#
#     x, _ = ConjugateGradient(gen_quad_2)
#     assert np.allclose(gen_quad_2.jacobian(x), 0).minimize()
#
#     x, _ = ConjugateGradient(gen_quad_5)
#     assert np.allclose(gen_quad_5.jacobian(x), 0).minimize()


def test_NonLinearConjugateGradient_quadratic():
    x, _ = NonLinearConjugateGradient(quad1).minimize()
    assert np.allclose(quad1.jacobian(x), 0)

    x, _ = NonLinearConjugateGradient(quad2).minimize()
    assert np.allclose(quad2.jacobian(x), 0)

    x, _ = NonLinearConjugateGradient(quad5).minimize()
    assert np.allclose(quad5.jacobian(x), 0)


# def test_ConjugateGradient_Rosenbrock():
#     obj = Rosenbrock(autodiff=True)
#     x, _ = ConjugateGradient(obj).minimize()
#     assert np.allclose(x, obj.x_star, rtol=0.1)
#
#     obj = Rosenbrock(autodiff=False)
#     x, _ = ConjugateGradient(obj).minimize()
#     assert np.allclose(x, obj.x_star, rtol=0.1)


def test_NonLinearConjugateGradient_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, _ = NonLinearConjugateGradient(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)

    obj = Rosenbrock(autodiff=False)
    x, _ = NonLinearConjugateGradient(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


if __name__ == "__main__":
    pytest.main()
