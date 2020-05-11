import numpy as np
import pytest

from yase.optimization.optimizer import quad2, quad1, Rosenbrock
from yase.optimization.unconstrained.line_search import QuadraticConjugateGradient, NonlinearConjugateGradient


def test_QuadraticConjugateGradient():
    assert np.allclose(QuadraticConjugateGradient(f=quad1, x=np.random.uniform(size=2)).minimize().x, quad1.x_star())
    assert np.allclose(QuadraticConjugateGradient(f=quad2, x=np.random.uniform(size=2)).minimize().x, quad2.x_star())


def test_NonlinearConjugateGradient_quadratic_FletcherReeves():
    assert np.allclose(NonlinearConjugateGradient(f=quad1, x=np.random.uniform(size=2)).minimize().x, quad1.x_star())
    assert np.allclose(NonlinearConjugateGradient(f=quad2, x=np.random.uniform(size=2)).minimize().x, quad2.x_star())


def test_NonlinearConjugateGradient_Rosenbrock_FletcherReeves():
    rosen = Rosenbrock()
    assert np.allclose(NonlinearConjugateGradient(f=rosen, x=np.random.uniform(size=2), wf=0).minimize().x,
                       rosen.x_star(), rtol=0.1)


def test_NonlinearConjugateGradient_quadratic_PolakRibiere():
    assert np.allclose(NonlinearConjugateGradient(f=quad1, x=np.random.uniform(size=2), wf=1).minimize().x,
                       quad1.x_star())
    assert np.allclose(NonlinearConjugateGradient(f=quad2, x=np.random.uniform(size=2), wf=1).minimize().x,
                       quad2.x_star())


def test_NonlinearConjugateGradient_Rosenbrock_PolakRibiere():
    rosen = Rosenbrock()
    assert np.allclose(NonlinearConjugateGradient(f=rosen, x=np.random.uniform(size=2), wf=1).minimize().x,
                       rosen.x_star(), rtol=0.1)


def test_NonlinearConjugateGradient_quadratic_HestenesStiefel():
    assert np.allclose(NonlinearConjugateGradient(f=quad1, x=np.random.uniform(size=2), wf=2).minimize().x,
                       quad1.x_star())
    assert np.allclose(NonlinearConjugateGradient(f=quad2, x=np.random.uniform(size=2), wf=2).minimize().x,
                       quad2.x_star())


def test_NonlinearConjugateGradient_Rosenbrock_HestenesStiefel():
    rosen = Rosenbrock()
    assert np.allclose(NonlinearConjugateGradient(f=rosen, x=np.random.uniform(size=2), wf=2, r_start=1).minimize().x,
                       rosen.x_star(), rtol=0.1)


def test_NonlinearConjugateGradient_quadratic_DaiYuan():
    assert np.allclose(NonlinearConjugateGradient(f=quad1, x=np.random.uniform(size=2), wf=3).minimize().x,
                       quad1.x_star())
    assert np.allclose(NonlinearConjugateGradient(f=quad2, x=np.random.uniform(size=2), wf=3).minimize().x,
                       quad2.x_star())


def test_NonlinearConjugateGradient_Rosenbrock_DaiYuan():
    rosen = Rosenbrock()
    assert np.allclose(NonlinearConjugateGradient(f=rosen, x=np.random.uniform(size=2), wf=3).minimize().x,
                       rosen.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
