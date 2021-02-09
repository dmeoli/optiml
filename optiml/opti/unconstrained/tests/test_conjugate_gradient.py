import numpy as np
import pytest

from optiml.opti import quad1, quad2
from optiml.opti.unconstrained import Rosenbrock
from optiml.opti.unconstrained.line_search import ConjugateGradient


def test_ConjugateGradient_quadratic_FletcherReeves():
    assert np.allclose(ConjugateGradient(f=quad1, x=np.random.uniform(size=2)).minimize().x, quad1.x_star())
    assert np.allclose(ConjugateGradient(f=quad2, x=np.random.uniform(size=2)).minimize().x, quad2.x_star())


def test_ConjugateGradient_Rosenbrock_FletcherReeves():
    rosen = Rosenbrock()
    assert np.allclose(ConjugateGradient(f=rosen, x=np.random.uniform(size=2), wf=0).minimize().x,
                       rosen.x_star(), rtol=0.1)


def test_ConjugateGradient_quadratic_PolakRibiere():
    assert np.allclose(ConjugateGradient(f=quad1, x=np.random.uniform(size=2), wf=1).minimize().x, quad1.x_star())
    assert np.allclose(ConjugateGradient(f=quad2, x=np.random.uniform(size=2), wf=1).minimize().x, quad2.x_star())


def test_ConjugateGradient_Rosenbrock_PolakRibiere():
    rosen = Rosenbrock()
    assert np.allclose(ConjugateGradient(f=rosen, x=np.random.uniform(size=2), wf=1).minimize().x,
                       rosen.x_star(), rtol=0.1)


def test_ConjugateGradient_quadratic_HestenesStiefel():
    assert np.allclose(ConjugateGradient(f=quad1, x=np.random.uniform(size=2), wf=2).minimize().x, quad1.x_star())
    assert np.allclose(ConjugateGradient(f=quad2, x=np.random.uniform(size=2), wf=2).minimize().x, quad2.x_star())


def test_ConjugateGradient_Rosenbrock_HestenesStiefel():
    rosen = Rosenbrock()
    assert np.allclose(ConjugateGradient(f=rosen, x=np.random.uniform(size=2), wf=2, r_start=1).minimize().x,
                       rosen.x_star(), rtol=0.1)


def test_ConjugateGradient_quadratic_DaiYuan():
    assert np.allclose(ConjugateGradient(f=quad1, x=np.random.uniform(size=2), wf=3).minimize().x, quad1.x_star())
    assert np.allclose(ConjugateGradient(f=quad2, x=np.random.uniform(size=2), wf=3).minimize().x, quad2.x_star())


def test_ConjugateGradient_Rosenbrock_DaiYuan():
    rosen = Rosenbrock()
    assert np.allclose(ConjugateGradient(f=rosen, x=np.random.uniform(size=2), wf=3).minimize().x,
                       rosen.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
