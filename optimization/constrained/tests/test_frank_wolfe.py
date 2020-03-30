import numpy as np
import pytest

from optimization.constrained.frank_wolfe import FrankWolfe
from optimization.optimization_function import BoxConstrainedQuadratic


def test():
    np.random.seed(2)
    assert np.allclose(FrankWolfe(BoxConstrainedQuadratic(n=2)).minimize()[0], 0.)


if __name__ == "__main__":
    pytest.main()
