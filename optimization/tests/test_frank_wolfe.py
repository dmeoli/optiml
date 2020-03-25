import numpy as np
import pytest

from optimization.constrained.frank_wolfe import FrankWolfe
from optimization.optimization_function import BoxConstrainedQuadratic


def test():
    np.random.seed(0)
    assert np.isclose(FrankWolfe(BoxConstrainedQuadratic()).minimize()[1], -2069.61640, rtol=1e-4)


if __name__ == "__main__":
    pytest.main()
