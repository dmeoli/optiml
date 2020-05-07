import numpy as np
import pytest

from yase.optimization.constrained import FrankWolfe, BoxConstrainedQuadratic


def test():
    np.random.seed(2)
    assert np.allclose(FrankWolfe(BoxConstrainedQuadratic(ndim=2)).minimize()[0], 0.)


if __name__ == "__main__":
    pytest.main()
