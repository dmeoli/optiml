import numpy as np
import pytest

from yase.optimization.box_constrained import ProjectedGradient, BoxConstrainedQuadratic


def test():
    np.random.seed(2)
    assert np.allclose(ProjectedGradient(BoxConstrainedQuadratic(ndim=2)).minimize().x, 0.)


if __name__ == "__main__":
    pytest.main()
