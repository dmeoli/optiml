import numpy as np
import pytest

from yase.optimization.box_constrained import FrankWolfe, BoxConstrainedQuadratic


def test():
    assert np.allclose(FrankWolfe(BoxConstrainedQuadratic()).minimize().x, 0.)


if __name__ == "__main__":
    pytest.main()
