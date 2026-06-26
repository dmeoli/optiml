import pytest

from optiml.opti import quad1
from optiml.opti.unconstrained.line_search import (SteepestGradientDescent, ConjugateGradient,
                                                   Newton, BFGS, LBFGS)
from optiml.opti.unconstrained.stochastic import StochasticGradientDescent, Adam


# these tests just exercise (and so cover) the verbose printing branches of the
# optimizers, which are otherwise never executed by the other tests

@pytest.mark.parametrize('optimizer', [SteepestGradientDescent, ConjugateGradient, Newton, BFGS, LBFGS])
def test_line_search_verbose(optimizer, capsys):
    optimizer(f=quad1, verbose=1).minimize()
    assert capsys.readouterr().out  # the iteration log was printed


@pytest.mark.parametrize('optimizer', [StochasticGradientDescent, Adam])
def test_stochastic_verbose(optimizer, capsys):
    optimizer(f=quad1, step_size=0.1, verbose=1).minimize()
    assert capsys.readouterr().out


if __name__ == "__main__":
    pytest.main()
