__all__ = ['QuadraticSteepestGradientDescent', 'QuadraticConjugateGradient',  # exact line search methods
           'Subgradient',  # inexact line search, 0th order methods
           # inexact line search, 1st order methods
           'SteepestGradientDescent', 'ConjugateGradient', 'NonlinearConjugateGradient', 'HeavyBallGradient',
           'GradientDescent',  # fixed step size methods, 1st order methods
           'Newton', 'BFGS', 'LBFGS',  # inexact line search, 2nd order methods
           'Adam', 'AMSGrad', 'AdaMax', 'AdaGrad', 'AdaDelta', 'RProp', 'RMSProp',  # adaptive methods
           'ProximalBundle']

from .gradient_descent import QuadraticSteepestGradientDescent, SteepestGradientDescent, GradientDescent
from .conjugate_gradient import QuadraticConjugateGradient, ConjugateGradient, NonlinearConjugateGradient
from .heavy_ball_gradient import HeavyBallGradient
from .proximal_bundle import ProximalBundle
from .subgradient import Subgradient
from .newton import Newton
from .quasi_newton import BFGS, LBFGS
from .adam import Adam
from .amsgrad import AMSGrad
from .adamax import AdaMax
from .adagrad import AdaGrad
from .adadelta import AdaDelta
from .rprop import RProp
from .rmsprop import RMSProp
