# OptiML
[![Build Status](https://travis-ci.org/dmeoli/optiml.svg?branch=master)](https://travis-ci.org/dmeoli/optiml) [![Coverage Status](https://coveralls.io/repos/github/dmeoli/optiml/badge.svg?branch=master)](https://coveralls.io/github/dmeoli/optiml?branch=master) [![Python Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dmeoli/optiml/master)

OptiML is a sklearn compatible custom reimplementation of *Support Vector Machines* and *Deep Neural Networks*, 
both with some of the most successful features according to the state of art.

This work was motivated by the possibility of being able to solve the optimization problem deriving from the mathematical 
formalization of these models through a wide range of optimization algorithms object of study and developed for the 
Computational Mathematics course  @ [Department of Computer Science](https://www.di.unipi.it/en/) @ 
[University of Pisa](https://www.unipi.it/index.php/english).

## Contents

- Numerical Optimization
    - Unconstrained Optimization
        - Line Search Methods
            - Exact Line Search Methods
                - [x] Quadratic Steepest Gradient Descent
                - [x] Quadratic Conjugate Gradient
            - Inexact Line Search Methods
                - 0th Order Methods
                    - [x] Subgradient
                - 1st Order Methods
                    - [x] Steepest Gradient Descent
                    - [ ] Conjugate Gradient
                        - [ ] Fletcher–Reeves formula
                        - [ ] Polak–Ribière formula
                        - [ ] Hestenes-Stiefel formula
                        - [ ] Dai-Yuan formula
                    - [x] Nonlinear Conjugate Gradient
                        - [x] Fletcher–Reeves formula
                        - [x] Polak–Ribière formula
                        - [x] Hestenes-Stiefel formula
                        - [x] Dai-Yuan formula
                    - [x] Heavy Ball Gradient
                - 2nd Order Methods
                    - [x] Newton
                    - Quasi-Newton
                        - [x] BFGS
                        - [ ] L-BFGS
        - Stochastic Methods
            - [x] Stochastic Gradient Descent
                - [x] standard momentum
                - [x] Nesterov momentum
                - [x] step size and momentum schedules
            - [x] Adam
                - [x] standard momentum
                - [x] Nesterov momentum (Nadam)
            - [x] AMSGrad
                - [x] standard momentum
                - [x] Nesterov momentum
            - [x] AdaMax
                - [x] standard momentum
                - [x] Nesterov momentum (NadaMax)
            - [x] AdaGrad
                - [x] standard momentum
                - [x] Nesterov momentum
            - [x] AdaDelta
                - [x] standard momentum
                - [x] Nesterov momentum
            - [x] RProp
                - [x] standard momentum
                - [x] Nesterov momentum
            - [x] RMSProp
                - [x] standard momentum
                - [x] Nesterov momentum
    - Constrained Quadratic Optimization Methods
        - Box-Constrained Quadratic Methods
            - [x] Projected Gradient
            - [x] Frank-Wolfe or Conditional Gradient
            - [x] Active Set
            - [x] Interior Point
            - [x] Lagrangian Dual Relaxation
        - [x] Sequential Minimal Optimization
        - [x] QP solver with [qpsolvers](https://github.com/stephane-caron/qpsolvers) interface to 
        [cvxopt](https://github.com/cvxopt/cvxopt), [quadprog](https://github.com/rmcgibbo/quadprog), [qpOASES](https://github.com/coin-or/qpOASES), [etc](https://github.com/stephane-caron/qpsolvers#solvers).

- Machine Learning
    - [x] Support Vector Machines
        - Formulations
            - [x] Primal
                - Kernels
                    - [x] linear kernel
            - [x] Dual
                - Kernels
                    - [x] linear kernel
                    - [x] polynomial kernel
                    - [x] rbf kernel
                    - [x] laplacian kernel
                    - [x] sigmoid kernel
        - [x] Support Vector Classifier
            - Losses
                - [x] Hinge (L1 Loss)
                - [x] Squared Hinge (L2 Loss)                            
            - Regularizers
                - [x] L1
                - [x] L2
        - [x] Support Vector Regression
            - Losses
                - [x] Epsilon-Insensitive (L1 Loss)
                - [x] Squared Epsilon-Insensitive (L2 Loss)
            - Regularizers
                - [x] L2
    - [x] Neural Networks
        - [x] Neural Network Classifier
        - [x] Neural Network Regressor
        - Losses
            - [x] Mean Squared Error
            - [x] Binary Cross Entropy
            - [x] Categorical Cross Entropy
            - [x] Sparse Categorical Cross Entropy
        - Regularizers
            - [x] L1 or Lasso
            - [x] L2 or Ridge or Tikhonov
        - Activations
            - [x] Sigmoid
            - [x] Tanh
            - [x] ReLU
            - [x] SoftMax
        - Layers
            - [x] Fully Connected
            - [x] Convolutional
                - [x] Conv 2D
                - [x] Max Pooling
                - [x] Avg Pooling
                - [x] Flatten
        - Initializers
            - [x] Xavier or Glorot normal and uniform
            - [x] He normal and uniform

## Installation

```
git clone https://github.com/dmeoli/optiml
cd optiml
pip install .
```

## Testing

After installation, you can launch the test suite from outside the source directory:

```
pytest optiml
```

## License [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This software is released under the MIT License. See the [LICENSE](LICENSE) file for details.
