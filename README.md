# OptiML
[![Build Status](https://travis-ci.com/dmeoli/optiml.svg?branch=master)](https://travis-ci.com/dmeoli/optiml) 
[![Coverage Status](https://coveralls.io/repos/github/dmeoli/optiml/badge.svg?branch=master)](https://coveralls.io/github/dmeoli/optiml?branch=master) 
[![Python Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue) 
[![PyPI Version](https://img.shields.io/pypi/v/optiml.svg?color=blue)](https://pypi.org/project/optiml/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/optiml.svg)](https://pypistats.org/packages/optiml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dmeoli/optiml/master)

OptiML is a sklearn compatible implementation of *Support Vector Machines* and *Deep Neural Networks*, 
both with some of the most successful features according to the state of the art.

This work was motivated by the possibility of being able to solve the optimization problem deriving from the mathematical 
formulation of these models through a wide range of optimization algorithms object of study and developed for the 
Numerical Methods and Optimization course  @ [Department of Computer Science](https://www.di.unipi.it/en/) @ 
[University of Pisa](https://www.unipi.it/index.php/english) under the supervision of prof. [Antonio Frangioni](http://pages.di.unipi.it/frangio/).

## Contents

- Numerical Optimization
    - Unconstrained Optimization
        - Line Search Methods
            - 1st Order Methods
                - [x] Steepest Gradient Descent
                - [x] Conjugate Gradient
                    - [x] Fletcher–Reeves formula
                    - [x] Polak–Ribière formula
                    - [x] Hestenes-Stiefel formula
                    - [x] Dai-Yuan formula
            - 2nd Order Methods
                - [x] Newton
                - Quasi-Newton
                    - [x] BFGS
                    - [ ] L-BFGS
        - Stochastic Methods
            - [x] Stochastic Gradient Descent
                - [x] Momentum
                    - [x] Polyak
                    - [x] Nesterov
            - [x] Adam
                - [x] Momentum
                    - [x] Polyak
                    - [x] Nesterov
            - [x] AMSGrad
                - [x] Momentum
                    - [x] Polyak
                    - [x] Nesterov
            - [x] AdaMax
                - [x] Momentum
                    - [x] Polyak
                    - [x] Nesterov
            - [x] AdaGrad
            - [x] AdaDelta
            - [x] RMSProp
                - [x] Momentum
                    - [x] Polyak
                    - [x] Nesterov
            - [x] Schedules
                - Step size
                    - [x] Decaying
                    - [x] Linear Annealing
                    - [x] Repeater
                - Momentum
                    - [x] Sutskever Blend
        - [x] Proximal Bundle with [cvxpy](https://github.com/cvxgrp/cvxpy) interface to 
          [ecos](https://github.com/embotech/ecos), [osqp](https://github.com/oxfordcontrol/osqp), 
          [scs](https://github.com/cvxgrp/scs), [etc](https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver).
    - Constrained Quadratic Optimization
        - Box-Constrained Quadratic Methods
            - [x] Projected Gradient
            - [x] Frank-Wolfe or Conditional Gradient
            - [x] Active Set
            - [x] Interior Point
        - [x] Lagrangian Dual
        - [x] Augmented Lagrangian Dual

- Machine Learning
    - [x] Support Vector Machines
        - Formulations
            - Primal
            - Wolfe Dual
            - Lagrangian Dual
        - [x] Support Vector Classifier
            - Losses
                - [x] Hinge (L1 Loss) ![l1_svc_loss](notebooks/optimization/tex/img/l1_svc_loss.png)
                - [x] Squared Hinge (L2 Loss) ![l2_svc_loss](notebooks/optimization/tex/img/l2_svc_loss.png)
        - [x] Support Vector Regression
            - Losses
                - [x] Epsilon-insensitive (L1 Loss) ![l1_svr_loss](notebooks/optimization/tex/img/l1_svr_loss.png)
                - [x] Squared Epsilon-insensitive (L2 Loss) ![l2_svr_loss](notebooks/optimization/tex/img/l2_svr_loss.png)
        - Kernels
            - [x] Linear
                
                | SVC          | SVR          |
                |    :----:    |    :----:    |
                | ![linear_dual_l1_svc_hyperplane](notebooks/optimization/tex/img/linear_dual_l1_svc_hyperplane.png) | ![linear_dual_l1_svc_hyperplane](notebooks/optimization/tex/img/linear_dual_l1_svr_hyperplane.png) |
                
            - [x] Polynomial
                 
                | SVC          | SVR          |
                |    :----:    |    :----:    |
                | ![poly_dual_l1_svc_hyperplane](notebooks/optimization/tex/img/poly_dual_l1_svc_hyperplane.png) | ![poly_dual_l1_svc_hyperplane](notebooks/optimization/tex/img/poly_dual_l1_svr_hyperplane.png) |              
                
            - [x] Gaussian
                
                | SVC          | SVR          |
                |    :----:    |    :----:    |
                | ![gaussian_dual_l1_svc_hyperplane](notebooks/optimization/tex/img/gaussian_dual_l1_svc_hyperplane.png) | ![gaussian_dual_l1_svc_hyperplane](notebooks/optimization/tex/img/gaussian_dual_l1_svr_hyperplane.png) |
                
            - [x] Laplacian
              
                | SVC          | SVR          |
                |    :----:    |    :----:    |
                | ![laplacian_dual_l1_svc_hyperplane](notebooks/optimization/tex/img/laplacian_dual_l1_svc_hyperplane.png) | ![laplacian_dual_l1_svc_hyperplane](notebooks/optimization/tex/img/laplacian_dual_l1_svr_hyperplane.png) |
            
            - [x] Sigmoid
        - Optimizers (ad hoc)
            - [x] Sequential Minimal Optimization (SMO)
            - [x] QP solver with [qpsolvers](https://github.com/stephane-caron/qpsolvers) interface to 
            [cvxopt](https://github.com/cvxopt/cvxopt), [quadprog](https://github.com/rmcgibbo/quadprog), 
            [qpOASES](https://github.com/coin-or/qpOASES), [etc](https://github.com/stephane-caron/qpsolvers#solvers).
    - [x] Neural Networks
        - [x] Neural Network Classifier
        - [x] Neural Network Regressor
        - Losses
            - [x] Mean Absolute Error (L1 Loss)
            - [x] Mean Squared Error (L2 Loss)
            - [x] Binary Cross Entropy
            - [x] Categorical Cross Entropy
            - [x] Sparse Categorical Cross Entropy
        - Regularizers
            - [x] L1 or Lasso
            - [x] L2 or Ridge or Tikhonov
        - Activations
            - [x] Linear
            - [x] Sigmoid
            - [x] Tanh
            - [x] ReLU
            - [x] SoftMax
        - Layers
            - [x] Fully Connected
        - Initializers
            - [x] Xavier or Glorot (normal and uniform)
            - [x] He (normal and uniform)

## Install

```
pip install optiml
```

## License [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This software is released under the MIT License. See the [LICENSE](LICENSE) file for details.
