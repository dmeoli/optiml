# Yet Another Sklearn Extension
[![Build Status](https://travis-ci.org/dmeoli/yase.svg?branch=master)](https://travis-ci.org/dmeoli/yase) [![Coverage Status](https://coveralls.io/repos/github/dmeoli/yase/badge.svg?branch=master)](https://coveralls.io/github/dmeoli/yase?branch=master) [![Python Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)

This code is a simple and modular implementation of some of the most important optimization algorithms used as core 
solver for many machine learning models developed during the Machine Learning & Numerical Methods and Optimization 
courses @ [Department of Computer Science](https://www.di.unipi.it/en/) @ [University of Pisa](https://www.unipi.it/index.php/english).

## Contents

- Numerical Optimization
    - Unconstrained Optimization
        - Line Search Methods [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmeoli/yase/blob/master/optimization/LineSearchMethods.ipynb)
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
        - Stochastic Methods [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmeoli/yase/blob/master/optimization/StochasticMethods.ipynb)
            - [x] Stochastic Gradient Descent
                - [x] standard momentum
                - [x] Nesterov momentum
                - [ ] step rate schedules
                - [ ] momentum schedules
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
        - [x] Proximal Bundle with [cvxpy](https://github.com/cvxgrp/cvxpy) interface
             - [x] standard momentum
             - [x] Nesterov momentum
        - [x] [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html) interface
    - Constrained Optimization
        - Quadratic Optimization
            - Box-Constrained Methods [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmeoli/yase/blob/master/optimization/BoxConstrainedMethods.ipynb)
                - [x] Projected Gradient
                - [x] Frank-Wolfe or Conditional Gradient
                - [x] Active Set
                - [x] Interior Point
                - [x] Lagrangian Dual
                - [x] Sequential Minimal Optimization (ad hoc for SVMs)
                - [x] BCQP solver with [scipy.optimize.slsqp](https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#sequential-least-squares-programming-slsqp-algorithm-method-slsqp) interface
            - [x] QP solver with [qpsolvers](https://github.com/stephane-caron/qpsolvers) interface
            - [x] QP solver with [scipy.optimize.slsqp](https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#sequential-least-squares-programming-slsqp-algorithm-method-slsqp) interface

    - Optimization Functions
        - Unconstrained
            - [x] Rosenbrock
            - [x] Quadratic
                - [x] Lagrangian Box-Constrained
        - Constrained
            - [x] Quadratic Box-Constrained

- Machine Learning Models
    - [x] Support Vector Machines [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmeoli/yase/blob/master/ml/SupportVectorMachines.ipynb)
        - [x] Support Vector Classifier
        - [x] Support Vector Regression
        - Kernels
            - [x] linear kernel
            - [x] polynomial kernel
            - [x] rbf kernel
            - [x] laplacian kernel
            - [x] sigmoid kernel
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
            - [x] Convolutional [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmeoli/yase/blob/master/ml/ConvolutionalNeuralNetworks.ipynb)
                - [x] Conv 2D
                - [x] Max Pooling
                - [x] Avg Pooling
                - [x] Flatten
            - [ ] Recurrent
                - [ ] Long Short-Term Memory (LSTM)
                - [ ] Gated Recurrent Units (GRU)
            - Normalization
                - [x] Dropout
                - [ ] Batch Normalization
        - Initializers
            - [x] Xavier or Glorot normal and uniform
            - [x] He normal and uniform

# Install

```
git clone https://github.com/dmeoli/yase
cd yase
pip install .
```

## License [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This software is released under the MIT License. See the [LICENSE](LICENSE) file for details.
