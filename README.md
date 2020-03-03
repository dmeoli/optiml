# Machine Learning & Numerical Optimization [![Build Status](https://travis-ci.org/dmeoli/MachineLearningNumericalOptimization.svg?branch=master)](https://travis-ci.org/dmeoli/MachineLearningNumericalOptimization)

This code is an implementation of machine learning models and numerical optimization algorithms developed during the 
Machine Learning & Numerical Methods and Optimization courses @ [Department of Computer Science](https://www.di.unipi.it/en/) 
@ [University of Pisa](https://www.unipi.it/index.php/english).

The main focus of this project is to provide a simple, intuitive and modular implementation of some of the most 
important optimization algorithms used as core solver for many machine learning models.

## Contents
- Optimization Algorithms
    - Constrained Optimization
        - [ ] Projected Gradient
        - [ ] Conditional Gradient or Frank-Wolfe
        - [ ] Active Set
        - [ ] Interior Point
        - [ ] Dual
    - Unconstrained Optimization
        - Exact Line Search Methods
            - [x] Quadratic Steepest Gradient Descent
            - [x] Quadratic Conjugate Gradient
        - Inexact Line Search Methods
            - [x] Subgradient
            - [x] Steepest Gradient Descent
            - [ ] Conjugate Gradient
            - [x] Nonlinear Conjugate Gradient
                - [x] Fletcher–Reeves formula
                - [x] Polak–Ribière formula
                - [x] Hestenes-Stiefel formula
                - [x] Dai-Yuan formula
            - [x] Heavy Ball Gradient
            - [x] Steepest Accelerated Gradient
            - [x] Newton
            - [x] BFGS quasi-Newton
            - [ ] L-BFGS quasi-Newton
        - Very Inexact Line Search or Fixed Step Size Methods
            - [x] Gradient Descent
                - [x] standard momentum
                - [x] Nesterov momentum
                - [ ] learning rate decay
                - [ ] momentum decay
            - [x] Accelerated Gradient
                - [ ] standard momentum
                - [ ] Nesterov momentum
                - [ ] learning rate decay
                - [ ] momentum decay
            - [x] Adam
                - [x] standard momentum
                - [x] Nadam (Nesterov momentum)
            - [x] AMSGrad
                - [x] standard momentum
                - [x] Nesterov momentum
            - [x] AdaMax
                - [x] standard momentum
                - [x] Nesterov momentum
            - [x] AdaGrad
                - [x] standard momentum
                - [x] Nesterov momentum
            - [x] AdaDelta
                - [x] standard momentum
                - [x] Nesterov momentum
            - [X] RProp
                - [x] standard momentum
                - [x] Nesterov momentum
            - [x] RMSProp
                - [x] standard momentum
                - [x] Nesterov momentum
                
- Machine Learning Models
    - Classification
        - [x] Logistic Regression
        - [x] Support Vector Machine
            - [x] linear kernel
            - [x] polynomial kernel
            - [x] rbf kernel
    - Regression
        - [x] Linear Regression
        - [ ] Support Vector Regression
            - [x] linear kernel
            - [x] polynomial kernel
            - [x] rbf kernel
    - [x] Neural Networks
        - [x] Losses
            - [x] Mean Squared Error
            - [x] Mean Absolute Error
            - [x] Categorical Cross Entropy
            - [x] Binary Cross Entropy
        - [x] Regularizers
            - [x] L1 or Lasso
            - [x] L2 or Ridge or Tikhonov
        - [x] Activations
            - [x] Sigmoid
            - [x] Tanh
            - [x] ReLU
            - [x] LeakyReLU
            - [x] ELU
            - [x] SoftMax
            - [x] SoftPlus
        - [ ] Layers
            - [x] Fully Connected
            - [x] Convolutional
            - [x] Max Pooling
            - [x] Avg Pooling
            - [x] Flatten
            - [x] Dropout
            - [ ] Batch Normalization
            - [ ] Recurrent
            - [ ] Long Short-Term Memory (LSTM)
            - [ ] Gated Recurrent Units (GRU)
        - [x] Initializers
            - [x] Xavier or Glorot normal and uniform
            - [x] He normal and uniform

## License [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This software is released under the MIT License. See the [LICENSE](LICENSE) file for details.
