# Machine Learning & Numerical Optimization [![Build Status](https://travis-ci.org/dmeoli/MachineLearningNumericalOptimization.svg?branch=master)](https://travis-ci.org/dmeoli/MachineLearningNumericalOptimization)

This code is an implementation of machine learning models and numerical optimization algorithms developed during the 
Machine Learning & Numerical Methods and Optimization courses @ [Department of Computer Science](https://www.di.unipi.it/en/) 
@ [University of Pisa](https://www.unipi.it/index.php/english).

The main focus of this project is to provide a simple, intuitive and modular implementation of some of the most 
important optimization algorithms used as core solvers for many machine learning models.

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
            - [x] Conjugate Gradient
            - [x] NonLinear Conjugate Gradient
            - [x] Heavy Ball Gradient
            - [x] Accelerated Gradient
            - [x] Newton
            - [x] BFGS Quasi-Newton
            - [ ] L-BFGS
        - Very Inexact Line Search or Fixed Step Size Methods
            - [x] Gradient Descent
            - [x] Adam
            - [ ] AdaMax
            - [ ] AdaGrad
            - [ ] AdaDelta
            - [ ] Rprop
            - [ ] RMSprop
        - Trust Region Methods
            
            
- Machine Learning Models
    - Classification
        - [x] Logistic Regression
        - [x] Support Vector Machine
    - Regression
        - [x] Linear Regression
        - [ ] Support Vector Regression
    - [x] Neural Networks

## License [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This software is released under the MIT License. See the [LICENSE](LICENSE) file for details.
