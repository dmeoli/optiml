import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.preprocessing import OneHotEncoder


# linear algebra utils

def cholesky_solve(A, b):
    """Solve a symmetric positive definite linear
    system A x = b using Cholesky factorization"""
    L = np.linalg.cholesky(A)  # complexity O(n^3/3)
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


def ldl_solve(ldl_factor, b):
    """Solve a symmetric indefinite linear system
    A x = b using the LDL^T Cholesky factorization."""
    L, D, P = ldl_factor  # complexity O(n^3/3)
    return np.linalg.solve(L.T, (np.linalg.solve(D, np.linalg.solve(L, b[P]))))


# metrics

def mean_euclidean_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return np.mean(np.linalg.norm(y_pred - y_true, axis=y_true.ndim - 1))  # for multi-output compatibility


# util functions

def clip(x, l, h):
    return max(l, min(x, h))


# dataset loaders

path = os.path.dirname(os.path.abspath(__file__))


def load_monk(n_dataset):
    if n_dataset not in (1, 2, 3):
        raise ValueError(f'unknown monk dataset {n_dataset}')
    monks = [np.delete(np.genfromtxt(path + '/ml/data/monks/monks-' + str(n_dataset) + '.' + type), obj=-1, axis=1)
             for type in ('train', 'test')]
    return (OneHotEncoder().fit_transform(monks[0][:, 1:]).toarray(),  # X_train
            OneHotEncoder().fit_transform(monks[1][:, 1:]).toarray(),  # X_test
            monks[0][:, 0].ravel(), monks[1][:, 0].ravel())  # y_train, y_test


def load_ml_cup():
    ml_cup = np.delete(np.genfromtxt(path + '/ml/data/ML-CUP19/ML-CUP19-TR.csv', delimiter=','), obj=0, axis=1)
    return ml_cup[:, :-2], ml_cup[:, -2:]


def load_ml_cup_blind():
    return np.delete(np.genfromtxt(path + '/ml/data/ML-CUP19/ML-CUP19-TS.csv', delimiter=','), obj=0, axis=1)


def load_mnist():
    mnist = np.load(path + '/ml/data/mnist.npz')
    return mnist['x_train'][:, :, :, None], mnist['x_test'][:, :, :, None], mnist['y_train'], mnist['y_test']


# data generators

def generate_linearly_separable_data():
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = -np.ones(len(X2))
    return np.vstack((X1, X2)), np.hstack((y1, y2))


def generate_linearly_separable_overlap_data():
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[1.5, 1.0], [1.0, 1.5]])
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = -np.ones(len(X2))
    return np.vstack((X1, X2)), np.hstack((y1, y2))


def generate_non_linearly_separable_data():
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0, 0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, 50)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 50)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
    y2 = -np.ones(len(X2))
    return np.vstack((X1, X2)), np.hstack((y1, y2))


def generate_non_linearly_regression_data():
    X = np.sort(4 * np.pi * np.random.rand(100)) - 2 * np.pi
    y = np.sinc(X)
    y += 0.25 * (0.5 - np.random.rand(100))  # noise
    return X.reshape(-1, 1), y


# plot functions

def plot_validation_curve(estimator, X, y, param_name, param_range, scorer, cv=5):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range,
                                                 cv=cv, scoring=scorer, n_jobs=-1)

    mean_train_score = np.mean(train_scores, axis=1)
    std_train_score = np.std(train_scores, axis=1)
    mean_test_score = np.mean(test_scores, axis=1)
    std_test_score = np.std(test_scores, axis=1)

    plt.title('validation curve')
    plt.xlabel(param_name)
    plt.ylabel('score')

    plt.plot(param_range, mean_train_score, label='training score', color='navy', marker='.', lw=2)
    plt.fill_between(param_range, mean_train_score - std_train_score,
                     mean_train_score + std_train_score, alpha=0.2, color='navy')
    plt.plot(param_range, mean_test_score, label='cross-validation score', color='darkorange', marker='.', lw=2)
    plt.fill_between(param_range, mean_test_score - std_test_score,
                     mean_test_score + std_test_score, alpha=0.2, color='darkorange')

    plt.legend().get_frame().set_facecolor('white')
    plt.show()


def plot_learning_curve(estimator, X, y, scorer, cv=5, train_sizes=np.linspace(.1, 1.0, 5)):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, train_sizes=train_sizes,
                                                            cv=cv, scoring=scorer, n_jobs=-1)

    mean_train_score = np.mean(train_scores, axis=1)
    std_train_score = np.std(train_scores, axis=1)
    mean_test_score = np.mean(test_scores, axis=1)
    std_test_score = np.std(test_scores, axis=1)

    plt.title('learning curve')
    plt.xlabel('training set size')
    plt.ylabel('score')

    plt.plot(train_sizes, mean_train_score, label='train score', color='navy', marker='.', lw=2)
    plt.fill_between(train_sizes, mean_train_score + std_train_score,
                     mean_train_score - std_train_score, color='navy', alpha=0.2)
    plt.plot(train_sizes, mean_test_score, label='cross-validation score', color='darkorange', marker='.', lw=2)
    plt.fill_between(train_sizes, mean_test_score + std_test_score,
                     mean_test_score - std_test_score, color='darkorange', alpha=0.2)

    plt.legend().get_frame().set_facecolor('white')
    plt.show()


def plot_model_loss(loss_history):
    fig, loss = plt.subplots()
    loss.plot(loss_history['training_loss'], color='navy', lw=2)
    loss.plot(loss_history['validation_loss'], color='darkorange', lw=2)
    loss.set_title('model loss')
    loss.set_xlabel('epoch')
    loss.set_ylabel('loss')
    loss.legend(['training', 'test']).get_frame().set_facecolor('white')
    plt.show()


def plot_model_accuracy(accuracy_history):
    fig, accuracy = plt.subplots()
    accuracy.plot(accuracy_history['training_accuracy'], color='navy', lw=2)
    accuracy.plot(accuracy_history['validation_accuracy'], color='darkorange', lw=2)
    accuracy.set_title('model accuracy')
    accuracy.set_xlabel('epoch')
    accuracy.set_ylabel('accuracy')
    accuracy.legend(['training', 'test']).get_frame().set_facecolor('white')
    plt.show()
