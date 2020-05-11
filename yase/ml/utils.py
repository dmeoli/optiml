import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC as SKLSVC
from sklearn.svm import SVR as SKLSVR
from sklearn.utils.multiclass import unique_labels

from .svm import SVM, SVC, SVR


# metrics

def mean_euclidean_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return np.mean(np.linalg.norm(y_pred - y_true, axis=y_true.ndim - 1))  # for multi-output compatibility


# dataset loaders

path = os.path.dirname(os.path.abspath(__file__))


def load_monk(n_dataset):
    if n_dataset not in (1, 2, 3):
        raise ValueError(f'unknown monk dataset {n_dataset}')
    monks = [np.delete(np.genfromtxt(path + '/data/monks/monks-' + str(n_dataset) + '.' + type), obj=-1, axis=1)
             for type in ('train', 'test')]
    return (OneHotEncoder().fit_transform(monks[0][:, 1:]).toarray(),  # X_train
            OneHotEncoder().fit_transform(monks[1][:, 1:]).toarray(),  # X_test
            monks[0][:, 0].ravel(), monks[1][:, 0].ravel())  # y_train, y_test


def load_ml_cup():
    ml_cup = np.delete(np.genfromtxt(path + '/data/ML-CUP19/ML-CUP19-TR.csv', delimiter=','), obj=0, axis=1)
    return ml_cup[:, :-2], ml_cup[:, -2:]


def load_ml_cup_blind():
    return np.delete(np.genfromtxt(path + '/data/ML-CUP19/ML-CUP19-TS.csv', delimiter=','), obj=0, axis=1)


# data generators

def generate_linearly_separable_data(size=100, random_state=None):
    rs = np.random.RandomState(random_state)
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    X1 = rs.multivariate_normal(mean1, cov, size)
    y1 = np.ones(len(X1))
    X2 = rs.multivariate_normal(mean2, cov, size)
    y2 = -np.ones(len(X2))
    return np.vstack((X1, X2)), np.hstack((y1, y2))


def generate_linearly_separable_overlap_data(size=100, random_state=None):
    rs = np.random.RandomState(random_state)
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[1.5, 1.0], [1.0, 1.5]])
    X1 = rs.multivariate_normal(mean1, cov, size)
    y1 = np.ones(len(X1))
    X2 = rs.multivariate_normal(mean2, cov, size)
    y2 = -np.ones(len(X2))
    return np.vstack((X1, X2)), np.hstack((y1, y2))


def generate_non_linearly_separable_data(size=50, random_state=None):
    rs = np.random.RandomState(random_state)
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0, 0.8], [0.8, 1.0]]
    X1 = rs.multivariate_normal(mean1, cov, size)
    X1 = np.vstack((X1, rs.multivariate_normal(mean3, cov, size)))
    y1 = np.ones(len(X1))
    X2 = rs.multivariate_normal(mean2, cov, size)
    X2 = np.vstack((X2, rs.multivariate_normal(mean4, cov, size)))
    y2 = -np.ones(len(X2))
    return np.vstack((X1, X2)), np.hstack((y1, y2))


def generate_non_linearly_regression_data(size=100, random_state=None):
    rs = np.random.RandomState(random_state)
    X = np.sort(4 * np.pi * rs.uniform(size=size)) - 2 * np.pi
    y = np.sinc(X)
    y += 0.25 * (0.5 - rs.uniform(size=size))  # noise
    return X.reshape(-1, 1), y


def generate_centred_and_normalized_regression_data(size=50, random_state=None):
    rs = np.random.RandomState(random_state)
    # generating sine curve and uniform noise
    X = np.linspace(0, 1, size)
    noise = 1 * rs.uniform(size=size)
    y = np.sin(X * 1.5 * np.pi)
    y += noise
    # centering the y data to avoid fit the intercept
    y -= y.mean()
    # design matrix is 2x, x^2
    X = np.vstack((2 * X, X ** 2)).T
    # normalizing the design matrix to facilitate visualization
    X = X / np.linalg.norm(X, axis=0)
    return X, y


# plot functions


def plot_svm_hyperplane(svm, X, y):
    plt.style.use('ggplot')

    ax = plt.axes()

    # axis labels and limits
    if isinstance(svm, ClassifierMixin):
        labels = unique_labels(y)
        X1, X2 = X[y == labels[0]], X[y == labels[1]]
        plt.xlabel('$x_1$', fontsize=9)
        plt.ylabel('$x_2$', fontsize=9)
        ax.set(xlim=(X1.min(), X1.max()), ylim=(X2.min(), X2.max()))
    elif isinstance(svm, RegressorMixin):
        plt.xlabel('$X$', fontsize=9)
        plt.ylabel('$y$', fontsize=9)

    plt.title(f'{"custom" if isinstance(svm, SVM) else "sklearn"} {type(svm).__name__} using '
              f'{svm.kernel + " kernel" if isinstance(svm.kernel, str) else svm.kernel.__name__.replace("_", " ")}',
              fontsize=9)

    # set the legend
    if isinstance(svm, ClassifierMixin):
        plt.legend([Line2D([0], [0], linestyle='none', marker='x', color='lightblue',
                           markerfacecolor='lightblue', markersize=9),
                    Line2D([0], [0], linestyle='none', marker='o', color='darkorange',
                           markerfacecolor='darkorange', markersize=9),
                    Line2D([0], [0], linestyle='-', marker='.', color='black',
                           markerfacecolor='darkorange', markersize=0),
                    Line2D([0], [0], linestyle='--', marker='.', color='black',
                           markerfacecolor='darkorange', markersize=0),
                    Line2D([0], [0], linestyle='none', marker='.', color='navy',
                           markerfacecolor='navy', markersize=9)],
                   ['negative -1', 'positive +1', 'decision boundary', 'margin', 'support vectors'],
                   fontsize='7', shadow=True).get_frame().set_facecolor('white')
    elif isinstance(svm, RegressorMixin):
        plt.legend([Line2D([0], [0], linestyle='none', marker='o', color='darkorange',
                           markerfacecolor='darkorange', markersize=9),
                    Line2D([0], [0], linestyle='-', marker='.', color='black',
                           markerfacecolor='darkorange', markersize=0),
                    Line2D([0], [0], linestyle='--', marker='.', color='black',
                           markerfacecolor='darkorange', markersize=0),
                    Line2D([0], [0], linestyle='none', marker='.', color='navy',
                           markerfacecolor='navy', markersize=9)],
                   ['training data', 'decision boundary', '$\epsilon$-insensitive tube', 'support vectors'],
                   fontsize='7', shadow=True).get_frame().set_facecolor('white')

    # training data
    if isinstance(svm, ClassifierMixin):
        plt.plot(X1[:, 0], X1[:, 1], marker='x', markersize=5, color='lightblue', linestyle='none')
        plt.plot(X2[:, 0], X2[:, 1], marker='o', markersize=4, color='darkorange', linestyle='none')
    else:
        plt.plot(X, y, marker='o', markersize=4, color='darkorange', linestyle='none')

    # support vectors
    if isinstance(svm, SVC) or isinstance(svm, SKLSVC):
        plt.scatter(X[svm.support_][:, 0], X[svm.support_][:, 1], s=60, color='navy')
    elif isinstance(svm, SVR) or isinstance(svm, SKLSVR):
        plt.scatter(X[svm.support_], y[svm.support_], s=60, color='navy')

    if isinstance(svm, SVC) or isinstance(svm, SKLSVC):
        _X1, _X2 = np.meshgrid(np.linspace(X1.min(), X1.max(), 50), np.linspace(X1.min(), X1.max(), 50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(_X1), np.ravel(_X2))])
        Z = svm.decision_function(X).reshape(_X1.shape)
        plt.contour(_X1, _X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        plt.contour(_X1, _X2, Z + 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
        plt.contour(_X1, _X2, Z - 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
    elif isinstance(svm, SVR) or isinstance(svm, SKLSVR):
        _X = np.linspace(-2 * np.pi, 2 * np.pi, 10000).reshape(-1, 1)
        Z = svm.predict(_X)
        ax.plot(_X, Z, color='k', linewidth=1)
        ax.plot(_X, Z + svm.epsilon, color='grey', linestyle='--', linewidth=1)
        ax.plot(_X, Z - svm.epsilon, color='grey', linestyle='--', linewidth=1)

    plt.show()


def plot_validation_curve(estimator, X, y, param_name, param_range, scorer, cv=5):
    plt.style.use('ggplot')

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
    plt.style.use('ggplot')

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
    plt.style.use('ggplot')

    fig, loss = plt.subplots()
    loss.plot(loss_history['train_loss'], color='navy', lw=2)
    loss.plot(loss_history['val_loss'], color='darkorange', lw=2)
    loss.set_title('model loss')
    loss.set_xlabel('epoch')
    loss.set_ylabel('loss')
    loss.legend(['training', 'validation']).get_frame().set_facecolor('white')
    plt.show()


def plot_model_accuracy(accuracy_history):
    plt.style.use('ggplot')

    fig, accuracy = plt.subplots()
    accuracy.plot(accuracy_history['train_acc'], color='navy', lw=2)
    accuracy.plot(accuracy_history['val_acc'], color='darkorange', lw=2)
    accuracy.set_title('model accuracy')
    accuracy.set_xlabel('epoch')
    accuracy.set_ylabel('accuracy')
    accuracy.legend(['training', 'validation']).get_frame().set_facecolor('white')
    plt.show()
