import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model._base import LinearClassifierMixin, LinearModel
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.svm import LinearSVC as SKLinearSVC
from sklearn.svm import LinearSVR as SKLinearSVR
from sklearn.svm import SVC as SKLSVC
from sklearn.svm import SVR as SKLSVR
from sklearn.utils.multiclass import unique_labels

from .svm import SVM, SVC, SVR
from .svm.kernels import Kernel


def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / window_size
    return np.convolve(interval, window, 'same')


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


def generate_nonlinearly_separable_data(size=100, random_state=None):
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


def generate_nonlinearly_regression_data(size=100, random_state=None):
    rs = np.random.RandomState(random_state)
    X = np.sort(2 * np.pi * rs.uniform(size=size))
    y = np.sin(X)
    y += 0.25 * (0.5 - rs.uniform(size=size))  # noise
    return X.reshape(-1, 1), y


def generate_centred_and_normalized_regression_data(size=100, random_state=None):
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
    ax = plt.axes(facecolor='#E6E6E6')  # gray background
    plt.grid(color='w', linestyle='solid')  # draw solid white grid lines
    ax.set_axisbelow(True)
    # hide top and right ticks
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    # hide axis spines
    for spine in ax.spines.values():
        spine.set_visible(False)

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

    kernel = ('' if (isinstance(svm, SVM) and not svm.dual or
                     isinstance(svm, SKLinearSVC) or isinstance(svm, SKLinearSVR)) else
              'using ' + (svm.kernel + ' kernel' if isinstance(svm.kernel, str) else
                          svm.kernel.__class__.__name__ if isinstance(svm.kernel, Kernel) else svm.kernel.__name__))
    plt.title(f'{"" if isinstance(svm, SVM) else "sklearn"} {svm.__class__.__name__} {kernel}', fontsize=9)

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

    # plot training data
    if isinstance(svm, ClassifierMixin):
        plt.plot(X1[:, 0], X1[:, 1], marker='x', markersize=5, color='lightblue', linestyle='none')
        plt.plot(X2[:, 0], X2[:, 1], marker='o', markersize=4, color='darkorange', linestyle='none')
    else:
        plt.plot(X, y, marker='o', markersize=4, color='darkorange', linestyle='none')

    # plot support vectors
    if isinstance(svm, ClassifierMixin):
        if isinstance(svm, SVC) and svm.dual or isinstance(svm, SKLSVC):
            plt.scatter(X[svm.support_][:, 0], X[svm.support_][:, 1], s=60, color='navy')
        elif isinstance(svm, SVC) and not svm.dual or isinstance(svm, SKLinearSVC):
            support_ = np.argwhere(np.abs(svm.decision_function(X)) <= 1).ravel()
            plt.scatter(X[support_][:, 0], X[support_][:, 1], s=60, color='navy')
    elif isinstance(svm, RegressorMixin):
        if isinstance(svm, SVR) and svm.dual or isinstance(svm, SKLSVR):
            plt.scatter(X[svm.support_], y[svm.support_], s=60, color='navy')
        elif isinstance(svm, SVR) and not svm.dual or isinstance(svm, SKLinearSVR):
            support_ = np.argwhere(np.abs(y - svm.predict(X)) >= svm.epsilon).ravel()
            plt.scatter(X[support_], y[support_], s=60, color='navy')

    # plot boundaries
    if isinstance(svm, ClassifierMixin):
        _X1, _X2 = np.meshgrid(np.linspace(X1.min(), X1.max(), 50), np.linspace(X1.min(), X1.max(), 50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(_X1), np.ravel(_X2))])
        Z = svm.decision_function(X).reshape(_X1.shape)
        plt.contour(_X1, _X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        plt.contour(_X1, _X2, Z + 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
        plt.contour(_X1, _X2, Z - 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
    elif isinstance(svm, RegressorMixin):
        _X = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
        Z = svm.predict(_X)
        ax.plot(_X, Z, color='k', linewidth=1)
        ax.plot(_X, Z + svm.epsilon, color='grey', linestyle='--', linewidth=1)
        ax.plot(_X, Z - svm.epsilon, color='grey', linestyle='--', linewidth=1)


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


def plot_learning_curve(estimator, X, y, scorer, cv=5, train_sizes=np.linspace(.1, 1.0, 5),
                        shuffle=False, random_state=None):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, train_sizes=train_sizes, cv=cv,
                                                            scoring=scorer, n_jobs=-1, shuffle=shuffle,
                                                            random_state=random_state)

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


def plot_model_loss(train_loss_history, val_loss_history=None):
    if val_loss_history is None:
        val_loss_history = []

    fig, loss = plt.subplots()
    loss.plot(train_loss_history, color='navy', lw=2)
    loss.plot(val_loss_history, color='darkorange', lw=2)
    loss.set_title('model loss')
    loss.set_xlabel('epoch')
    loss.set_ylabel('loss')
    loss.legend(['training', 'validation']).get_frame().set_facecolor('white')


def plot_model_accuracy(train_score_history, val_score_history=None):
    if val_score_history is None:
        val_score_history = []

    fig, accuracy = plt.subplots()
    accuracy.plot(train_score_history, color='navy', lw=2)
    accuracy.plot(val_score_history, color='darkorange', lw=2)
    accuracy.set_title('model accuracy')
    accuracy.set_xlabel('epoch')
    accuracy.set_ylabel('accuracy')
    accuracy.legend(['training', 'validation']).get_frame().set_facecolor('white')
