import matplotlib.pyplot as plt
import numpy as np


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


# util functions

def clip(x, l, h):
    return max(l, min(x, h))


# plot functions


def plot_trajectory_optimization(f, opt, x_min, x_max, y_min, y_max):
    fig = f.plot(x_min, x_max, y_min, y_max)
    x0_history = opt['x0_history'] if isinstance(opt, dict) else opt.x0_history
    x1_history = opt['x1_history'] if isinstance(opt, dict) else opt.x1_history
    f_x_history = opt['f_x_history'] if isinstance(opt, dict) else opt.f_x_history
    fig.axes[0].plot(x0_history, x1_history, f_x_history, marker='.', color='k')
    angles_x = np.array(x0_history)[1:] - np.array(x0_history)[:-1]
    angles_y = np.array(x1_history)[1:] - np.array(x1_history)[:-1]
    fig.axes[1].quiver(x0_history[:-1], x1_history[:-1], angles_x, angles_y,
                       scale_units='xy', angles='xy', scale=1)
    plt.show()
