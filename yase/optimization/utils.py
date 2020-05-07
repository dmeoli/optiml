import matplotlib.pyplot as plt
import numpy as np
import qpsolvers

from scipy.optimize import minimize


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


# external

def solve_qp(f, G, h, A, b, solver='quadprog'):
    return qpsolvers.solve_qp(f.Q, f.q, G, h, A, b, solver=solver)


def scipy_solve_qp(f, G, h, A, b, max_iter, verbose):
    return minimize(fun=f.function, jac=f.jacobian,
                    method='slsqp', x0=np.zeros(f.ndim),
                    constraints=({'type': 'ineq',
                                  'fun': lambda x: h - np.dot(G, x),
                                  'jac': lambda x: -G},
                                 {'type': 'eq',
                                  'fun': lambda x: np.dot(A, x) - b,
                                  'jac': lambda x: A}),
                    options={'maxiter': max_iter,
                             'disp': verbose}).x


def scipy_solve_bcqp(f, A, ub, max_iter, verbose):
    return minimize(fun=f.function, jac=f.jacobian,
                    method='slsqp', x0=np.zeros(f.ndim),
                    constraints={'type': 'eq',
                                 'fun': lambda x: np.dot(A, x),
                                 'jac': lambda x: A},
                    bounds=[(0, u) for u in ub],
                    options={'maxiter': max_iter,
                             'disp': verbose}).x


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
