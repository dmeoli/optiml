import numpy as np
from scipy.optimize import minimize


def solve_qp(P, q, G, h, A, b, solver='quadprog'):
    return solve_qp(P, q, G, h, A, b, solver)


def scipy_solve_qp(f, G, h, A, b, max_iter, verbose):
    return minimize(fun=f.function, jac=f.jacobian,
                    method='slsqp', x0=np.zeros(f.n),
                    constraints=({'type': 'ineq',
                                  'fun': lambda x: h - np.dot(G, x),
                                  'jac': lambda x: -G},
                                 {'type': 'eq',
                                  'fun': lambda x: np.dot(A, x) - b,
                                  'jac': lambda x: A}),
                    options={'maxiter': max_iter,
                             'disp': verbose}).x


def scipy_solve_svm(f, A, ub, max_iter, verbose):
    return minimize(fun=f.function, jac=f.jacobian,
                    method='slsqp', x0=np.zeros(f.n),
                    constraints={'type': 'eq',
                                 'fun': lambda x: np.dot(A, x),
                                 'jac': lambda x: A},
                    bounds=[(0, u) for u in ub],
                    options={'maxiter': max_iter,
                             'disp': verbose}).x
