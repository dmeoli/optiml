import matplotlib.pyplot as plt
import numpy as np
from casadi import ldl_solve, ldl
from matplotlib.colors import SymLogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from qpsolvers import solve_qp
from scipy.sparse.linalg import gmres

from .unconstrained import ProximalBundle


def cholesky_null_space(A):
    A = np.atleast_2d(A)
    # Z = scipy.null_space(A)  # more complex but more stable since uses SVD
    Q = np.linalg.qr(A.T, mode='complete')[0]
    # null space aka kernel
    Z = Q[:, A.shape[0]:]  # orthonormal basis for the null space of A, i.e., ker(A)
    assert np.allclose(A.dot(Z), 0)
    return Z


def solve_lagrangian_equality_constrained_quadratic(Q, q, A, b, method='gmres'):
    """
    Solve a quadratic function subject to equality constraint:

            1/2 x^T Q x + q^T x : A x = b

    by solving the KKT system:

            | Q A^T | |   x   | = | -q |
            | A  0  | | lmbda |   |  b |

    See more @ https://www.math.uh.edu/~rohop/fall_06/Chapter3.pdf
    """
    A = np.atleast_2d(A)
    kkt_Q = np.vstack((np.hstack((Q, A.T)),
                       np.hstack((A, np.zeros((A.shape[0], A.shape[0]))))))
    kkt_q = np.hstack((-q, b))
    if method == 'gmres':
        x_lmbda = gmres(kkt_Q, kkt_q)[0]
    elif method == 'ldl':
        x_lmbda = np.array(ldl_solve(kkt_q, *ldl(kkt_Q))).ravel()
    # assert np.allclose(x_lmbda[:-A.shape[0]], solve_qp(P=Q, q=q, A=A, b=b, solver='cvxopt'))
    return x_lmbda[:-A.shape[0]], x_lmbda[-A.shape[0]:]


# bcqp generator

def generate_box_constrained_quadratic(ndim=2, actv=0.5, rank=1.1, ecc=0.99, ub_min=8, ub_max=12, seed=123456):
    """
    Generate a box-constrained quadratic function defined as:

                1/2 x^T Q x + q^T x : 0 <= x <= ub

    :param ndim: (integer, scalar): the size of the problem
    :param actv: (real, scalar, default 0.5): how many box constraints (as a
                 fraction of the number of variables n of the problems) the
                 unconstrained optimum will violate, and therefore we expect to be
                 active in the constrained optimum; note that there is no guarantee that
                 exactly acvt constraints will be active, they may be less or (more
                 likely) more, except when actv = 0 because then the unconstrained
                 optimum is surely feasible and therefore it will be the constrained
                 optimum as well
    :param rank: (real, scalar, default 1.1): Q will be obtained as Q = G^T G, with
                 G a m \times n random matrix with m = rank * n. If rank > 1 then Q can
                 be expected to be full-rank, if rank < 1 it will not
    :param ecc: (real, scalar, default 0.99): the eccentricity of Q, i.e., the
                ratio ( \lambda_1 - \lambda_n ) / ( \lambda_1 + \lambda_n ), with
                \lambda_1 the largest eigenvalue and \lambda_n the smallest one. Note
                that this makes sense only if \lambda_n > 0, for otherwise the
                eccentricity is always 1; hence, this setting is ignored if
                \lambda_n = 0, i.e., Q is not full-rank (see above). An eccentricity of
                0 means that all eigenvalues are equal, as eccentricity -> 1 the
                largest eigenvalue gets larger and larger w.r.t. the smallest one
    :param seed: (integer, default 0): the seed for the random number generator
    :param ub_min: (real, scalar, default 8): the minimum value of each ub_i
    :param ub_max: (real, scalar, default 12): the maximum value of each ub_i
    """

    if not ndim >= 2:
        raise ValueError('ndim must be >= 2')
    ndim = round(ndim)
    if ndim <= 0:
        raise ValueError('n must be > 0')
    if not 0 <= actv <= 1:
        raise ValueError('actv has to lie in [0, 1]')
    if not rank > 0:
        raise ValueError('rank must be > 0')
    if not 0 <= ecc < 1:
        raise ValueError('ecc has to lie in [0, 1)')
    if not ub_min > 0:
        raise ValueError('ub_min must be > 0')
    if ub_max <= ub_min:
        raise ValueError('ub_max must be > ub_min')

    np.random.seed(seed)

    ub = ub_min * np.ones(ndim) + (ub_max - ub_min) * np.random.rand(ndim)

    G = np.random.rand(ndim, round(rank * ndim)).T
    Q = G.T.dot(G)

    # compute eigenvalue decomposition
    D, V = np.linalg.eigh(Q)  # V.dot(np.diag(D)).dot(V.T) = Q

    if D[0] > 1e-14:  # smallest eigenvalue
        # modify eccentricity only if \lambda_n > 0, for when \lambda_n = 0 the
        # eccentricity is 1 by default. The formula is:
        #
        #                         \lambda_i - \lambda_n            2 ecc
        # \lambda_i = \lambda_n + --------------------- \lambda_n -------
        #                         \lambda_1 - \lambda_n           1 - ecc
        #
        # This leaves \lambda_n unchanged, and modifies all the other ones
        # proportionally so that:
        #
        #   \lambda_1 - \lambda_n
        #   --------------------- = ecc
        #   \lambda_1 - \lambda_n

        l = D[0] + (D[0] / (D[-1] - D[0])) * (2 * ecc / (1 - ecc)) * (D - D[0])

        Q = V.dot(np.diag(l)).dot(V.T)

    # we first generate the unconstrained minimum z of the problem in the form:
    #
    #          min 1/2 (x - z)^T Q (x - z)
    #    min 1/2 x^T Q x - z^T Q x + 1/2 z^T Q z
    #
    # and then we set q = -z^T Q

    z = np.zeros(ndim)

    # out_b[i] = True if z[i] will be out of the bounds
    out_b = np.random.rand(ndim) <= actv

    # 50/50 chance of being left of lb or right of ub
    lr = np.random.rand(ndim) <= 0.5
    l = np.logical_and(out_b, lr)
    r = np.logical_and(out_b, np.logical_not(lr))

    # a random amount left of the lb[0]
    z[l] = -np.random.rand(sum(l)) * ub[l]

    # a random amount right of the ub[u]
    z[r] = ub[r] * (1 + np.random.rand(sum(r)))

    out_b = np.logical_not(out_b)  # entries that will be inside the bound
    # pick at random in [0, u]
    z[out_b] = np.random.rand(sum(out_b)) * ub[out_b]

    q = -Q.dot(z)

    return Q, q, ub


# plot functions

def plot_surface_contour(f, x_min, x_max, y_min, y_max, ub=None, primal=True):
    dual = None
    # plot primal with constraints
    if hasattr(f, 'primal') and primal:  # is_lagrangian_dual()
        dual = f
        f = dual.primal

    X, Y = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    Z = np.array([f(np.concatenate((np.array([x, y]), f.dual_x)))  # x, mu_lmbda*
                  # is_lagrangian_dual() and not is_augmented_lagrangian_dual()
                  if hasattr(f, 'primal') and not hasattr(f, 'rho') else f(np.array([x, y]))
                  for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    surface_contour = plt.figure(figsize=(16, 8))

    # 3D surface plot
    ax = surface_contour.add_subplot(1, 2, 1, projection='3d', elev=50, azim=-50)
    ax.plot_surface(X, Y, Z, norm=SymLogNorm(linthresh=abs(Z.min()), base=np.e), cmap='jet', alpha=0.5)
    if f.f_star() < np.inf:
        ax.plot(*f.x_star(), f.f_star(), marker='*', color='b', markersize=10,
                linestyle='None', label='global optima')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel(f'${f.__class__.__name__}$')

    constrained = False

    if dual and (dual.A is not None and not np.all((dual.A == 0))):
        _X, _Y = np.meshgrid(np.arange(x_min, x_max, 2), np.arange(y_min, y_max, 2))
        _Z = np.array([f(np.array([x, y]))
                       for x, y in zip(_X.ravel(), _Y.ravel())]).reshape(_X.shape)
        # y = m x => m = -(A[0] / A[1]), i.e., slope
        # TODO include b in the slope
        surf1 = ax.plot_surface(_X, -(dual.A[:, 0] / dual.A[:, 1]) * _X, _Z, color='b', label='$Ax=b$')
        constrained = True
        # bug https://stackoverflow.com/a/55534939/5555994
        surf1._facecolors2d = surf1._facecolor3d
        surf1._edgecolors2d = surf1._edgecolor3d

    if (ub is not None  # bcqp optimizer
            or (dual and dual.ub is not None)
            or (dual and dual.lb is not None)):
        _lb = ([0, 0] if ub is not None else  # bcqp optimizer just with lb = 0
               dual.lb if dual.lb is not None else  # dual with explicit lb
               [X.min(), Y.min()])  # dual without explicit lb, so we take [x_min, y_min]
        _ub = (ub if ub is not None else  # bcqp optimizer with given ub
               dual.ub if dual.ub is not None else  # dual with explicit ub
               [X.max(), Y.max()])  # dual without explicit ub, so we take [x_max, y_max]

        # 3D box-constraints plot
        z_min, z_max = Z.min(), Z.max()
        # vertices of the box
        v = np.array([[_ub[0], _lb[0], z_min], [_lb[0], _lb[1], z_min],
                      [_lb[1], _ub[1], z_min], [_ub[0], _ub[1], z_min],
                      [_ub[0], _lb[0], z_max], [_lb[0], _lb[1], z_max],
                      [_lb[1], _ub[1], z_max], [_ub[0], _ub[1], z_max]])
        # generate list of sides' polygons of our box
        verts = [[v[0], v[1], v[2], v[3]],
                 [v[4], v[5], v[6], v[7]],
                 [v[0], v[1], v[5], v[4]],
                 [v[2], v[3], v[7], v[6]],
                 [v[1], v[2], v[6], v[5]],
                 [v[4], v[7], v[3], v[0]]]
        # plot sides
        surf2 = ax.add_collection3d(Poly3DCollection(verts, facecolors='k', edgecolors='k', alpha=0.1,
                                                     label=('$x \leq ub$' if np.all(_lb == [X.min(), Y.min()]) else
                                                            '$x \geq lb$' if np.all(_ub == [X.max(), Y.max()]) else
                                                            '$lb \leq x \leq ub$')))
        constrained = True
        # bug https://stackoverflow.com/a/55534939/5555994
        surf2._facecolors2d = surf2._facecolor3d
        surf2._edgecolors2d = surf2._edgecolor3d

    if constrained:
        if ub is not None:  # bcqp optimizer
            x_star = solve_qp(P=f.Q,
                              q=f.q,
                              lb=np.zeros_like(f.q),
                              ub=ub)
            ax.plot(*x_star, f(x_star), marker='*', color='r', markersize=10,
                    linestyle='None', label='constrained optima')
        else:
            ax.plot(*dual.x_star(), dual.f_star(), marker='*', color='r', markersize=10,
                    linestyle='None', label='constrained optima')
        ax.legend()

    # 2D contour plot
    ax = surface_contour.add_subplot(1, 2, 2)
    ax.contour(X, Y, Z, 70, cmap='jet', alpha=0.5)
    if f.f_star() < np.inf:
        ax.plot(*f.x_star(), marker='*', color='b', markersize=10, linestyle='None')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    constrained = False

    if dual and (dual.A is not None and not np.all((dual.A == 0))):
        _X = np.arange(x_min, x_max, 2)
        # y = m x => m = -(A[0] / A[1]), i.e., slope
        # TODO include b in the slope
        ax.plot(_X, -(dual.A[:, 0] / dual.A[:, 1]) * _X, color='b')
        constrained = True

    if (ub is not None  # bcqp optimizer
            or (dual and dual.ub is not None)
            or (dual and dual.lb is not None)):
        _lb = ([0, 0] if ub is not None else  # bcqp optimizer just with lb = 0
               dual.lb if dual.lb is not None else  # dual with explicit lb
               [X.min(), Y.min()])  # dual without explicit lb, so we take [x_min, y_min]
        _ub = (ub if ub is not None else  # bcqp optimizer with given ub
               dual.ub if dual.ub is not None else  # dual with explicit ub
               [X.max(), Y.max()])  # dual without explicit ub, so we take [x_max, y_max]

        # 2D box-constraints plot
        ax.fill_between([_lb[0], _ub[0]],
                        [_lb[1], _lb[1]],
                        [_ub[1], _ub[1]], color='0.8', edgecolor='k')
        constrained = True

    if constrained:
        if ub is not None:  # bcqp optimizer
            x_star = solve_qp(P=f.Q,
                              q=f.q,
                              lb=np.zeros_like(f.q),
                              ub=ub)
            ax.plot(*x_star, marker='*', color='r', markersize=10)
        else:
            ax.plot(*dual.x_star(), marker='*', color='r', markersize=10)

    return surface_contour


def plot_trajectory_optimization(surface_contour, opt, color='k', label=None, linewidth=1.5):
    if label is None:
        label = opt.__class__.__name__

    # 3D trajectory optimization plot
    surface_contour.axes[0].plot(opt.x0_history, opt.x1_history, opt.f_x_history,
                                 marker='.', color=color, label=label, linewidth=linewidth)
    angles_x = np.array(opt.x0_history)[1:] - np.array(opt.x0_history)[:-1]
    angles_y = np.array(opt.x1_history)[1:] - np.array(opt.x1_history)[:-1]
    # 2D trajectory optimization plot
    surface_contour.axes[1].quiver(opt.x0_history[:-1], opt.x1_history[:-1], angles_x, angles_y,
                                   scale_units='xy', angles='xy', scale=1, color=color, linewidth=linewidth)
    if isinstance(opt, ProximalBundle):  # plot ns steps
        # 3D trajectory optimization plot
        surface_contour.axes[0].plot(opt.x0_history_ns, opt.x1_history_ns, opt.f_x_history_ns,
                                     marker='.', color='b')
        angles_x = np.array(opt.x0_history_ns)[1:] - np.array(opt.x0_history_ns)[:-1]
        angles_y = np.array(opt.x1_history_ns)[1:] - np.array(opt.x1_history_ns)[:-1]
        # 2D trajectory optimization plot
        surface_contour.axes[1].quiver(opt.x0_history_ns[:-1], opt.x1_history_ns[:-1], angles_x, angles_y,
                                       scale_units='xy', angles='xy', scale=1, color='b')
    surface_contour.axes[0].legend()
    return surface_contour


# utility for Jupyter Notebook
def plot_surface_trajectory_optimization(f, opt, x_min, x_max, y_min, y_max, primal=True,
                                         color='k', label=None, linewidth=1.5):
    ub = opt.ub if hasattr(opt, 'ub') else None
    plot_trajectory_optimization(plot_surface_contour(f, x_min, x_max, y_min, y_max, ub, primal),
                                 opt, color, label, linewidth)
