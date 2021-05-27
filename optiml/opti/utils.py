import matplotlib.pyplot as plt
import numpy as np
from casadi import ldl_solve, ldl
from matplotlib.colors import SymLogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse.linalg import minres

from .unconstrained import ProximalBundle


def solve_lagrangian_equality_constrained_quadratic(Q, q, A, method='minres'):
    """
    Solve a quadratic function subject to equality constraint:

            1/2 x^T Q x + q^T x : A x = b = 0

    by solving the KKT system:

            | Q A^T | |   x   | = |   -q   |
            | A  0  | | lmbda |   |  b = 0 |
    """
    A = np.atleast_2d(A).astype(float)
    if method in ('minres', 'ldl'):
        kkt_Q = np.vstack((np.hstack((Q, A.T)),
                           np.hstack((A, np.zeros((A.shape[0], A.shape[0]))))))
        kkt_q = np.hstack((-q, np.zeros((A.shape[0],))))
        if method == 'minres':
            x_lmbda = minres(kkt_Q, kkt_q)[0]
        elif method == 'ldl':
            #  https://www.math.uh.edu/~rohop/fall_06/Chapter3.pdf#page=3
            x_lmbda = np.array(ldl_solve(kkt_q, *ldl(kkt_Q))).ravel()
        # assert np.allclose(x_lmbda[:-A.shape[0]], solve_qp(P=Q, q=q, A=A, b=np.zeros(1), solver='cvxopt'))
        return x_lmbda[:-A.shape[0]], x_lmbda[-A.shape[0]:]
    elif method == 'nullspace':
        q, r = np.linalg.qr(A.T, mode='complete')
        Z = q[:, A.shape[0]:]  # orthonormal basis for the null space of A, i.e., kernel(A)
        assert np.allclose(A.dot(Z), 0)
        Y = q[:, :A.shape[0]]
        Rp = r[:A.shape[0], :]
        x_special = np.linalg.lstsq(A, -np.array([0]), rcond=None)[0]
        reduced_Q = Z.T.dot(Q).dot(Z)
        reduced_q = Z.T.dot(q) + Z.T.dot(Q).dot(x_special)
        x_g = np.linalg.lstsq(reduced_Q, -reduced_q, rcond=None)[0]
        x = Z.dot(x_g) + x_special
        lmbda = np.linalg.lstsq(Rp, -Y.T.dot(Q.dot(x) + q), rcond=None)[0]
        return x, lmbda
    elif method == 'schur':
        # http://www.cs.nthu.edu.tw/~cherung/teaching/2009cs5321/link/chap16ex.pdf#page=2
        raise NotImplementedError
    else:
        raise NotImplementedError


# bcqp generator

def generate_box_constrained_quadratic(ndim=2, actv=0.5, rank=1.1, ecc=0.99, ub_min=8, ub_max=12, seed=None):
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

    G = np.random.rand(round(rank * ndim), ndim)
    Q = G.T.dot(G)

    # compute eigenvalue decomposition
    D, V = np.linalg.eigh(Q)  # V.dot(np.diag(D)).dot(V.T) = Q

    if min(D) > 1e-14:  # smallest eigenvalue
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

def plot_surface_contour(f, x_min, x_max, y_min, y_max):
    dual = None
    if hasattr(f, 'primal'):  # lagrangian dual
        dual = f
        f = dual.primal

    X, Y = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    Z = np.array([f(np.array([x, y]))
                  for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    surface_contour = plt.figure(figsize=(16, 8))

    # 3D surface plot
    ax = surface_contour.add_subplot(1, 2, 1, projection='3d', elev=50, azim=-50)
    ax.plot_surface(X, Y, Z, norm=SymLogNorm(linthresh=abs(Z.min()), base=np.e), cmap='jet', alpha=0.5)
    ax.plot(*f.x_star(), f.f_star(), marker='*', color='r', markersize=10)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel(f'${type(f).__name__}$')

    if dual and hasattr(dual, 'A'):
        X, Y = np.meshgrid(np.arange(x_min, x_max, 2), np.arange(y_min, y_max, 2))
        Z = np.array([f(np.array([x, y]))
                      for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
        # y = m x + q => m = -(A[0] / A[1]), q = 0
        surf1 = ax.plot_surface(X, -(dual.A[0] / dual.A[1]) * X, Z, color='b', label='$Ax=0$')
        # bug https://stackoverflow.com/a/55534939/5555994
        surf1._facecolors2d = surf1._facecolor3d
        surf1._edgecolors2d = surf1._edgecolor3d

    if dual and hasattr(dual, 'ub'):  # and so also `lb`

        # 3D box-constraints plot
        z_min, z_max = Z.min(), Z.max()
        # vertices of the box
        ub = dual.ub
        v = np.array([[ub[0], 0, z_min], [0, 0, z_min],
                      [0, ub[1], z_min], [ub[0], ub[1], z_min],
                      [ub[0], 0, z_max], [0, 0, z_max],
                      [0, ub[1], z_max], [ub[0], ub[1], z_max]])
        # generate list of sides' polygons of our box
        verts = [[v[0], v[1], v[2], v[3]],
                 [v[4], v[5], v[6], v[7]],
                 [v[0], v[1], v[5], v[4]],
                 [v[2], v[3], v[7], v[6]],
                 [v[1], v[2], v[6], v[5]],
                 [v[4], v[7], v[3], v[0]]]
        # plot sides
        surf2 = ax.add_collection3d(Poly3DCollection(verts, facecolors='k', edgecolors='k',
                                                     alpha=0.1, label='$0 \leq x \leq ub$'))
        # bug https://stackoverflow.com/a/55534939/5555994
        surf2._facecolors2d = surf2._facecolor3d
        surf2._edgecolors2d = surf2._edgecolor3d

    elif dual and hasattr(dual, 'lb'):

        # 3D box-constraints plot
        z_min, z_max = Z.min(), Z.max()
        # vertices of the box
        v = np.array([[x_max, 0, z_min], [0, 0, z_min],
                      [0, y_max, z_min], [x_max, y_max, z_min],
                      [x_max, 0, z_max], [0, 0, z_max],
                      [0, y_max, z_max], [x_max, y_max, z_max]])
        # generate list of sides' polygons of our box
        verts = [[v[0], v[1], v[2], v[3]],
                 [v[4], v[5], v[6], v[7]],
                 [v[0], v[1], v[5], v[4]],
                 [v[2], v[3], v[7], v[6]],
                 [v[1], v[2], v[6], v[5]],
                 [v[4], v[7], v[3], v[0]]]
        # plot sides
        surf3 = ax.add_collection3d(Poly3DCollection(verts, facecolors='k', edgecolors='k',
                                                     alpha=0.1, label='$x \geq 0$'))
        # bug https://stackoverflow.com/a/55534939/5555994
        surf3._facecolors2d = surf3._facecolor3d
        surf3._edgecolors2d = surf3._edgecolor3d

    ax.legend()

    # 2D contour plot
    ax = surface_contour.add_subplot(1, 2, 2)
    ax.contour(X, Y, Z, 70, cmap='jet', alpha=0.5)
    ax.plot(*f.x_star(), marker='*', color='r', markersize=10)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    if dual and hasattr(dual, 'A'):
        X = np.arange(x_min, x_max, 2)
        # y = m x + q => m = -(A[0] / A[1]), q = 0
        ax.plot(X, -(dual.A[0] / dual.A[1]) * X, color='b')

    if dual and hasattr(dual, 'ub'):  # and so also `lb`

        # 2D box-constraints plot
        ub = dual.ub
        ax.plot([0, 0, ub[0], ub[0], 0],
                [0, ub[1], ub[1], 0, 0], color='k')
        ax.fill_between([0, ub[0]],
                        [0, 0],
                        [ub[1], ub[1]], color='0.8')

    elif dual and hasattr(dual, 'lb'):

        x_max = ax.get_xlim()[1]
        y_max = ax.get_ylim()[1]
        ax.plot([0, 0, x_max, x_max, 0],
                [0, y_max, y_max, 0, 0], color='k')
        ax.fill_between([0, x_max],
                        [0, 0],
                        [y_max, y_max], color='0.8')

    return surface_contour


def plot_trajectory_optimization(surface_contour, opt, color='k', label=None, linewidth=1.5):
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


def plot_surface_trajectory_optimization(f, opt, x_min, x_max, y_min, y_max,
                                         color='k', label=None, linewidth=1.5):
    return plot_trajectory_optimization(plot_surface_contour(f, x_min, x_max, y_min, y_max),
                                        opt, color, label, linewidth)
