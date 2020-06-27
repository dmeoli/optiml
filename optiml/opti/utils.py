import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# linear algebra utils

def cholesky_solve(L, b):
    """Solve a symmetric positive definite linear
    system A x = b using Cholesky factorization"""
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


# function generators

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

def plot_surface_contour(f, opt, x_min, x_max, y_min, y_max):
    X, Y = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    Z = np.array([f.function(np.array([x, y]))
                  for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    fig = plt.figure(figsize=(16, 8))

    # 3D surface plot
    ax = fig.add_subplot(1, 2, 1, projection='3d', elev=50, azim=-50)
    ax.plot_surface(X, Y, Z, norm=SymLogNorm(linthresh=abs(Z.min()), base=np.e), cmap='jet', alpha=0.5)
    ax.plot([f.x_star()[0]], [f.x_star()[1]], [f.f_star()], marker='*', color='r', markersize=10)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel(f'${type(f).__name__}$')

    ub = opt.ub if hasattr(opt, 'ub') else opt.f.ub if hasattr(opt.f, 'ub') else None

    if ub is not None:
        # 3D box-constraints plot
        z_min, z_max = Z.min(), Z.max()
        # vertices of the box
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
        ax.add_collection3d(Poly3DCollection(verts, facecolors='black', linewidths=1.,
                                             edgecolors='k', alpha=0.1))

    # 2D contour plot
    ax = fig.add_subplot(1, 2, 2)
    ax.contour(X, Y, Z, 70, cmap='jet', alpha=0.5)
    ax.plot(*f.x_star(), marker='*', color='r', linestyle='None', markersize=10)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    if ub is not None:
        # 2D box-constraints plot
        ax.plot([0, 0, ub[0], ub[0], 0],
                [0, ub[1], ub[1], 0, 0], color='k', linewidth=1.5)
        ax.fill_between([0, ub[0]],
                        [0, 0],
                        [ub[1], ub[1]], color='0.8')

    return fig


def plot_trajectory_optimization(f, opt, x_min, x_max, y_min, y_max):
    fig = plot_surface_contour(f, opt, x_min, x_max, y_min, y_max)
    x0_history = opt['x0_history'] if isinstance(opt, dict) else opt.x0_history
    x1_history = opt['x1_history'] if isinstance(opt, dict) else opt.x1_history
    f_x_history = opt['f_x_history'] if isinstance(opt, dict) else opt.f_x_history
    # 3D trajectory optimization plot
    fig.axes[0].plot(x0_history, x1_history, f_x_history, marker='.', color='k')
    angles_x = np.array(x0_history)[1:] - np.array(x0_history)[:-1]
    angles_y = np.array(x1_history)[1:] - np.array(x1_history)[:-1]
    # 2D trajectory optimization plot
    fig.axes[1].quiver(x0_history[:-1], x1_history[:-1], angles_x, angles_y,
                       scale_units='xy', angles='xy', scale=1)
    plt.show()
