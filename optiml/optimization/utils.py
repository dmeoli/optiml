import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# from .constrained import BoxConstrainedQuadratic


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

def plot_surface_contour(f, x_min, x_max, y_min, y_max):
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

    if isinstance(f, BoxConstrainedQuadratic):
        # 3D box-constraints plot
        z_min, z_max = Z.min(), Z.max()
        # vertices of the box
        v = np.array([[f.ub[0], 0, z_min], [0, 0, z_min],
                      [0, f.ub[1], z_min], [f.ub[0], f.ub[1], z_min],
                      [f.ub[0], 0, z_max], [0, 0, z_max],
                      [0, f.ub[1], z_max], [f.ub[0], f.ub[1], z_max]])
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

    if isinstance(f, BoxConstrainedQuadratic):
        # 2D box-constraints plot
        ax.plot([0, 0, f.ub[0], f.ub[0], 0],
                [0, f.ub[1], f.ub[1], 0, 0], color='k', linewidth=1.5)
        ax.fill_between([0, f.ub[0]],
                        [0, 0],
                        [f.ub[1], f.ub[1]], color='0.8')

    return fig


def plot_trajectory_optimization(f, opt, x_min, x_max, y_min, y_max):
    fig = plot_surface_contour(f, x_min, x_max, y_min, y_max)
    x0_history = opt['x0_history'] if isinstance(opt, dict) else opt.x0_history
    x1_history = opt['x1_history'] if isinstance(opt, dict) else opt.x1_history
    f_x_history = opt['f_x_history'] if isinstance(opt, dict) else opt.f_x_history
    fig.axes[0].plot(x0_history, x1_history, f_x_history, marker='.', color='k')
    angles_x = np.array(x0_history)[1:] - np.array(x0_history)[:-1]
    angles_y = np.array(x1_history)[1:] - np.array(x1_history)[:-1]
    fig.axes[1].quiver(x0_history[:-1], x1_history[:-1], angles_x, angles_y,
                       scale_units='xy', angles='xy', scale=1)
    plt.show()
