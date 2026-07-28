import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from cd.algs.cd_poly import CDPolynomial
from cd.utils.plotting import plot_contours

d = 1
N = 4
eig = 2

pf = PolynomialFeatures(degree=d)


def k(x, y, d):
    return (1 + x @ y) ** d


def generate_points_on_circle(ranges, num):
    spans = [end - start for (start, end) in ranges]
    total_span = sum(spans)
    # Number of points per range, proportional to span
    points_per_range = [int(np.round(num * span / total_span)) for span in spans]
    # Adjust to ensure total points sum to num
    diff = num - sum(points_per_range)
    for i in range(abs(diff)):
        points_per_range[i % len(points_per_range)] += np.sign(diff)
    points = []
    for (start, end), n in zip(ranges, points_per_range):
        angles = np.linspace(start, end, n, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        points.append(np.stack([x, y], axis=1))
    return np.vstack(points)


if __name__ == "__main__":

    x = generate_points_on_circle([(-.3, .3), 
                                   (-.3 + np.pi, .3 + np.pi)
                                   #(-.3+np.pi/2,.3+np.pi/2),
                                   #(-.3-np.pi/2,.3-np.pi/2)
                                   ]
                                  , num=N)

    v = pf.fit_transform(x)
    M = v.T @ v / N
    K = v @ v.T
    
    p = np.array([-1, 0, 0, 1, 0, 1])

    lam, v = np.linalg.eig(M)
    idx = np.argsort(lam)
    lam = lam[idx]
    v = v[:, idx]
    print(f'rank of M: {np.linalg.matrix_rank(M)}')

    # Evaluate the polynomial on a grid to plot its level sets
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 300), np.linspace(-1.2, 1.2, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # (num grid pts, 2)
    phi_grid = pf.transform(grid_points)  # (num grid pts, poly)
    zz = np.sum((phi_grid @ v[:, 1:]) ** 2 / lam[1:], axis=1)
    zz = zz.reshape(xx.shape)
    contour = plt.contourf(xx, yy, zz, levels=20, cmap='Oranges')
    plt.colorbar(contour)
    zero_contour = plt.contour(xx, yy, zz, levels=[0], colors='black', linewidths=2)

    """ zz = np.einsum('bi,ij,bj->b', 
                   phi_grid,
                   np.linalg.inv(M),
                   phi_grid) # type: ignore
    zz = zz.reshape(xx.shape)
    contour = plt.contourf(xx, yy, zz, levels=20, cmap='Oranges')
    plt.colorbar(contour)
    zero_contour = plt.contour(xx, yy, zz, levels=[0], colors='black', linewidths=2) """
    
    plt.scatter(*x.T, s=20)

    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    lam = np.linalg.eigvalsh(K)
    print(lam)
    plt.plot(range(len(lam)), lam, 'x-')
    plt.yscale('log')
    plt.show()