from math import comb

# below is required for tests to pass -- default is float32.
import jax

jax.config.update("jax_enable_x64", True)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from sklearn.metrics.pairwise import laplacian_kernel, polynomial_kernel, rbf_kernel

matplotlib.use("QtAgg")

from algs.cd_poly import CDPolynomial, CDPolynomialKernel


def test_polynomial_kernel():
    for method in ["chol", "solve"]:
        np.random.seed(2)
        for n in range(2, 5):
            for deg in range(1, 4):
                D = comb(n + deg, deg)

                # for N = D we expect that both give the same results.
                N = D
                train_data = np.random.uniform(low=-1, high=2, size=(N, n))

                cd_poly = CDPolynomial(
                    train_data,
                    basis="mons",
                    degree=deg,
                    method=method,
                    verbose=True,
                )
                kernel = lambda x, data: polynomial_kernel(
                    x, data, degree=deg, gamma=1.0
                )
                cd_kernel = CDPolynomialKernel(
                    train_data, kernel=kernel, method=method, verbose=True
                )

                # do a certain number of random tests to verify
                for _ in range(10):
                    train_point = np.random.uniform(low=-1, high=2, size=(1, n))

                    kernel_eval = kernel(train_point, train_data)
                    for i, xi in enumerate(train_data):
                        kernel_test = (1 + train_point.flatten() @ xi) ** deg
                        np.testing.assert_allclose(kernel_eval[0, i], kernel_test)

                    value_poly = cd_poly(train_point)
                    value_kernel = cd_kernel(train_point)
                    assert abs((value_kernel - value_poly) / value_poly) < 1e-3


def plot_kernels(factor=1.0):
    np.random.seed(2)
    n = 2
    deg = 4
    D = comb(n + deg, deg)

    N = int(D * factor)
    data = np.random.uniform(low=-1, high=2, size=(N, 2))

    poly = CDPolynomial(data, degree=deg, method="solve", verbose=True)
    fig, axs = plt.subplots(1, 4)
    fig.set_size_inches(16, 4)
    ax = axs[0]

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    vals = poly.plot_map(ax, norm=SymLogNorm(1e-3))  # type: ignore
    # vals = poly.plot(ax)
    ax.scatter(*data.T, color="black", marker="x")
    # ax.set_aspect("equal")
    ax.set_title("Original CD")

    assert poly.M.shape[0] == D

    kernel_laplace = lambda x, data: laplacian_kernel(x, data, gamma=2.0)
    kernel_gauss = lambda x, data: rbf_kernel(x, data, gamma=2.0)
    kernel_poly = lambda x, data: polynomial_kernel(x, data, degree=deg, gamma=1.0)

    vals_dict = {}
    for kernel, name, ax in zip(
        [kernel_laplace, kernel_gauss, kernel_poly],
        ["Laplace", "Gaussian", "Polynomial"],
        axs[1:],
    ):
        print(f"Using {name} kernel")
        poly_kernel = CDPolynomialKernel(
            data, kernel=kernel, method="solve", verbose=True
        )

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        vals_dict[name] = poly_kernel.plot_map(ax, norm=SymLogNorm(1e-3))  # type: ignore
        ax.scatter(*data.T, color="black", marker="x")
        # ax.set_aspect("equal")
        ax.set_title(name + " kernel")

    if N == D:
        # test again that kernelized and normal form give the same resutls for polynomial.
        np.testing.assert_allclose(vals_dict["Polynomial"], vals)  # type: ignore
    print("done")
    plt.show()
    plt.show(block=True)


if __name__ == "__main__":
    # First, sanity check that both kernelized form and normal form
    # give the same results.

    test_polynomial_kernel()

    # Plot results using different kernels.
    # 1. Use N = D.
    plot_kernels(factor=1)

    # 2. Use N = D / 2
    # Note that the normal CD kernel gives no good results here,
    # while the kernelized form continues showing something reasonable.
    plot_kernels(factor=0.5)

    # 3. Use N = D * 2
    # Note that the kernelized form is ill-conditioned in this case.
    plot_kernels(factor=2.0)
    print("all tests passed.")
