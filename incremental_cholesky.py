import math
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as scl
import seaborn as sns


def generate_exponents(num_vars, max_degree):

    def _generate_exponents_recursive(num_vars, max_degree_rem):
        if num_vars == 1:
            return [[max_degree_rem]]

        exponents_list = []
        for i in range(max_degree_rem + 1):
            sub_exponents = _generate_exponents_recursive(
                num_vars - 1, max_degree_rem - i
            )
            for sub_exp in sub_exponents:
                exponents_list.append([i] + sub_exp)
        return exponents_list

    all_exponents = []
    for deg in range(max_degree + 1):
        all_exponents.extend(_generate_exponents_recursive(num_vars, deg)[::-1])

    # A more direct, non-recursive way to generate all exponents up to d
    # is to generate exponents for n+1 variables of degree exactly d.
    # This is left as above for clarity of degree-by-degree construction.

    return np.array(all_exponents)


def get_monomial_vector_vectorized(points, d):
    """
    A vectorized implementation to compute the moment matrix for a general
    degree d and a matrix of points (N x n).

    The moment matrix M is calculated as:
        M = V.T @ V
    where V is the matrix of monomial vectors for each point. This implementation
    avoids explicit loops over the points by using NumPy broadcasting.

    Args:
        points (np.ndarray): An N x n array of N points in n dimensions.
        d (int): The maximum degree of the monomials.

    Returns:
        np.ndarray: The (D x D) moment matrix, where D is the number of
                    monomials up to degree d.
    """
    n = points.shape[1]

    # Step 1: Generate all monomial exponents up to degree d.
    # An exponent is a vector [e_1, e_2, ..., e_n] where sum(e_i) <= d.
    # This helper function generates exponents recursively. While the function
    # itself is not vectorized, it is only run once and is significantly more
    # efficient than the method in the original code.
    exponents = generate_exponents(n, d)
    # 'exponents' is a (D, n) matrix, where D is the number of monomials.

    # Step 2: Compute the monomial matrix V in a vectorized way.
    # V will be an (N, D) matrix where V[i, j] is the j-th monomial
    # evaluated at the i-th point.

    # We use broadcasting to compute this efficiently:
    #   - points[:, np.newaxis, :] has shape (N, 1, n)
    #   - exponents has shape (D, n)
    # Broadcasting `points` against `exponents` results in an array of shape (N, D, n).
    # We then take the product over the last axis (axis=2) to get the final monomial values.
    V = np.prod(points[:, np.newaxis, :] ** exponents, axis=2)
    return V


def get_moment_matrix_vectorized(points, d):
    V = get_monomial_vector_vectorized(points, d)
    # Step 3: Compute the moment matrix.
    M = V.T @ V
    return M


def get_monomial_vector_single(vars, deg):
    n = len(vars)
    exponents = [exp for exp in product(range(deg + 1), repeat=n) if sum(exp) == deg][
        ::-1
    ]
    return np.array([np.prod(vars ** np.array(exp)) for exp in exponents])


def get_monomial_vector(vars, max_degree):
    n = len(vars)
    N_d = math.comb(n + max_degree, max_degree)
    v = np.empty(N_d)
    i = 0
    for deg in range(max_degree + 1):
        v_deg = get_monomial_vector_single(vars, deg)
        v[i : i + len(v_deg)] = v_deg
        i += len(v_deg)
    return v


def get_moment_matrix(points, d):
    """
    For general degree d and matrix of points N x n, compute the moment matrix:

    ..math:
        M = sum_i v_d(x_i) v_d(x_i).T

    where v_d is the monomial basis of degree d.
    """
    V = np.array([get_monomial_vector(p, d) for p in points])
    return V.T @ V


def get_partial_moment_matrices(dataset, degree):
    """
    Return subblocks of the moment matrix at the given degree:

    M_2 = | M_1  A_2.T |
          | A_2  B_2   |
    so A_d is the cross moment matrix between degree d and d-1, and B_d is the moment matrix at degree d.
    """

    def get_partial_single(x):
        v_1 = get_monomial_vector(x, degree - 1)
        v_2 = get_monomial_vector_single(x, degree)
        A_2 = v_2[:, None] @ v_1[None, :]
        B_2 = v_2[:, None] @ v_2[None, :]
        return A_2, B_2

    N = dataset.shape[0]
    for i in range(N):
        A_2_i, B_2_i = get_partial_single(dataset[i])
        if i == 0:
            A_2 = A_2_i
            B_2 = B_2_i
        else:
            A_2 += A_2_i
            B_2 += B_2_i

    return A_2, B_2


def resurive_choleksky(L_1, u_1, A_2, B_2, b_2):
    """
    Given the Choleksy factorization of M_1 = L_1 @ L_1.T,
    and the solution to L_1 @ u_1 = v_1,

    compute the Choleksy factorization of
    M_2 = | M_1 A_2   | = L_2 @ L_2.T
          | A_2.T B_2 |
    and solve L_2 @ u_2 = v_2, where v_2 = | v_1 |
                                           | b_2 |

    This function returns C_2, D_2, e_2, which are defined as:
    L_2 = | L_1 0   | and u_2 = | u_1 |
          | C_2 D_2 |           | e_2 |
    """
    # solve C_2 L_1.T = A_2
    C_2 = scl.solve_triangular(L_1, A_2.T, lower=True).T
    D_2 = scl.cholesky(B_2 - C_2 @ C_2.T, lower=True)
    e_2 = scl.solve_triangular(D_2, b_2 - C_2 @ u_1, lower=True)
    return C_2, D_2, e_2


def incremental_cholesky(dataset, degree, x):
    """
    Compute the cholesky factorization of the moment matrix up to a given degree,
    using the points in dataset. At each level, also solve the system L @ u(x) = v(x), where v(x) is the
    monomial basis evaluated at x, so that we can compute:
    p(x) = v(x).T @ M^{-1} @ v(x) = v(x).T @ L.T^{-1} L^{-1} v(x) = || u(x) ||^2
    """

    M_d = get_moment_matrix(dataset, 1)
    v_d = get_monomial_vector(x, 1)
    L_d = scl.cholesky(M_d, lower=True)
    u_d = scl.solve_triangular(L_d, v_d, lower=True)
    for d in range(2, degree + 1):

        b_d = get_monomial_vector_single(x, d)
        A_d, B_d = get_partial_moment_matrices(dataset, d)
        C_d, D_d, e_d = resurive_choleksky(L_d, u_d, A_d, B_d, b_d)
        L_d = np.block([[L_d, np.zeros((L_d.shape[0], D_d.shape[1]))], [C_d, D_d]])
        u_d = np.concatenate([u_d, e_d])

    return L_d, u_d


def incremental_cholesky_cached(M_d, v_d, degree, cached_A_B_b={}):
    """
    Compute the cholesky factorization of the moment matrix up to a given degree,
    using the points in dataset. At each level, also solve the system L @ u(x) = v(x), where v(x) is the
    monomial basis evaluated at x, so that we can compute:
    p(x) = v(x).T @ M^{-1} @ v(x) = v(x).T @ L.T^{-1} L^{-1} v(x) = || u(x) ||^2
    """
    L_d = scl.cholesky(M_d, lower=True)
    u_d = scl.solve_triangular(L_d, v_d, lower=True)
    for d in range(2, degree + 1):
        C_d, D_d, e_d = resurive_choleksky(L_d, u_d, *cached_A_B_b[d])
        L_d = np.block([[L_d, np.zeros((L_d.shape[0], D_d.shape[1]))], [C_d, D_d]])
        u_d = np.concatenate([u_d, e_d])
    return L_d, u_d


def cache_A_B_b(dataset, degree, x):
    cached_A_B_b = {}
    for d in range(2, degree + 1):
        b_d = get_monomial_vector_single(x, d)
        A_d, B_d = get_partial_moment_matrices(dataset, d)
        cached_A_B_b[d] = (A_d, B_d, b_d)
    return cached_A_B_b


def run_time_study(n_list, degree_list):
    import time

    data = []
    for n in n_list:
        cached_M_v = {}
        cached_A_B_b = {}
        sizes = {}
        N = math.comb(n + max(degree_list), max(degree_list)) * 2
        dataset = np.random.rand(N, n)
        x = np.random.rand(n)
        for degree in degree_list:
            print(f"============== n={n}, d={degree} ================")
            time_dict = {}
            for d in range(1, degree + 1):
                if d in cached_M_v:
                    continue
                v_d = get_monomial_vector_vectorized(x[None, :], d)[0, :]
                M_d = get_moment_matrix_vectorized(dataset, d)
                cached_M_v[d] = (M_d, v_d)
                sizes[d] = len(v_d)
                if d >= 2:
                    b_d = v_d[sizes[d - 1] :]
                    A_d = M_d[sizes[d - 1] :, : sizes[d - 1]]
                    B_d = M_d[sizes[d - 1] :, sizes[d - 1] :]
                    cached_A_B_b[d] = (A_d, B_d, b_d)

            print("Size largest moment matrix:", M_d.shape)

            t1 = time.time()
            for d in range(1, degree + 1):
                M_d, v_d = cached_M_v[d]
                L_d = scl.cholesky(M_d, lower=True)
                u_d = scl.solve_triangular(L_d, v_d, lower=True)
            t2 = time.time()
            tot1 = (t2 - t1) * 1e3
            print(f"Direct Total: {tot1:.6f} seconds")
            time_dict["Direct Cached"] = tot1

            # t1 = time.time()
            # L_d_test, u_d_test = incremental_cholesky(dataset, degree, x)
            # t2 = time.time()
            # tot2 = (t2 - t1) * 1e3
            # time_dict["Incremental Naive"] = tot2
            # print(f"Incremental Total: {(t2-t1)*1e3:.6f} seconds")

            # not doing this again cause it was easier to get the
            # submatrices from the full moment matrices.
            # cached_A_B_b = cache_A_B_b(dataset, degree, x)

            M_d, v_d = cached_M_v[1]
            t1 = time.time()
            L_d_test, u_d_test = incremental_cholesky_cached(
                M_d, v_d, degree, cached_A_B_b=cached_A_B_b
            )
            t2 = time.time()
            tot2 = (t2 - t1) * 1e3
            time_dict["Incremental Cached"] = tot2
            print(f"Incremental Total: {tot2:.6f} seconds")

            # sanity check
            for k, v in time_dict.items():
                data.append({"n": n, "degree": degree, "method": k, "time_ms": v})

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    import os

    fname = "incremental_cholesky_benchmark.csv"
    overwrite = False
    if os.path.exists(fname) and not overwrite:
        df = pd.read_csv(fname)
    else:
        n_list = np.arange(5, 21, step=5)
        degree_list = [2, 3, 4]
        # for debugging
        # n_list = [15]
        # degree_list = [3]
        df = run_time_study(n_list, degree_list)
        df.to_csv(fname, index=False)
        print("saved benchmark data to", fname)

    for d, df_here in df.groupby("degree"):
        fig, ax = plt.subplots()
        sns.lineplot(df_here, y="time_ms", x="n", hue="method")
        ax.set_yscale("log")
        ax.set_title(f"Degree {d}")

    print("Done evaluating.")
