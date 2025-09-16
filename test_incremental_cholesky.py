import numpy as np
import scipy.linalg as scl

from incremental_cholesky import (
    get_moment_matrix,
    get_moment_matrix_vectorized,
    get_monomial_vector,
    get_monomial_vector_single,
    get_partial_moment_matrices,
    incremental_cholesky,
)


def test_get_monomial_matrix():
    points = np.array([[1.0, 2], [3, 4], [5, 6]])
    M = get_moment_matrix(points, 2)

    expected_V = np.array(
        [
            [1.0, 1, 2, 1, 2, 4],
            [1.0, 3, 4, 9, 12, 16],
            [1.0, 5, 6, 25, 30, 36],
        ]
    ).T
    expected_M = expected_V @ expected_V.T

    assert np.allclose(M, expected_M), f"Expected {expected_M}, but got {M}"

    M_vec = get_moment_matrix_vectorized(points, 2)
    assert np.allclose(M_vec, expected_M), f"Expected {expected_M}, but got {M_vec}"


def test_get_monomial_vector():
    x = np.array([2.0, 3.0])

    v = get_monomial_vector(x, 0)
    np.testing.assert_allclose(np.array([1]), v)

    v = get_monomial_vector(x, 1)
    np.testing.assert_allclose(np.array([1, x[0], x[1]]), v)

    v = get_monomial_vector(x, 2)
    np.testing.assert_allclose(
        np.array([1, x[0], x[1], x[0] ** 2, x[0] * x[1], x[1] ** 2]), v
    )

    v = get_monomial_vector(x, 3)
    np.testing.assert_allclose(
        np.array(
            [
                1,
                x[0],
                x[1],
                x[0] ** 2,
                x[0] * x[1],
                x[1] ** 2,
                x[0] ** 3,
                x[0] ** 2 * x[1],
                x[0] * x[1] ** 2,
                x[1] ** 3,
            ]
        ),
        v,
    )


def test_get_monomial_vector_single():
    x = np.array([2.0, 3.0])

    v = get_monomial_vector_single(x, 0)
    np.testing.assert_allclose(np.array([1]), v)

    v = get_monomial_vector_single(x, 1)
    np.testing.assert_allclose(np.array([x[0], x[1]]), v)

    v = get_monomial_vector_single(x, 2)
    np.testing.assert_allclose(np.array([x[0] ** 2, x[0] * x[1], x[1] ** 2]), v)

    v = get_monomial_vector_single(x, 3)
    np.testing.assert_allclose(
        np.array(
            [
                x[0] ** 3,
                x[0] ** 2 * x[1],
                x[0] * x[1] ** 2,
                x[1] ** 3,
            ]
        ),
        v,
    )


def test_get_partial_moment_matrices():
    dataset = np.array([[1.0, 2], [3, 4], [5, 6]])
    x = np.array([2.0, 3.0])
    degree = 1

    M_3 = get_moment_matrix(dataset, degree + 1)

    A_3, B_3 = get_partial_moment_matrices(dataset, degree + 1)

    M_2 = get_moment_matrix(dataset, degree)

    M_3_test = np.block(
        [
            [M_2, A_3.T],
            [A_3, B_3],
        ]
    )
    np.testing.assert_allclose(M_3_test, M_3)


def test_incremental_cholesky():
    n = 2
    degree = 3

    dataset = np.random.rand(30, n)
    x = np.random.rand(n)

    M_d = get_moment_matrix(dataset, degree)
    v_d = get_monomial_vector(x, degree)

    L_d = scl.cholesky(M_d, lower=True)
    u_d = scl.solve_triangular(L_d, v_d, lower=True)

    L_d_test, u_d_test = incremental_cholesky(dataset, degree, x)
    np.testing.assert_allclose(L_d_test, L_d)
    np.testing.assert_allclose(u_d_test, u_d)


if __name__ == "__main__":
    test_get_monomial_vector_single()
    test_get_monomial_vector()
    test_get_monomial_matrix()

    test_get_partial_moment_matrices()

    test_incremental_cholesky()

    print("All tests passed.")
