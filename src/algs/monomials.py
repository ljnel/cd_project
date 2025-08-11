"""
This modules contains functions to generate the monomial feature vector v(x) which satisfies k(x,y) = (1 + x'y^d) = v(x)'v(y)

It is almost the same as our usual monomials, but with appropriate scaling of certain terms.

This code was adapted from Gemini.
"""

import math

import numpy as np

# For caching factorial calculations
_FACTORIAL_CACHE = {}


def factorial(n):
    """Computes factorial with caching."""
    if n not in _FACTORIAL_CACHE:
        _FACTORIAL_CACHE[n] = math.factorial(n)
    return _FACTORIAL_CACHE[n]


def _generate_exponent_tuples(n_vars, total_degree):
    """
    A recursive generator to find all non-negative integer solutions to
    j_1 + j_2 + ... + j_{n_vars} = total_degree.
    Yields tuples of exponents (j_1, ..., j_{n_vars}).
    """
    if n_vars == 1:
        yield (total_degree,)
        return

    for i in range(total_degree + 1):
        # For each possible value 'i' for the first variable's exponent,
        # find all exponent combinations for the remaining variables.
        for rest_of_exponents in _generate_exponent_tuples(
            n_vars - 1, total_degree - i
        ):
            yield (i,) + rest_of_exponents


def get_monomial(x, d):
    """
    Generates the monomial feature vector corresponding to the polynomial kernel (1 + x'y)^d.

    The feature map phi(x) is constructed such that phi(x).T @ phi(y) = (1 + x.T @ y)^d.

    Args:
        x (list or np.ndarray): The input vector of shape (n,).
        n (int): The dimension of the input vector x.
        d (int): The degree of the polynomial kernel.

    Returns:
        np.ndarray: The high-dimensional feature vector phi(x).
    """
    n = len(x)
    features = []

    # Pre-calculate factorial of d for efficiency
    fact_d = factorial(d)

    # Iterate through all total degrees from 0 to d
    for k in range(d + 1):
        # Generate all unique exponent combinations (j_1, ..., j_n) that sum to k
        exponent_tuples = _generate_exponent_tuples(n, k)

        for J in exponent_tuples:  # J is a tuple like (j_1, j_2, ..., j_n)
            # 1. Calculate the multinomial coefficient part of the scaling factor
            # The full coefficient is sqrt(d! / ((d-k)! * j_1! * ... * j_n!))
            denom_factorials = factorial(d - k)
            for j_i in J:
                denom_factorials *= factorial(j_i)

            coeff = math.sqrt(fact_d / denom_factorials)

            # 2. Calculate the monomial value: x_1^j_1 * x_2^j_2 * ... * x_n^j_n
            monomial_value = np.prod(x**J)

            # 3. Append the scaled monomial to the feature vector
            features.append(coeff * monomial_value)
    return np.array(features)


def poly_kernel(x, y, d):
    return (1 + x.T @ y) ** d


def verify_monomial_feature_map(n, d):
    """
    Tests the get_monomial function by comparing its output's dot product
    with the direct kernel calculation.
    """
    np.random.seed(42)
    x = np.random.randn(n)
    y = np.random.randn(n)

    phi_x = get_monomial(x, d)
    phi_y = get_monomial(y, d)

    dot_product_result = phi_x.T @ phi_y
    kernel_result = poly_kernel(x, y, d)
    assert np.isclose(dot_product_result, kernel_result), "Verification FAILED!"


if __name__ == "__main__":
    verify_monomial_feature_map(n=2, d=1)
    verify_monomial_feature_map(n=2, d=2)
    verify_monomial_feature_map(n=3, d=3)
    verify_monomial_feature_map(n=4, d=2)
    verify_monomial_feature_map(n=10, d=5)
    print("all tests passed")
