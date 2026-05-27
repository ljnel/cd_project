"""Exact CD polynomial for the uniform distribution on [-1, 1].

The orthonormal polynomials w.r.t. the density 1/2 on [-1, 1] are
    p_k(x) = sqrt(2k + 1) * P_k(x),
where P_k is the Legendre polynomial. The CD polynomial of order d
(degree 2d) is the reproducing kernel on the diagonal:
    K_d(x) = sum_{k=0}^{d} p_k(x)^2 = sum_{k=0}^{d} (2k + 1) * P_k(x)^2.
"""

import sympy as sp


def cd_poly_uniform(d: int, x: sp.Symbol | None = None) -> sp.Expr:
    """Return the exact CD polynomial of order d for Uniform[-1, 1]."""
    if x is None:
        x = sp.symbols("x")
    return sp.expand(sum((2 * k + 1) * sp.legendre(k, x) ** 2 for k in range(d + 1)))


if __name__ == "__main__":
    x = sp.symbols("x")
    for d in range(1, 6):
        poly = cd_poly_uniform(d, x)
        print(f"d = {d}: K_d(x) = {poly}")
        print(f"        K_d(0) = {poly.subs(x, 0)},  K_d(1) = {poly.subs(x, 1)}")
