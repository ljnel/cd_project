from algs.bases import *
import numpy as np
import jax.numpy as jnp

def test_shape():
    bs = BasisSpec(n_vars=3, degree=4)
    x = np.random.randn(100, 3)
    X = ChebyshevBasis(bs).transform(x)
    assert X.shape == (100, 35)

def test_cheb_values():
    "Check consistency with numpy's Chebyshev polynomials."
    bs = BasisSpec(n_vars=1, degree=2)
    t = np.linspace(-1, 1, 100, dtype=np.float64)

    cheb_poly_vals = ChebyshevBasis(bs).transform(t[:, None])
    assert cheb_poly_vals.shape == (100, 3)

    expected = np.polynomial.chebyshev.chebval(t, [1, 1, 1])
    assert np.allclose(cheb_poly_vals.sum(axis=1), expected)