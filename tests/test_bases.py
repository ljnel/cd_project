import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures

from cd.algs.bases.poly_basis import BasisSpec, ChebyshevBasis, MonomialBasis


def test_shape():
    bs = BasisSpec(n_vars=3, degree=4)
    x = torch.randn(100, 3)
    X = ChebyshevBasis(bs).transform(x)
    assert X.shape == (100, 35)

def test_mon_values():
    "Check for consistency with sklearn."
    # NB: for now, my basis is in graded-lex order...
    x = torch.randn(100, 2)
    poly = torch.from_numpy(PolynomialFeatures(degree=3).fit_transform(x)).float()
    bs = BasisSpec(n_vars=2, degree=3)
    polyy = MonomialBasis(bs).transform(x)

    assert poly.shape == polyy.shape
    assert torch.allclose(poly.sum(dim=1), polyy.sum(dim=1), atol=1e-5)

def test_cheb_values():
    "Check consistency with numpy's Chebyshev polynomials."
    bs = BasisSpec(n_vars=1, degree=2)
    n = 100
    t = torch.linspace(-1, 1, n)

    cheb_poly_vals = ChebyshevBasis(bs).transform(t[:, None])
    assert cheb_poly_vals.shape == (n, 3)

    expected = np.polynomial.chebyshev.chebval(t, [1, 1, 1])
    assert torch.allclose(cheb_poly_vals.sum(dim=1), 
                          expected)