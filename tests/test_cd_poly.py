import numpy as np
from algs.cd_poly import CDPolynomial

def test_cd_shape():
    z = np.random.randn(1000, 2)
    p = CDPolynomial(z, degree=3)
    assert p.n_terms == 10

def test_basis_independence():
    # check independence of polynomial basis
    n = 100
    x = np.random.randn(n, 2)
    x_test = 2 * np.random.rand(n, 2) - 1
    p = CDPolynomial(x, degree=2)

    x0, x1 = x.T
    # feature map arising from the polynomial kernel
    phi = np.column_stack([np.ones(n), np.sqrt(2)*x0, np.sqrt(2)*x1,
                           x0**2, np.sqrt(2)*x0*x1, x1**2])
    assert phi.shape == (n, 6)
    kernel_mat = (np.einsum('ik,jk->ij', x, x) + 1) ** 2
    assert np.allclose(phi @ phi.T, kernel_mat)
    M = (phi.T @ phi) / n
    x0, x1 = x_test.T
    phi_test = phi = np.column_stack([np.ones(n), np.sqrt(2)*x0, np.sqrt(2)*x1,
                                      x0**2, np.sqrt(2)*x0*x1, x1**2])
    cd_poly_features = np.einsum('bi,ij,bj->b', phi_test, np.linalg.inv(M), phi_test)
    assert np.allclose(cd_poly_features, p(x_test))

def test_cd_rf():
    z = np.random.randn(1000, 5)
    p = CDPolynomial(z, degree=5, basis='rf', n_components=100)
    assert p.M.shape == (100, 100)

def test_rf_methods():
    z = np.random.randn(1000, 5)
    p = CDPolynomial(z, degree=5, basis='rf', n_components=100, method='qr')
    pp = CDPolynomial(z, degree=5, basis='rf', n_components=100, method='chol')

    z_test = np.random.randn(100, 5)
    assert np.allclose(p(z_test), pp(z_test))